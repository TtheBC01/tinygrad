// Flash decoding for the LLM's fp16 KV cache (one query token): attention of each query head over keys [0, valid_end), split
// across NSPLIT blocks per KV head so the whole GPU streams the cache, then a second kernel merges the partial softmaxes.
// Decode is memory bound, so this uses plain FMA: a warp takes one key at a time, each lane owning HD/32 head dims.
// The G = NH/NKV query heads of a KV head share each K/V row read (GQA).
// nv.py prepends nv_common.cuh and #defines PARTIAL or COMBINE, KNAME NB NH NKV NCTX HD NSPLIT.

#define NEG_INF __int_as_float(0xff800000)  // -inf (nvrtc --minimal has no math.h INFINITY)
#define G (NH / NKV)
#define DPL (HD / 32)  // head dims per lane
#define NWARPS 4
#define PSTRIDE (HD + 2)  // a partial: o[HD] (scaled by 2^-m), then m (log2 units), then l
static_assert(HD % 64 == 0 && NH % NKV == 0, "head dim must be 64 or 128");

__device__ __forceinline__ void load_row(float *f, const half *p) {  // DPL consecutive halves
#if DPL == 4
  uint2 u = *(const uint2 *)p;
  float2 a = __half22float2(*(half2 *)&u.x), b = __half22float2(*(half2 *)&u.y);
  f[0] = a.x, f[1] = a.y, f[2] = b.x, f[3] = b.y;
#else
  float2 a = __half22float2(*(const half2 *)p);
  f[0] = a.x, f[1] = a.y;
#endif
}

#ifdef PARTIAL
// grid (NSPLIT, NB * NKV), 128 threads. P: [NB][NH][NSPLIT][PSTRIDE] fp32. Q: [NB][NH][HD] fp16.
extern "C" __global__ void __launch_bounds__(32 * NWARPS)
KNAME(float *__restrict__ P, const half *__restrict__ Q, const half *__restrict__ KV, const int *__restrict__ params) {
  __shared__ float sm_m[NWARPS][G], sm_l[NWARPS][G], sm_o[NWARPS][G][HD];
  const int split = blockIdx.x, b = blockIdx.y / NKV, kvh = blockIdx.y % NKV, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int valid_end = params[0], per = (valid_end + NSPLIT - 1) / NSPLIT;
  const int k0 = split * per, k1 = min(valid_end, k0 + per);
  const float scale = 1.4426950408889634f * rsqrtf((float)HD);  // log2(e)/sqrt(HD): softmax via exp2

  float q[G][DPL], o[G][DPL], m[G], l[G];
#pragma unroll
  for (int g = 0; g < G; g++) {
    load_row(q[g], Q + ((size_t)b * NH + kvh * G + g) * HD + lane * DPL);
#pragma unroll
    for (int i = 0; i < DPL; i++) q[g][i] *= scale, o[g][i] = 0.0f;
    m[g] = NEG_INF, l[g] = 0.0f;
  }
  const half *kp = KV + ((size_t)(0 * NB + b) * NKV + kvh) * NCTX * HD + lane * DPL;
  const half *vp = KV + ((size_t)(1 * NB + b) * NKV + kvh) * NCTX * HD + lane * DPL;
  for (int k = k0 + warp; k < k1; k += NWARPS) {
    float kf[DPL], vf[DPL];
    load_row(kf, kp + (size_t)k * HD);
    load_row(vf, vp + (size_t)k * HD);
#pragma unroll
    for (int g = 0; g < G; g++) {
      float s = 0.0f;
#pragma unroll
      for (int i = 0; i < DPL; i++) s += q[g][i] * kf[i];
#pragma unroll
      for (int off = 16; off; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
      const float mn = fmaxf(m[g], s), a = exp2f(m[g] - mn), p = exp2f(s - mn);
      l[g] = l[g] * a + p;
#pragma unroll
      for (int i = 0; i < DPL; i++) o[g][i] = o[g][i] * a + p * vf[i];
      m[g] = mn;
    }
  }
  // merge the 4 warps
#pragma unroll
  for (int g = 0; g < G; g++) {
    if (lane == 0) sm_m[warp][g] = m[g], sm_l[warp][g] = l[g];
#pragma unroll
    for (int i = 0; i < DPL; i++) sm_o[warp][g][lane * DPL + i] = o[g][i];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < G * HD; idx += 32 * NWARPS) {
    const int g = idx / HD, d = idx % HD;
    float M = NEG_INF;
#pragma unroll
    for (int w = 0; w < NWARPS; w++) M = fmaxf(M, sm_m[w][g]);
    float L = 0.0f, Od = 0.0f;
    if (M != NEG_INF) {  // an empty split (valid_end < NSPLIT * per) has no keys
#pragma unroll
      for (int w = 0; w < NWARPS; w++) {
        const float s = exp2f(sm_m[w][g] - M);
        L += sm_l[w][g] * s, Od += sm_o[w][g][d] * s;
      }
    }
    float *pp = P + (((size_t)b * NH + kvh * G + g) * NSPLIT + split) * PSTRIDE;
    pp[d] = Od;
    if (d == 0) pp[HD] = M, pp[HD + 1] = L;
  }
}
#endif

#ifdef COMBINE
// grid (NB * NH), HD threads. O: [NB][NH][HD] fp32 (one token).
extern "C" __global__ void __launch_bounds__(HD) KNAME(float *__restrict__ O, const float *__restrict__ P) {
  const float *pp = P + (size_t)blockIdx.x * NSPLIT * PSTRIDE;
  const int d = threadIdx.x;
  float M = NEG_INF;
  for (int s = 0; s < NSPLIT; s++) M = fmaxf(M, pp[s * PSTRIDE + HD]);
  float L = 0.0f, Od = 0.0f;
  for (int s = 0; s < NSPLIT; s++) {
    const float ms = pp[s * PSTRIDE + HD];
    if (ms == NEG_INF) continue;
    const float w = exp2f(ms - M);
    L += pp[s * PSTRIDE + HD + 1] * w, Od += pp[s * PSTRIDE + d] * w;
  }
  O[(size_t)blockIdx.x * HD + d] = Od / L;
}
#endif
