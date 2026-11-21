// Flash decoding for the LLM's fp16 KV cache (one query token): attention of each query head over keys [0, valid_end), split
// across NSPLIT blocks per KV head so the whole GPU streams the cache, then the partial softmaxes are merged (COMBINE here, or the
// output projection's activation quantization, nv_quant.cu). Decode is memory bound, so this uses plain FMA. Each key is taken by
// LPK = HD/8 lanes, 8 head dims (one 16-byte load) per lane, so a warp holds 32/LPK keys per step and UNR steps' loads are issued
// before any is used: enough bytes in flight to stream the cache (one key per warp at a time was latency bound, ~300 GB/s).
// The G = NH/NKV query heads of a KV head share each K/V row read (GQA).
// nv.py prepends nv_common.cuh and nv_decode_merge.cuh and #defines PARTIAL or COMBINE, KNAME NB NH NKV NCTX HD NSPLIT NWARPS UNR MINKEYS.

#define G (NH / NKV)
#define LPK (HD / 8)             // lanes per key
#define KPW (32 / LPK)           // keys per warp per step
#define PSTRIDE (HD + 2)
static_assert((HD == 64 || HD == 128) && NH % NKV == 0, "head dim must be 64 or 128");
static_assert(MINKEYS == NWARPS * KPW * UNR, "MINKEYS: keys per block per iteration, the fewest a split gets");

__device__ __forceinline__ void load8(float (&f)[8], const half *p) {  // 8 consecutive halves
  const uint4 u = *(const uint4 *)p;
  const half2 *h = (const half2 *)&u;
#pragma unroll
  for (int i = 0; i < 4; i++) { const float2 a = __half22float2(h[i]); f[2 * i] = a.x, f[2 * i + 1] = a.y; }
}

#ifdef PARTIAL
static_assert(sizeof(float) * NWARPS * G * (HD + 2) <= 48 * 1024, "too many query heads per KV head for shared memory");
// grid (NSPLIT, NB * NKV), NWARPS * 32 threads. P: [NB][NH][NSPLIT][PSTRIDE] fp32 (only the non-empty splits are written).
// Q: [NB][NH][HD] fp16. params[0] = valid_end.
extern "C" __global__ void __launch_bounds__(32 * NWARPS)
KNAME(float *__restrict__ P, const half *__restrict__ Q, const half *__restrict__ KV, const int *__restrict__ params) {
  __shared__ float sm_m[NWARPS][G], sm_l[NWARPS][G], sm_o[NWARPS][G][HD];
  const int split = blockIdx.x, b = blockIdx.y / NKV, kvh = blockIdx.y % NKV, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int sub = lane / LPK, dl = lane % LPK;  // which of the warp's keys, which 8 dims
  const int valid_end = params[0];
  float q[G][8], o[G][8], m[G], l[G];
#pragma unroll
  for (int g = 0; g < G; g++) load8(q[g], Q + ((size_t)b * NH + kvh * G + g) * HD + dl * 8);  // in flight with params
  const int per = decode_split_keys(valid_end, NSPLIT, MINKEYS), k0 = split * per, k1 = min(valid_end, k0 + per);
  if (k0 >= k1) return;  // an empty split: the merge reads only the splits that have keys
  const float scale = 1.4426950408889634f * rsqrtf((float)HD);  // log2(e)/sqrt(HD): softmax via exp2
#pragma unroll
  for (int g = 0; g < G; g++) {
#pragma unroll
    for (int i = 0; i < 8; i++) q[g][i] *= scale, o[g][i] = 0.0f;
    m[g] = NEG_INF, l[g] = 0.0f;
  }
  const half *kp = KV + ((size_t)(0 * NB + b) * NKV + kvh) * NCTX * HD + dl * 8;
  const half *vp = KV + ((size_t)(1 * NB + b) * NKV + kvh) * NCTX * HD + dl * 8;
  for (int kb = k0 + warp * KPW * UNR; kb < k1; kb += MINKEYS) {  // warp-uniform: the shuffles below see every lane
    float kf[UNR][8], vf[UNR][8];
    bool ok[UNR];
#pragma unroll
    for (int u = 0; u < UNR; u++) {
      const int k = kb + u * KPW + sub;
      ok[u] = k < k1;
      const size_t off = (size_t)(ok[u] ? k : k0) * HD;  // a valid row either way: no branch around the loads
      load8(kf[u], kp + off);
      load8(vf[u], vp + off);
    }
#pragma unroll
    for (int g = 0; g < G; g++) {
      float s[UNR];
#pragma unroll
      for (int u = 0; u < UNR; u++) {
        s[u] = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; i++) s[u] += q[g][i] * kf[u][i];
      }
#pragma unroll
      for (int off = LPK / 2; off; off >>= 1)
#pragma unroll
        for (int u = 0; u < UNR; u++) s[u] += __shfl_xor_sync(0xffffffff, s[u], off);
      float mn = m[g];
#pragma unroll
      for (int u = 0; u < UNR; u++) if (ok[u]) mn = fmaxf(mn, s[u]);
      if (mn == NEG_INF) continue;  // no keys yet for this lane group
      const float a = exp2f(m[g] - mn);
#pragma unroll
      for (int i = 0; i < 8; i++) o[g][i] *= a;
      l[g] *= a;
#pragma unroll
      for (int u = 0; u < UNR; u++) {
        const float p = ok[u] ? exp2f(s[u] - mn) : 0.0f;
        l[g] += p;
#pragma unroll
        for (int i = 0; i < 8; i++) o[g][i] += p * vf[u][i];
      }
      m[g] = mn;
    }
  }
  // merge the warp's key groups (lanes LPK apart), then the warps through shared memory
#pragma unroll
  for (int g = 0; g < G; g++) {
#pragma unroll
    for (int off = LPK; off < 32; off <<= 1) {
      const float m2 = __shfl_xor_sync(0xffffffff, m[g], off), l2 = __shfl_xor_sync(0xffffffff, l[g], off);
      const float M = fmaxf(m[g], m2);
      const float a = M == NEG_INF ? 0.0f : exp2f(m[g] - M), a2 = M == NEG_INF ? 0.0f : exp2f(m2 - M);
#pragma unroll
      for (int i = 0; i < 8; i++) o[g][i] = o[g][i] * a + __shfl_xor_sync(0xffffffff, o[g][i], off) * a2;
      l[g] = l[g] * a + l2 * a2, m[g] = M;
    }
    if (lane == 0) sm_m[warp][g] = m[g], sm_l[warp][g] = l[g];
    if (sub == 0) {
#pragma unroll
      for (int i = 0; i < 8; i++) sm_o[warp][g][dl * 8 + i] = o[g][i];
    }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < G * HD; idx += 32 * NWARPS) {
    const int g = idx / HD, d = idx % HD;
    float M = NEG_INF;
#pragma unroll
    for (int w = 0; w < NWARPS; w++) M = fmaxf(M, sm_m[w][g]);  // finite: the split has keys
    float L = 0.0f, Od = 0.0f;
#pragma unroll
    for (int w = 0; w < NWARPS; w++) {
      const float s = exp2f(sm_m[w][g] - M);  // 0 for a warp that had no keys
      L += sm_l[w][g] * s, Od += sm_o[w][g][d] * s;
    }
    float *pp = P + (((size_t)b * NH + kvh * G + g) * NSPLIT + split) * PSTRIDE;
    pp[d] = Od;
    if (d == 0) pp[HD] = M, pp[HD + 1] = L;
  }
}
#endif

#ifdef COMBINE
// grid (NB * NH), HD threads: warp w merges head dims w*32 .. w*32+31. O: [NB][NH][HD] fp32 (one token). params[0] = valid_end.
extern "C" __global__ void __launch_bounds__(HD) KNAME(float *__restrict__ O, const float *__restrict__ P, const int *__restrict__ params) {
  const int ns = decode_active_splits(params[0], NSPLIT, MINKEYS), d = threadIdx.x;
  O[(size_t)blockIdx.x * HD + d] = decode_merge<NSPLIT>(P + (size_t)blockIdx.x * NSPLIT * PSTRIDE, ns, HD, d & 31, d);
}
#endif
