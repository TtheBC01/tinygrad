// Decode GEMVs on GGUF quantized weights (Q4_K, Q6_K, Q8_0): out[t][n] = sum_k x[t][k] * W[n][k] for a few tokens.
// Like llama.cpp's mmvq: the activations are first quantized to int8 in blocks of 32 (QUANTIZE: per block a scale d and
// d * sum(q)), then the GEMV takes integer dot products of 4 weights x 4 activations at a time (dp4a) and applies the scales in fp32.
// Decode is memory bound: each warp streams whole quant blocks with lanes on consecutive words, the activations are re-read from L1.
// nv.py prepends nv_common.cuh and #defines QUANTIZE or GEMV, KNAME, K_DIM, and for GEMV: QTYPE N_DIM NTOK KSPLIT.

// activations for one token, int32 words: K/4 words of int8 values, then K/32 (d, d * sum q) float pairs
#define XWORDS (K_DIM / 4 + K_DIM / 16)

#ifdef QUANTIZE
// grid (NTOK * K / 128), 128 threads: one warp per block of 32 values. X: [NTOK][K] fp32 -> XQ: [NTOK][XWORDS]
extern "C" __global__ void __launch_bounds__(128) KNAME(int *__restrict__ XQ, const float *__restrict__ X) {
  const int g = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;  // global block of 32
  const int t = g / (K_DIM / 32), b = g % (K_DIM / 32);
  const float v = X[(size_t)t * K_DIM + b * 32 + lane];
  float amax = fabsf(v);
#pragma unroll
  for (int off = 16; off; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
  const float d = amax / 127.0f;
  const int q = d == 0.0f ? 0 : __float2int_rn(v / d);
  int sum = q;
#pragma unroll
  for (int off = 16; off; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
  int *xt = XQ + (size_t)t * XWORDS;
  ((signed char *)xt)[b * 32 + lane] = (signed char)q;
  if (lane == 0) ((float2 *)(xt + K_DIM / 4))[b] = make_float2(d, d * (float)sum);
}
#endif

#ifdef GEMV
#define NWARPS 4
#define ROWS (NWARPS / KSPLIT)  // output rows per block; KSPLIT warps split each row's K
// Every format is read 16 bytes per lane (32 weights): a warp step covers 1024 weights (Q4_K, Q6_K: 4 superblocks of 8 lanes)
// or 512 (Q8_0: 16 blocks of 2 lanes), and the step loop runs over STEPS of them.
#if QTYPE == 12      // Q4_K: 256 weights in 144 bytes: d, dmin (fp16), 12 bytes of 6-bit scales/mins, 128 bytes of nibbles
#define BLOCK 256
#define BBYTES 144
#define BPS 4        // blocks per warp step
#elif QTYPE == 14    // Q6_K: 256 weights in 210 bytes: 128 bytes low nibbles, 64 bytes high 2 bits, 16 int8 scales, d (fp16)
#define BLOCK 256
#define BBYTES 210
#define BPS 4
#elif QTYPE == 8     // Q8_0: 32 weights in 34 bytes: d (fp16), 32 int8
#define BLOCK 32
#define BBYTES 34
#define BPS 16
#else
#error "unsupported QTYPE"
#endif
#define NBLOCKS (K_DIM / BLOCK)
#define STEPS ((NBLOCKS + BPS - 1) / BPS)
#define ROW_BYTES ((size_t)NBLOCKS * BBYTES)
static_assert(K_DIM % BLOCK == 0, "K must be a multiple of the quant block");
// Q4_K blocks are 16-byte aligned (nv.py checks the buffer offset); Q6_K/Q8_0 blocks only 2-byte, read with ld16 below, which
// needs 8-byte aligned rows
static_assert(QTYPE == 12 || ROW_BYTES % 8 == 0, "rows must be 8-byte aligned");

__device__ __forceinline__ uint4 ld16(const unsigned char *a) {
  // 16 bytes at a 2-byte aligned address: three aligned 64-bit loads, shifted into place. The third load may read up to 8 bytes
  // past the 16, which stays inside the block for every use here (none of them ends a block).
  const unsigned long long *p = (const unsigned long long *)((size_t)a & ~(size_t)7);
  const int sh = ((int)(size_t)a & 7) * 8;
  const unsigned long long w0 = p[0], w1 = p[1], w2 = p[2];
  const unsigned long long lo = sh ? (w0 >> sh) | (w1 << (64 - sh)) : w0, hi = sh ? (w1 >> sh) | (w2 << (64 - sh)) : w1;
  return make_uint4((unsigned)lo, (unsigned)(lo >> 32), (unsigned)hi, (unsigned)(hi >> 32));
}
__device__ __forceinline__ float ld_half(const unsigned char *p) { return __half2float(*(const half *)p); }
__device__ __forceinline__ int dp16(uint4 q, int4 x, int acc) {  // 16 int8 x 16 int8
  return __dp4a((int)q.w, x.w, __dp4a((int)q.z, x.z, __dp4a((int)q.y, x.y, __dp4a((int)q.x, x.x, acc))));
}
__device__ __forceinline__ uint4 and4(uint4 a, unsigned m) { return make_uint4(a.x & m, a.y & m, a.z & m, a.w & m); }
__device__ __forceinline__ uint4 shr4(uint4 a, int s) { return make_uint4(a.x >> s, a.y >> s, a.z >> s, a.w >> s); }

// grid (ceil(N / ROWS)), NWARPS * 32 threads. OUT: [NTOK][N] fp32. W: [N][ROW_BYTES]. XQ: [NTOK][XWORDS] from QUANTIZE.
extern "C" __global__ void __launch_bounds__(NWARPS * 32)
KNAME(float *__restrict__ OUT, const unsigned char *__restrict__ W, const int *__restrict__ XQ) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = blockIdx.x * ROWS + warp / KSPLIT, ks = warp % KSPLIT;
  float acc[NTOK];
#pragma unroll
  for (int t = 0; t < NTOK; t++) acc[t] = 0.0f;
  if (row < N_DIM) {
    const unsigned char *wr = W + (size_t)row * ROW_BYTES;
#pragma unroll 4
    for (int st = ks; st < STEPS; st += KSPLIT) {
#if QTYPE == 12
      // lane: 64-weight chunk j of its superblock (32 bytes: low nibbles = sub-block 2j, high = 2j+1), half p of it
      const int ub = st * BPS + (lane >> 3), j = (lane >> 1) & 3, p = lane & 1;
      if (ub < NBLOCKS) {
        const unsigned char *sb = wr + (size_t)ub * BBYTES;
        const uint4 hd = *(const uint4 *)sb, qs = *(const uint4 *)(sb + 16 + 32 * j + 16 * p);  // d, dmin, scales | nibbles
        const float d = __half2float(__ushort_as_half((unsigned short)(hd.x & 0xffff))),
                    dmin = __half2float(__ushort_as_half((unsigned short)(hd.x >> 16)));
        // 6-bit scale and min of sub-blocks 2j and 2j+1 (ggml get_scale_min_k4): bytes 0-3 scales, 4-7 mins, 8-11 their top bits
        float scl[2], mn[2];
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int sbk = 2 * j + h, sh = 8 * (sbk & 3);  // byte sbk & 3 of each scale word
          const unsigned b0 = (hd.y >> sh) & 0xff, b1 = (hd.z >> sh) & 0xff, b2 = (hd.w >> sh) & 0xff;
          const unsigned s = sbk < 4 ? b0 & 63 : (b2 & 0xF) | ((b0 >> 6) << 4), m = sbk < 4 ? b1 & 63 : (b2 >> 4) | ((b1 >> 6) << 4);
          scl[h] = d * (float)s, mn[h] = dmin * (float)m;
        }
        const uint4 lo = and4(qs, 0x0F0F0F0F), hi = and4(shr4(qs, 4), 0x0F0F0F0F);
        const int e = ub * 256 + 64 * j + 16 * p;  // first element of the low nibbles; the high ones are 32 later
#pragma unroll
        for (int t = 0; t < NTOK; t++) {
          const int *xt = XQ + (size_t)t * XWORDS;
          const float4 ds = *(const float4 *)(xt + K_DIM / 4 + 2 * (e / 32));  // (d, d*sum) of sub-blocks 2j and 2j+1
          float r = scl[0] * ds.x * (float)dp16(lo, *(const int4 *)(xt + e / 4), 0) +
                    scl[1] * ds.z * (float)dp16(hi, *(const int4 *)(xt + e / 4 + 8), 0);
          if (p == 0) r -= mn[0] * ds.y + mn[1] * ds.w;  // the mins: once per sub-block
          acc[t] += r;
        }
      }
#elif QTYPE == 14
      // lane: half h (128 weights) of its superblock, 16 of that half's 64 low-nibble bytes at b: low nibbles are weights b..,
      // high ones 64+b..; their 2 high bits are in qh bytes b % 32.. at shift 2 * (b / 32), and 4 more for the high nibbles
      const int ub = st * BPS + (lane >> 3), h = (lane >> 2) & 1, b = (lane & 3) * 16;
      if (ub < NBLOCKS) {
        const unsigned char *sb = wr + (size_t)ub * BBYTES;
        const uint4 ql = ld16(sb + 64 * h + b), qh = ld16(sb + 128 + 32 * h + (b & 31));
        const signed char *scs = (const signed char *)(sb + 192 + 8 * h);
        const float d = ld_half(sb + 208), sc_lo = d * (float)scs[b >> 4], sc_hi = d * (float)scs[4 + (b >> 4)];
        const int sh = 2 * (b >> 5);
        const uint4 vl = and4(ql, 0x0F0F0F0F), vh = and4(shr4(ql, 4), 0x0F0F0F0F);
        const uint4 hl = and4(shr4(qh, sh), 0x03030303), hh = and4(shr4(qh, sh + 4), 0x03030303);
        uint4 q_lo, q_hi;  // 6-bit values minus 32, as int8
        q_lo.x = __vsubss4(vl.x | (hl.x << 4), 0x20202020), q_lo.y = __vsubss4(vl.y | (hl.y << 4), 0x20202020);
        q_lo.z = __vsubss4(vl.z | (hl.z << 4), 0x20202020), q_lo.w = __vsubss4(vl.w | (hl.w << 4), 0x20202020);
        q_hi.x = __vsubss4(vh.x | (hh.x << 4), 0x20202020), q_hi.y = __vsubss4(vh.y | (hh.y << 4), 0x20202020);
        q_hi.z = __vsubss4(vh.z | (hh.z << 4), 0x20202020), q_hi.w = __vsubss4(vh.w | (hh.w << 4), 0x20202020);
        const int e = ub * 256 + 128 * h + b;
#pragma unroll
        for (int t = 0; t < NTOK; t++) {
          const int *xt = XQ + (size_t)t * XWORDS;
          const float2 *ds = (const float2 *)(xt + K_DIM / 4);
          acc[t] += sc_lo * ds[e / 32].x * (float)dp16(q_lo, *(const int4 *)(xt + e / 4), 0) +
                    sc_hi * ds[e / 32 + 2].x * (float)dp16(q_hi, *(const int4 *)(xt + e / 4 + 16), 0);
        }
      }
#elif QTYPE == 8
      // lane: half p (16 int8) of block ub
      const int ub = st * BPS + (lane >> 1), p = lane & 1;
      if (ub < NBLOCKS) {
        const unsigned char *qb = wr + (size_t)ub * BBYTES;
        const uint4 q = ld16(qb + 2 + 16 * p);
        const float d = ld_half(qb);
        const int e = ub * 32 + 16 * p;
#pragma unroll
        for (int t = 0; t < NTOK; t++) {
          const int *xt = XQ + (size_t)t * XWORDS;
          acc[t] += d * ((const float2 *)(xt + K_DIM / 4))[ub].x * (float)dp16(q, *(const int4 *)(xt + e / 4), 0);
        }
      }
#endif
    }
  }
#pragma unroll
  for (int t = 0; t < NTOK; t++)
#pragma unroll
    for (int off = 16; off; off >>= 1) acc[t] += __shfl_xor_sync(0xffffffff, acc[t], off);
#if KSPLIT > 1
  __shared__ float part[NWARPS][NTOK];
  if (lane == 0)
#pragma unroll
    for (int t = 0; t < NTOK; t++) part[warp][t] = acc[t];
  __syncthreads();
  if (ks == 0 && lane == 0)
#pragma unroll
    for (int t = 0; t < NTOK; t++) {
#pragma unroll
      for (int s = 1; s < KSPLIT; s++) acc[t] += part[warp + s][t];
    }
#endif
  if (ks == 0 && lane == 0 && row < N_DIM)
#pragma unroll
    for (int t = 0; t < NTOK; t++) OUT[(size_t)t * N_DIM + row] = acc[t];
}
#endif

#ifdef DEQUANT
// Prefill on quant weights: dequantize to fp16 for the tensor core GEMM. DEP is the layer's input, unused: it makes this kernel
// wait for the layer, so each layer's fp16 copy lives only until its GEMM (else the scheduler can run every layer's dequantize
// first, and they'd all be in memory at once). grid (N * K / 8 / 128), 128 threads: 8 weights (one 16-byte store) per thread.
#if QTYPE == 12
#define BLOCK 256
#define BBYTES 144
#elif QTYPE == 14
#define BLOCK 256
#define BBYTES 210
#elif QTYPE == 8
#define BLOCK 32
#define BBYTES 34
#endif
extern "C" __global__ void __launch_bounds__(128) KNAME(half *__restrict__ OUT, const unsigned char *__restrict__ W, const void *DEP) {
  const size_t i8 = (size_t)blockIdx.x * 128 + threadIdx.x;  // group of 8 weights
  const size_t blk = i8 * 8 / BLOCK;
  const int e0 = (int)(i8 * 8 % BLOCK);
  const unsigned char *b = W + blk * BBYTES;
  float v[8];
#if QTYPE == 12
  // weights e0..e0+7: chunk j = e0 / 64 of 32 qs bytes, low nibbles (sub-block 2j) for the first 32, high (2j+1) for the rest
  const int j = e0 / 64, hi = (e0 % 64) >= 32, l0 = e0 % 32, sbk = 2 * j + hi;
  const float d = __half2float(*(const half *)b), dmin = __half2float(*(const half *)(b + 2));
  const unsigned char *sc = b + 4;
  const int s = sbk < 4 ? sc[sbk] & 63 : (sc[sbk + 4] & 0xF) | ((sc[sbk - 4] >> 6) << 4);
  const int m = sbk < 4 ? sc[sbk + 4] & 63 : (sc[sbk + 4] >> 4) | ((sc[sbk] >> 6) << 4);
#pragma unroll
  for (int i = 0; i < 8; i++) v[i] = d * s * (float)((b[16 + 32 * j + l0 + i] >> (4 * hi)) & 0xF) - dmin * m;
#elif QTYPE == 14
  // weights e0..: half h, quadrant qd of 32: low/high nibble of ql[64h + 32 (qd & 1) + l], 2 bits of qh[32h + l] at shift 2 qd
  const int h = e0 / 128, t = e0 % 128, qd = t / 32, l0 = t % 32;
  const float d = __half2float(*(const half *)(b + 208)) * (float)((const signed char *)b)[192 + 8 * h + t / 16];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int lo = (b[64 * h + 32 * (qd & 1) + l0 + i] >> (4 * (qd >> 1))) & 0xF, hb = (b[128 + 32 * h + l0 + i] >> (2 * qd)) & 3;
    v[i] = d * (float)((lo | (hb << 4)) - 32);
  }
#elif QTYPE == 8
  const float d = __half2float(*(const half *)b);
#pragma unroll
  for (int i = 0; i < 8; i++) v[i] = d * (float)((const signed char *)b)[2 + e0 + i];
#endif
  half2 o[4];
#pragma unroll
  for (int i = 0; i < 4; i++) o[i] = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
  *(uint4 *)(OUT + i8 * 8) = *(uint4 *)o;
}
#endif
