// Decode GEMVs on GGUF quantized weights (Q4_K, Q6_K, Q8_0): out[t][n] = sum_k x[t][k] * W[n][k] for a few tokens.
// Like llama.cpp's mmvq: the activations are first quantized to int8 in blocks of 32 (QUANTIZE: per block a scale d and
// d * sum(q)), then the GEMV takes integer dot products of 4 weights x 4 activations at a time (dp4a) and applies the scales in fp32.
// Decode is memory bound: each warp streams whole quant blocks with lanes on consecutive words, the activations are re-read from L1.
// One GEMV (or DEQUANT) launch can cover several weights that share the input (q/k/v, gate/up), each its own quant type: their
// outputs are concatenated. nv.py prepends nv_common.cuh and #defines QUANTIZE, GEMV or DEQUANT, KNAME, K_DIM, and the
// segments: SEG_PARAMS (the weight pointers W0, W1, ...) and SEG_DISPATCH (which weight a block or thread works on).
// activations for one token, int32 words: K/4 words of int8 values, then K/32 (d, d * sum q) float pairs
#define XWORDS (K_DIM / 4 + K_DIM / 16)

#ifdef QUANTIZE
// grid (NTOK * K / 128), 128 threads: one warp per block of 32 values. X: [NTOK][K] fp32 -> XQ: [NTOK][XWORDS]
// With GLU, X is a fused gate/up GEMV's output [NTOK][2K] and the values are the gated activation (model.gated_activation)
// g * sigmoid(GLU_ALPHA g) * (u + GLU_UP_BIAS), g and u first clamped when GLU_LIMIT is defined.
// With NORM, X ([NTOK][K] XTYPE) is the input of an RMSNorm with weight NW ([K] WTYPE) and the values are its output,
// x * rsqrt(mean(x^2) + NORM_EPS) * WLOAD(w) (WLOAD can round w as the model's weight dtype): each block sums its token's squares
// itself (re-reading the row from L2).
// With ATTN, X is flash decoding's partials ([NTOK][K/HD heads][NSPLIT][HD + 2], nv_decode.cu's P) and params[0] its valid_end:
// the values are the merged attention output (decode_merge), which is never written out.
#ifndef XTYPE
#define XTYPE float
#endif
#ifdef NORM
#define EXTRA_PARAM , const WTYPE *__restrict__ NW
#ifndef WLOAD
#define WLOAD(w) ((float)(w))
#endif
#elif defined(ATTN)
#define EXTRA_PARAM , const int *__restrict__ params
#else
#define EXTRA_PARAM
#endif
extern "C" __global__ void __launch_bounds__(128) KNAME(int *__restrict__ XQ, const XTYPE *__restrict__ X EXTRA_PARAM) {
  const int g = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;  // global block of 32
  const int t = g / (K_DIM / 32), b = g % (K_DIM / 32);  // K % 128 == 0: the block's 4 warps share t
#ifdef GLU
  float gt = X[(size_t)t * 2 * K_DIM + b * 32 + lane], up = X[(size_t)t * 2 * K_DIM + K_DIM + b * 32 + lane];
#ifdef GLU_LIMIT
  gt = fminf(gt, GLU_LIMIT), up = fminf(fmaxf(up, -GLU_LIMIT), GLU_LIMIT);
#endif
  const float v = gt / (1.0f + __expf(-GLU_ALPHA * gt)) * (up + GLU_UP_BIAS);
#elif defined(NORM)
  const XTYPE *xr = X + (size_t)t * K_DIM;
  float ss = 0.0f;
#pragma unroll 4
  for (int i = threadIdx.x; i < K_DIM; i += 128) { const float a = (float)xr[i]; ss += a * a; }
#pragma unroll
  for (int off = 16; off; off >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, off);
  __shared__ float wss[4];
  if (lane == 0) wss[threadIdx.x >> 5] = ss;
  __syncthreads();
  const float v = (float)xr[b * 32 + lane] * rsqrtf((wss[0] + wss[1] + wss[2] + wss[3]) / (float)K_DIM + NORM_EPS) * WLOAD(NW[b * 32 + lane]);
#elif defined(ATTN)
  static_assert(HD % 32 == 0, "a block of 32 values lies in one head");
  const int h = b * 32 / HD;
  // row t is token t % NT of its sequence, at position valid_end - NT + t % NT
  const float v = decode_merge<NSPLIT>(X + ((size_t)t * (K_DIM / HD) + h) * NSPLIT * (HD + 2),
                                       decode_active_splits(params[0] - NT + 1 + t % NT, NSPLIT, MINKEYS), HD, lane, b * 32 % HD + lane);
#else
  const float v = X[(size_t)t * K_DIM + b * 32 + lane];
#endif
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


// quant formats: weights per block, bytes per block, and for the GEMV blocks per warp step. Every format is read 16 bytes per
// lane (32 weights): a warp step covers 1024 weights (Q4_K, Q6_K: 4 superblocks of 8 lanes) or 512 (Q8_0: 16 blocks of 2 lanes).
template <int QT> struct QF;
// Q4_K: 256 weights in 144 bytes: d, dmin (fp16), 12 bytes of 6-bit scales/mins, 128 bytes of nibbles
template <> struct QF<12> { static constexpr int BLOCK = 256, BBYTES = 144, BPS = 4; };
// Q6_K: 256 weights in 210 bytes: 128 bytes low nibbles, 64 bytes high 2 bits, 16 int8 scales, d (fp16)
template <> struct QF<14> { static constexpr int BLOCK = 256, BBYTES = 210, BPS = 4; };
// Q5_K: Q4_K plus a fifth bit per weight: d, dmin, 12 bytes of scales/mins, 32 bytes qh (bit j of qh[l] is weight l of sub-block j),
// 128 bytes of nibbles
template <> struct QF<13> { static constexpr int BLOCK = 256, BBYTES = 176, BPS = 4; };
// Q8_0: 32 weights in 34 bytes: d (fp16), 32 int8
template <> struct QF<8> { static constexpr int BLOCK = 32, BBYTES = 34, BPS = 16; };
template <int QT> struct QRow {
  static_assert(K_DIM % QF<QT>::BLOCK == 0, "K must be a multiple of the quant block");
  static constexpr int NBLOCKS = K_DIM / QF<QT>::BLOCK, STEPS = (NBLOCKS + QF<QT>::BPS - 1) / QF<QT>::BPS;
  static constexpr size_t BYTES = (size_t)NBLOCKS * QF<QT>::BBYTES;
  // Q4_K blocks are 16-byte aligned (nv.py checks the buffer offset); Q6_K/Q8_0 blocks only 2-byte, read with ld16 below, which
  // needs 8-byte aligned rows
  static_assert(QT == 12 || BYTES % 8 == 0, "rows must be 8-byte aligned");
};

#ifdef GEMV
#define NWARPS 4
#ifndef GEMV_R
#define GEMV_R 1  // rows per warp (more than one with several tokens: the activations are loaded once for all of them)
#endif
#define ROWS (NWARPS / KSPLIT * GEMV_R)  // output rows per block; KSPLIT warps split each row's K

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

// one warp step st: this lane's 32 weights of a row times every token's activations. The activations a lane needs depend only on
// (st, lane), so they're loaded once (XA, gemv_x) and used for the GEMV_R rows a warp handles (gemv_w): with several tokens the
// per-token activation loads from L1 otherwise outweigh the weights' DRAM traffic.
template <int QT> struct XA;
template <int QT> __device__ __forceinline__ void gemv_x(const int *XQ, int st, int lane, XA<QT> (&x)[NTOK]);
template <int QT> __device__ __forceinline__ void gemv_w(const unsigned char *wr, int st, int lane, const XA<QT> (&x)[NTOK], float (&acc)[NTOK]);

// Q4_K. lane: 64-weight chunk j of its superblock (32 bytes: low nibbles = sub-block 2j, high = 2j+1), half p of it
template <> struct XA<12> { int4 lo, hi; float4 ds; };  // the chunk's activations, (d, d*sum) of sub-blocks 2j and 2j+1
template <> __device__ __forceinline__ void gemv_x<12>(const int *XQ, int st, int lane, XA<12> (&x)[NTOK]) {
  const int ub = st * QF<12>::BPS + (lane >> 3), j = (lane >> 1) & 3, p = lane & 1;
  if (ub >= QRow<12>::NBLOCKS) return;
  const int e = ub * 256 + 64 * j + 16 * p;  // first element of the low nibbles; the high ones are 32 later
#pragma unroll
  for (int t = 0; t < NTOK; t++) {
    const int *xt = XQ + (size_t)t * XWORDS;
    x[t].lo = *(const int4 *)(xt + e / 4), x[t].hi = *(const int4 *)(xt + e / 4 + 8);
    x[t].ds = *(const float4 *)(xt + K_DIM / 4 + 2 * (e / 32));
  }
}
template <> __device__ __forceinline__ void gemv_w<12>(const unsigned char *wr, int st, int lane, const XA<12> (&x)[NTOK], float (&acc)[NTOK]) {
  const int ub = st * QF<12>::BPS + (lane >> 3), j = (lane >> 1) & 3, p = lane & 1;
  if (ub >= QRow<12>::NBLOCKS) return;
  const unsigned char *sb = wr + (size_t)ub * QF<12>::BBYTES;
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
#pragma unroll
  for (int t = 0; t < NTOK; t++) {
    float r = scl[0] * x[t].ds.x * (float)dp16(lo, x[t].lo, 0) + scl[1] * x[t].ds.z * (float)dp16(hi, x[t].hi, 0);
    if (p == 0) r -= mn[0] * x[t].ds.y + mn[1] * x[t].ds.w;  // the mins: once per sub-block
    acc[t] += r;
  }
}

// Q5_K: Q4_K's lane layout (the activations are the same), the nibbles at 48, the fifth bits at 16
template <> struct XA<13> : XA<12> {};
template <> __device__ __forceinline__ void gemv_x<13>(const int *XQ, int st, int lane, XA<13> (&x)[NTOK]) {
  gemv_x<12>(XQ, st, lane, reinterpret_cast<XA<12> (&)[NTOK]>(x));
}
template <> __device__ __forceinline__ void gemv_w<13>(const unsigned char *wr, int st, int lane, const XA<13> (&x)[NTOK], float (&acc)[NTOK]) {
  const int ub = st * QF<13>::BPS + (lane >> 3), j = (lane >> 1) & 3, p = lane & 1;
  if (ub >= QRow<13>::NBLOCKS) return;
  const unsigned char *sb = wr + (size_t)ub * QF<13>::BBYTES;
  const uint4 hd = *(const uint4 *)sb, qh = *(const uint4 *)(sb + 16 + 16 * p), qs = *(const uint4 *)(sb + 48 + 32 * j + 16 * p);
  const float d = __half2float(__ushort_as_half((unsigned short)(hd.x & 0xffff))),
              dmin = __half2float(__ushort_as_half((unsigned short)(hd.x >> 16)));
  float scl[2], mn[2];
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int sbk = 2 * j + h, sh = 8 * (sbk & 3);
    const unsigned b0 = (hd.y >> sh) & 0xff, b1 = (hd.z >> sh) & 0xff, b2 = (hd.w >> sh) & 0xff;
    const unsigned s = sbk < 4 ? b0 & 63 : (b2 & 0xF) | ((b0 >> 6) << 4), m = sbk < 4 ? b1 & 63 : (b2 >> 4) | ((b1 >> 6) << 4);
    scl[h] = d * (float)s, mn[h] = dmin * (float)m;
  }
  // 5-bit values: nibble | (fifth bit << 4), the bit of sub-block 2j for the low nibbles, 2j+1 for the high ones
  const uint4 lo = make_uint4((qs.x & 0x0F0F0F0F) | (((qh.x >> (2 * j)) & 0x01010101) << 4), (qs.y & 0x0F0F0F0F) | (((qh.y >> (2 * j)) & 0x01010101) << 4),
                              (qs.z & 0x0F0F0F0F) | (((qh.z >> (2 * j)) & 0x01010101) << 4), (qs.w & 0x0F0F0F0F) | (((qh.w >> (2 * j)) & 0x01010101) << 4));
  const uint4 hi = make_uint4(((qs.x >> 4) & 0x0F0F0F0F) | (((qh.x >> (2 * j + 1)) & 0x01010101) << 4),
                              ((qs.y >> 4) & 0x0F0F0F0F) | (((qh.y >> (2 * j + 1)) & 0x01010101) << 4),
                              ((qs.z >> 4) & 0x0F0F0F0F) | (((qh.z >> (2 * j + 1)) & 0x01010101) << 4),
                              ((qs.w >> 4) & 0x0F0F0F0F) | (((qh.w >> (2 * j + 1)) & 0x01010101) << 4));
#pragma unroll
  for (int t = 0; t < NTOK; t++) {
    float r = scl[0] * x[t].ds.x * (float)dp16(lo, x[t].lo, 0) + scl[1] * x[t].ds.z * (float)dp16(hi, x[t].hi, 0);
    if (p == 0) r -= mn[0] * x[t].ds.y + mn[1] * x[t].ds.w;
    acc[t] += r;
  }
}

// Q6_K. lane: half h (128 weights) of its superblock, 16 of that half's 64 low-nibble bytes at b: low nibbles are weights b..,
// high ones 64+b..; their 2 high bits are in qh bytes b % 32.. at shift 2 * (b / 32), and 4 more for the high nibbles
template <> struct XA<14> { int4 lo, hi; float dl, dh; };
template <> __device__ __forceinline__ void gemv_x<14>(const int *XQ, int st, int lane, XA<14> (&x)[NTOK]) {
  const int ub = st * QF<14>::BPS + (lane >> 3), h = (lane >> 2) & 1, b = (lane & 3) * 16;
  if (ub >= QRow<14>::NBLOCKS) return;
  const int e = ub * 256 + 128 * h + b;
#pragma unroll
  for (int t = 0; t < NTOK; t++) {
    const int *xt = XQ + (size_t)t * XWORDS;
    const float2 *ds = (const float2 *)(xt + K_DIM / 4);
    x[t].lo = *(const int4 *)(xt + e / 4), x[t].hi = *(const int4 *)(xt + e / 4 + 16);
    x[t].dl = ds[e / 32].x, x[t].dh = ds[e / 32 + 2].x;
  }
}
template <> __device__ __forceinline__ void gemv_w<14>(const unsigned char *wr, int st, int lane, const XA<14> (&x)[NTOK], float (&acc)[NTOK]) {
  const int ub = st * QF<14>::BPS + (lane >> 3), h = (lane >> 2) & 1, b = (lane & 3) * 16;
  if (ub >= QRow<14>::NBLOCKS) return;
  const unsigned char *sb = wr + (size_t)ub * QF<14>::BBYTES;
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
#pragma unroll
  for (int t = 0; t < NTOK; t++)
    acc[t] += sc_lo * x[t].dl * (float)dp16(q_lo, x[t].lo, 0) + sc_hi * x[t].dh * (float)dp16(q_hi, x[t].hi, 0);
}

// Q8_0. lane: half p (16 int8) of block ub
template <> struct XA<8> { int4 q; float d; };
template <> __device__ __forceinline__ void gemv_x<8>(const int *XQ, int st, int lane, XA<8> (&x)[NTOK]) {
  const int ub = st * QF<8>::BPS + (lane >> 1), p = lane & 1;
  if (ub >= QRow<8>::NBLOCKS) return;
  const int e = ub * 32 + 16 * p;
#pragma unroll
  for (int t = 0; t < NTOK; t++) {
    const int *xt = XQ + (size_t)t * XWORDS;
    x[t].q = *(const int4 *)(xt + e / 4), x[t].d = ((const float2 *)(xt + K_DIM / 4))[ub].x;
  }
}
template <> __device__ __forceinline__ void gemv_w<8>(const unsigned char *wr, int st, int lane, const XA<8> (&x)[NTOK], float (&acc)[NTOK]) {
  const int ub = st * QF<8>::BPS + (lane >> 1), p = lane & 1;
  if (ub >= QRow<8>::NBLOCKS) return;
  const unsigned char *qb = wr + (size_t)ub * QF<8>::BBYTES;
  const uint4 q = ld16(qb + 2 + 16 * p);
  const float d = ld_half(qb);
#pragma unroll
  for (int t = 0; t < NTOK; t++) acc[t] += d * x[t].d * (float)dp16(q, x[t].q, 0);
}

// row block blk of weight W ([N][QRow<QT>::BYTES]): output columns blk * ROWS.. of OUT ([NTOK][N_OUT], already offset to W's
// first column). Warp w handles rows (w / KSPLIT) * GEMV_R .. + GEMV_R, its K share ks.
template <int QT, int N>
__device__ __forceinline__ void gemv_rows(float *OUT, const unsigned char *W, const int *XQ, int blk, float (*part)[GEMV_R][NTOK],
                                          const float *RES = nullptr) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = blk * ROWS + (warp / KSPLIT) * GEMV_R, ks = warp % KSPLIT;
  float acc[GEMV_R][NTOK];
#pragma unroll
  for (int r = 0; r < GEMV_R; r++)
#pragma unroll
    for (int t = 0; t < NTOK; t++) acc[r][t] = 0.0f;
  if (row0 < N) {
#pragma unroll (GEMV_R > 1 ? 2 : 4)
    for (int st = ks; st < QRow<QT>::STEPS; st += KSPLIT) {
      XA<QT> x[NTOK];
      gemv_x<QT>(XQ, st, lane, x);
#pragma unroll
      for (int r = 0; r < GEMV_R; r++)
        if (GEMV_R == 1 || row0 + r < N) gemv_w<QT>(W + (size_t)(row0 + r) * QRow<QT>::BYTES, st, lane, x, acc[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < GEMV_R; r++)
#pragma unroll
    for (int t = 0; t < NTOK; t++)
#pragma unroll
      for (int off = 16; off; off >>= 1) acc[r][t] += __shfl_xor_sync(0xffffffff, acc[r][t], off);
#if KSPLIT > 1
  if (lane == 0)
#pragma unroll
    for (int r = 0; r < GEMV_R; r++)
#pragma unroll
      for (int t = 0; t < NTOK; t++) part[warp][r][t] = acc[r][t];
  __syncthreads();
  if (ks == 0 && lane == 0)
#pragma unroll
    for (int r = 0; r < GEMV_R; r++)
#pragma unroll
      for (int t = 0; t < NTOK; t++) {
#pragma unroll
        for (int s = 1; s < KSPLIT; s++) acc[r][t] += part[warp + s][r][t];
      }
#endif
  if (ks == 0 && lane == 0)
#pragma unroll
    for (int r = 0; r < GEMV_R; r++)
      if (row0 + r < N)
#pragma unroll
        for (int t = 0; t < NTOK; t++) {
#ifdef RESID  // the residual stream added as it's stored (one weight: OUT and RES are [NTOK][N])
          OUT[(size_t)t * N_OUT + row0 + r] = acc[r][t] + RES[(size_t)t * N_OUT + row0 + r];
#else
          OUT[(size_t)t * N_OUT + row0 + r] = acc[r][t];
#endif
        }
}

// grid (total row blocks over the segments), NWARPS * 32 threads. OUT: [NTOK][N_OUT] fp32. XQ: [NTOK][XWORDS] from QUANTIZE.
// SEG_DISPATCH calls gemv_rows for the segment that owns blockIdx.x (uniform per block, so the __syncthreads in it is safe).
#ifdef RESID
#define RES_PARAM , const float *__restrict__ RES
#else
#define RES_PARAM
#endif
extern "C" __global__ void __launch_bounds__(NWARPS * 32) KNAME(float *__restrict__ OUT, SEG_PARAMS, const int *__restrict__ XQ RES_PARAM) {
  __shared__ float part[NWARPS][GEMV_R][NTOK];
  const int blk = blockIdx.x;
  SEG_DISPATCH
}
#endif

#ifdef DEQUANT
// Prefill on quant weights: dequantize to fp16 for the tensor core GEMM. DEP is the layer's input, unused: it makes this kernel
// wait for the layer, so each layer's fp16 copy lives only until its GEMM (else the scheduler can run every layer's dequantize
// first, and they'd all be in memory at once). grid (N_OUT * K / 8 / 128), 128 threads: 8 weights (one 16-byte store) per thread.
// OUT: [N_OUT][K] fp16, the segments' rows one after another (or two segments' rows alternating, for the GEMM's SwiGLU epilogue);
// SEG_DISPATCH calls dequant8 on the segment that owns the row.

// weights col..col+7 of row wr
template <int QT> __device__ __forceinline__ void dequant8(const unsigned char *wr, int col, float (&v)[8]);
template <> __device__ __forceinline__ void dequant8<12>(const unsigned char *wr, int col, float (&v)[8]) {
  const unsigned char *b = wr + (size_t)(col / 256) * 144;
  const int e0 = col % 256;
  // weights e0..e0+7: chunk j = e0 / 64 of 32 qs bytes, low nibbles (sub-block 2j) for the first 32, high (2j+1) for the rest
  const int j = e0 / 64, hi = (e0 % 64) >= 32, l0 = e0 % 32, sbk = 2 * j + hi;
  const float d = __half2float(*(const half *)b), dmin = __half2float(*(const half *)(b + 2));
  const unsigned char *sc = b + 4;
  const int s = sbk < 4 ? sc[sbk] & 63 : (sc[sbk + 4] & 0xF) | ((sc[sbk - 4] >> 6) << 4);
  const int m = sbk < 4 ? sc[sbk + 4] & 63 : (sc[sbk + 4] >> 4) | ((sc[sbk] >> 6) << 4);
#pragma unroll
  for (int i = 0; i < 8; i++) v[i] = d * s * (float)((b[16 + 32 * j + l0 + i] >> (4 * hi)) & 0xF) - dmin * m;
}
template <> __device__ __forceinline__ void dequant8<13>(const unsigned char *wr, int col, float (&v)[8]) {
  const unsigned char *b = wr + (size_t)(col / 256) * 176;
  const int e0 = col % 256, j = e0 / 64, hi = (e0 % 64) >= 32, l0 = e0 % 32, sbk = 2 * j + hi;
  const float d = __half2float(*(const half *)b), dmin = __half2float(*(const half *)(b + 2));
  const unsigned char *sc = b + 4;
  const int s = sbk < 4 ? sc[sbk] & 63 : (sc[sbk + 4] & 0xF) | ((sc[sbk - 4] >> 6) << 4);
  const int m = sbk < 4 ? sc[sbk + 4] & 63 : (sc[sbk + 4] >> 4) | ((sc[sbk] >> 6) << 4);
#pragma unroll
  for (int i = 0; i < 8; i++)
    v[i] = d * s * (float)(((b[48 + 32 * j + l0 + i] >> (4 * hi)) & 0xF) | (((b[16 + l0 + i] >> sbk) & 1) << 4)) - dmin * m;
}
template <> __device__ __forceinline__ void dequant8<14>(const unsigned char *wr, int col, float (&v)[8]) {
  const unsigned char *b = wr + (size_t)(col / 256) * 210;
  const int e0 = col % 256;
  // weights e0..: half h, quadrant qd of 32: low/high nibble of ql[64h + 32 (qd & 1) + l], 2 bits of qh[32h + l] at shift 2 qd
  const int h = e0 / 128, t = e0 % 128, qd = t / 32, l0 = t % 32;
  const float d = __half2float(*(const half *)(b + 208)) * (float)((const signed char *)b)[192 + 8 * h + t / 16];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int lo = (b[64 * h + 32 * (qd & 1) + l0 + i] >> (4 * (qd >> 1))) & 0xF, hb = (b[128 + 32 * h + l0 + i] >> (2 * qd)) & 3;
    v[i] = d * (float)((lo | (hb << 4)) - 32);
  }
}
template <> __device__ __forceinline__ void dequant8<8>(const unsigned char *wr, int col, float (&v)[8]) {
  const unsigned char *b = wr + (size_t)(col / 32) * 34;
  const int e0 = col % 32;
  const float d = __half2float(*(const half *)b);
#pragma unroll
  for (int i = 0; i < 8; i++) v[i] = d * (float)((const signed char *)b)[2 + e0 + i];
}

extern "C" __global__ void __launch_bounds__(128) KNAME(half *__restrict__ OUT, SEG_PARAMS, const void *DEP) {
  const size_t i8 = (size_t)blockIdx.x * 128 + threadIdx.x;  // group of 8 weights
  const int row = (int)(i8 * 8 / K_DIM), col = (int)(i8 * 8 % K_DIM);
  float v[8];
  SEG_DISPATCH
  half2 o[4];
#pragma unroll
  for (int i = 0; i < 4; i++) o[i] = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
  *(uint4 *)(OUT + i8 * 8) = *(uint4 *)o;
}
#endif

#ifdef REQUANT
// Prefill on quant weights with the int8 GEMM (nv_mmq.cu): each 32-weight block as int8 values and a float2, w = s*q - m for Q8_0 /
// Q4_K ((s, m), m = 0 for Q8_0) or one scale per 16 for Q6_K ((s_lo, s_hi), q = 6-bit value - 32). Exact: no rounding.
// grid (N_OUT * K / 32 / 128), 128 threads: one block per thread. W8: [N_OUT][K] int8, WS: [N_OUT][K/32] float2, the segments'
// rows one after another or alternating (SEG_DISPATCH calls requant32 on the segment that owns the row). DEP as DEQUANT's.
template <int QT> __device__ __forceinline__ void requant32(const unsigned char *wr, int blk, unsigned (&q)[8], float2 &sc);
template <> __device__ __forceinline__ void requant32<12>(const unsigned char *wr, int blk, unsigned (&q)[8], float2 &sc) {
  const unsigned char *b = wr + (size_t)(blk / 8) * 144;
  const int j = blk % 8;  // sub-block: low nibbles of qs[32 (j/2) ..] for even j, high for odd
  const float d = __half2float(*(const half *)b), dmin = __half2float(*(const half *)(b + 2));
  const unsigned char *scs = b + 4;
  const int s = j < 4 ? scs[j] & 63 : (scs[j + 4] & 0xF) | ((scs[j - 4] >> 6) << 4);
  const int m = j < 4 ? scs[j + 4] & 63 : (scs[j + 4] >> 4) | ((scs[j] >> 6) << 4);
  const uint4 *qs = (const uint4 *)(b + 16 + 32 * (j / 2));  // 16-byte aligned: superblocks are 144 bytes from an aligned row
#pragma unroll
  for (int i = 0; i < 2; i++) {
    const uint4 u = qs[i];
    const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int k = 0; k < 4; k++) q[4 * i + k] = (w[k] >> (4 * (j & 1))) & 0x0F0F0F0F;
  }
  sc = make_float2(d * s, dmin * m);
}
template <> __device__ __forceinline__ void requant32<13>(const unsigned char *wr, int blk, unsigned (&q)[8], float2 &sc) {
  const unsigned char *b = wr + (size_t)(blk / 8) * 176;
  const int j = blk % 8;  // sub-block: nibbles (j & 1) of qs[32 (j/2) ..], fifth bits = bit j of qh[0..31]
  const float d = __half2float(*(const half *)b), dmin = __half2float(*(const half *)(b + 2));
  const unsigned char *scs = b + 4;
  const int s = j < 4 ? scs[j] & 63 : (scs[j + 4] & 0xF) | ((scs[j - 4] >> 6) << 4);
  const int m = j < 4 ? scs[j + 4] & 63 : (scs[j + 4] >> 4) | ((scs[j] >> 6) << 4);
  const uint4 *qs = (const uint4 *)(b + 48 + 32 * (j / 2)), *qh = (const uint4 *)(b + 16);
#pragma unroll
  for (int i = 0; i < 2; i++) {
    const uint4 u = qs[i], h = qh[i];
    const unsigned w[4] = {u.x, u.y, u.z, u.w}, hb[4] = {h.x, h.y, h.z, h.w};
#pragma unroll
    for (int k = 0; k < 4; k++) q[4 * i + k] = ((w[k] >> (4 * (j & 1))) & 0x0F0F0F0F) | (((hb[k] >> j) & 0x01010101) << 4);
  }
  sc = make_float2(d * s, dmin * m);
}
template <> __device__ __forceinline__ void requant32<14>(const unsigned char *wr, int blk, unsigned (&q)[8], float2 &sc) {
  const unsigned char *b = wr + (size_t)(blk / 8) * 210;
  const int t = (blk % 8) * 32, h = t / 128, qd = (t % 128) / 32;  // as dequant8<14>, for l = 0..31
  const float d = __half2float(*(const half *)(b + 208));
  const signed char *scs = (const signed char *)b + 192 + 8 * h + 2 * qd;
  const unsigned char *ql = b + 64 * h + 32 * (qd & 1), *qh = b + 128 + 32 * h;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    unsigned v = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const int l = 4 * i + k, lo = (ql[l] >> (4 * (qd >> 1))) & 0xF, hb = (qh[l] >> (2 * qd)) & 3;
      v |= (unsigned)(((lo | (hb << 4)) - 32) & 0xFF) << (8 * k);
    }
    q[i] = v;
  }
  sc = make_float2(d * scs[0], d * scs[1]);
}
template <> __device__ __forceinline__ void requant32<8>(const unsigned char *wr, int blk, unsigned (&q)[8], float2 &sc) {
  const unsigned char *b = wr + (size_t)blk * 34;
  const unsigned short *h = (const unsigned short *)(b + 2);  // 2-byte aligned
#pragma unroll
  for (int i = 0; i < 8; i++) q[i] = (unsigned)h[2 * i] | ((unsigned)h[2 * i + 1] << 16);
  sc = make_float2(__half2float(*(const half *)b), 0.0f);
}

extern "C" __global__ void __launch_bounds__(128) KNAME(unsigned char *__restrict__ W8, float2 *__restrict__ WS, SEG_PARAMS, const void *DEP) {
  const size_t i = (size_t)blockIdx.x * 128 + threadIdx.x;  // block of 32 weights
  const int row = (int)(i / (K_DIM / 32)), blk = (int)(i % (K_DIM / 32));
  unsigned q[8];
  float2 sc;
  SEG_DISPATCH
  uint4 *o = (uint4 *)(W8 + i * 32);
  o[0] = make_uint4(q[0], q[1], q[2], q[3]);
  o[1] = make_uint4(q[4], q[5], q[6], q[7]);
  WS[i] = sc;
}
#endif

