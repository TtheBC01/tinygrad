// Causal flash attention (FlashAttention-2 style) for Ampere (sm_80+) on the LLM's fp16 KV cache, for prefill.
// O[b, t, h, :] = softmax(Q[b, h, t, :] . K[b, kvh, :valid, :]^T / sqrt(HD), causal) @ V[b, kvh, :valid, :],  kvh = h / (NH / NKV)
// Q: [NB][NH][TQ][HD] fp16 (TQ = padded query rows), KV: [2][NB][NKV][NCTX][HD] fp16 (K then V, the whole cache),
// params: {valid_end, q_start} int32 at runtime (keys [0, valid_end) are valid, query row t sits at position q_start + t).
// Query row t attends to keys [0, min(q_start + t + 1, valid_end)). O: [NB][TQ][NH][HD] fp32, so the model's transpose is free.
// One block = BM = 16*WARPS query rows of one head; each warp owns 16 rows. Keys stream through shared memory BN at a time
// (STAGES-deep cp.async), S = Q K^T and P V run on mma.sync m16n8k16 with fp32 accumulators, softmax is online in fp32.
// nv.py prepends nv_common.cuh and #defines KNAME NB NH NKV TQ NCTX HD WARPS BN STAGES.

#define NEG_INF __int_as_float(0xff800000)  // -inf (nvrtc --minimal has no math.h INFINITY)
#define BM (16 * WARPS)
#define NTHREADS (32 * WARPS)
#define DK (HD / 16)  // k16 steps over the head dim (S = Q K^T)
#define DN (HD / 8)   // n8 tiles over the head dim (O)
#define NN (BN / 8)   // n8 tiles over the keys (S)
#define KK (BN / 16)  // k16 steps over the keys (P V)
static_assert(HD % 64 == 0, "the swizzle needs >= 8 x 16B chunks per row");
static_assert(BN % 16 == 0 && TQ % BM == 0 && NCTX % BN == 0 && NH % NKV == 0, "shapes must tile");

// [rows][HD] halves, rows are >= 128B: XOR the 16B chunk with row&7, so ldmatrix (8 rows at one chunk) is conflict-free
__device__ __forceinline__ int sw(int row, int chunk) { return row * HD + ((chunk ^ (row & 7)) << 3); }
__device__ __forceinline__ unsigned pack_half2(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *(unsigned *)&h;
}

extern "C" __global__ void __launch_bounds__(NTHREADS)
KNAME(float *__restrict__ O, const half *__restrict__ Q, const half *__restrict__ KV, const int *__restrict__ params) {
  extern __shared__ __align__(128) unsigned char smem[];
  half *sK = (half *)smem, *sV = sK + STAGES * BN * HD;  // [STAGES][BN][HD] each; the Q tile borrows the front first

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int valid_end = params[0], q_start = params[1];
  // the last query blocks see the most keys: launch them first so they don't form the tail
  const int mb = (TQ / BM - 1) - blockIdx.x, bh = blockIdx.y, b = bh / NH, h = bh % NH, kvh = h / (NH / NKV);
  const half *qg = Q + ((size_t)bh * TQ + mb * BM) * HD;
  const half *kg = KV + ((size_t)(0 * NB + b) * NKV + kvh) * NCTX * HD;
  const half *vg = KV + ((size_t)(1 * NB + b) * NKV + kvh) * NCTX * HD;

  // Q tile -> shared -> A fragments in registers, kept for the whole loop
  for (int c = tid; c < BM * HD / 8; c += NTHREADS) cp_async16(sK + sw(c / (HD / 8), c % (HD / 8)), qg + c * 8);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();
  unsigned qf[DK][4];
#pragma unroll
  for (int kk = 0; kk < DK; kk++) ldmatrix_x4(qf[kk], sK + sw(warp * 16 + (lane & 15), kk * 2 + (lane >> 4)));
  __syncthreads();  // the K/V stages overwrite the Q tile

  // this thread's two rows (fragment rows g and g+8 of the warp's 16) and how many keys each may see
  const int q_lo = q_start + mb * BM, r0 = warp * 16 + (lane >> 2), r1 = r0 + 8;
  const int lim0 = min(q_lo + r0 + 1, valid_end), lim1 = min(q_lo + r1 + 1, valid_end);
  const int kend = min(q_lo + BM, valid_end), ntiles = (kend + BN - 1) / BN;
  const int nomask_end = min(q_lo + 1, valid_end);  // tiles ending at or before this need no mask in any row of the block

  auto load_kv = [&](int stage, int j) {
    half *ks = sK + stage * BN * HD, *vs = sV + stage * BN * HD;
    for (int c = tid; c < BN * HD / 8; c += NTHREADS) {
      int row = c / (HD / 8), ch = c % (HD / 8);
      size_t g = (size_t)(j * BN + row) * HD + ch * 8;
      cp_async16(ks + sw(row, ch), kg + g);
      cp_async16(vs + sw(row, ch), vg + g);
    }
  };

  float o[DN][4];
#pragma unroll
  for (int i = 0; i < DN; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
  float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.0f, l1 = 0.0f;  // running max (log2 units) and this thread's share of the row sum
  const float scale = 1.4426950408889634f * rsqrtf((float)HD);  // log2(e)/sqrt(HD): softmax via exp2

#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < ntiles) load_kv(s, s);
    cp_async_commit();
  }
  for (int j = 0; j < ntiles; j++) {
    cp_async_wait<STAGES - 2>();  // tile j has landed
    __syncthreads();              // and every warp is done with tile j-1, whose slot the prefetch below refills
    if (j + STAGES - 1 < ntiles) load_kv((j + STAGES - 1) % STAGES, j + STAGES - 1);
    cp_async_commit();
    const half *ks = sK + (j % STAGES) * BN * HD, *vs = sV + (j % STAGES) * BN * HD;

    // S = Q K^T for this warp's 16 rows x BN keys. K is [key][d], i.e. the mma's col-major B: plain ldmatrix
    float s[NN][4];
#pragma unroll
    for (int i = 0; i < NN; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f;
#pragma unroll
    for (int kk = 0; kk < DK; kk++)
#pragma unroll
      for (int nt = 0; nt < NN; nt += 2) {
        unsigned r[4];
        ldmatrix_x4(r, ks + sw(nt * 8 + ((lane >> 4) << 3) + (lane & 7), kk * 2 + ((lane >> 3) & 1)));
        mma16816(s[nt], qf[kk], r);
        mma16816(s[nt + 1], qf[kk], r + 2);
      }

    // scale to log2 units, mask, and the online softmax update
    const bool masked = (j + 1) * BN > nomask_end;
    float mx0 = m0, mx1 = m1;
#pragma unroll
    for (int nt = 0; nt < NN; nt++)
#pragma unroll
      for (int c = 0; c < 4; c++) {
        float v = s[nt][c] * scale;
        if (masked && j * BN + nt * 8 + (lane & 3) * 2 + (c & 1) >= (c < 2 ? lim0 : lim1)) v = NEG_INF;
        s[nt][c] = v;
        if (c < 2) mx0 = fmaxf(mx0, v); else mx1 = fmaxf(mx1, v);
      }
    // a row's 16 fragment values live in the 4 lanes of a quad
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    // every row has key 0 in tile 0, so mx is finite from then on and the rescale below never sees -inf - -inf
    const float a0 = exp2f(m0 - mx0), a1 = exp2f(m1 - mx1);
    m0 = mx0, m1 = mx1;
    float rs0 = 0.0f, rs1 = 0.0f;
#pragma unroll
    for (int nt = 0; nt < NN; nt++) {
      s[nt][0] = exp2f(s[nt][0] - m0); s[nt][1] = exp2f(s[nt][1] - m0);
      s[nt][2] = exp2f(s[nt][2] - m1); s[nt][3] = exp2f(s[nt][3] - m1);
      rs0 += s[nt][0] + s[nt][1];
      rs1 += s[nt][2] + s[nt][3];
    }
    l0 = l0 * a0 + rs0, l1 = l1 * a1 + rs1;
#pragma unroll
    for (int dn = 0; dn < DN; dn++) o[dn][0] *= a0, o[dn][1] *= a0, o[dn][2] *= a1, o[dn][3] *= a1;

    // O += P V. The S accumulators of two adjacent n8 tiles are exactly the A fragment of one k16 step. V is [key][d]: ldmatrix.trans
#pragma unroll
    for (int kk = 0; kk < KK; kk++) {
      unsigned pa[4] = {pack_half2(s[2 * kk][0], s[2 * kk][1]), pack_half2(s[2 * kk][2], s[2 * kk][3]),
                        pack_half2(s[2 * kk + 1][0], s[2 * kk + 1][1]), pack_half2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
      for (int dn = 0; dn < DN; dn += 2) {
        unsigned r[4];
        ldmatrix_x4_trans(r, vs + sw(kk * 16 + (lane & 15), dn + (lane >> 4)));
        mma16816(o[dn], pa, r);
        mma16816(o[dn + 1], pa, r + 2);
      }
    }
  }

  // finish the row sums across the quad, normalize, store O[b][t][h][:]
  l0 += __shfl_xor_sync(0xffffffff, l0, 1); l0 += __shfl_xor_sync(0xffffffff, l0, 2);
  l1 += __shfl_xor_sync(0xffffffff, l1, 1); l1 += __shfl_xor_sync(0xffffffff, l1, 2);
  const float inv0 = 1.0f / l0, inv1 = 1.0f / l1;
  float *o0 = O + (((size_t)b * TQ + mb * BM + r0) * NH + h) * HD + (lane & 3) * 2;
  float *o1 = O + (((size_t)b * TQ + mb * BM + r1) * NH + h) * HD + (lane & 3) * 2;
#pragma unroll
  for (int dn = 0; dn < DN; dn++) {
    *(float2 *)&o0[dn * 8] = make_float2(o[dn][0] * inv0, o[dn][1] * inv0);
    *(float2 *)&o1[dn * 8] = make_float2(o[dn][2] * inv1, o[dn][3] * inv1);
  }
}
