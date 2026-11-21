// int8 tensor core GEMM for the LLM's prefill on GGUF quant weights (llama.cpp's "MMQ" idea) for Ampere (sm_80/86):
// C[M,N] = X @ W^T with X quantized to int8 per 32 values along K (nv_quant.cu's QUANTIZE: per row K int8, then K/32 (d, d*sum))
// and W as int8 [N][K] plus a float2 per 32-block (nv_quant.cu's REQUANT from Q4_K / Q6_K / Q8_0):
//   Q8_0, Q4_K: (s, m), w = s*q - m (m = 0 for Q8_0)  ->  sum_k x w = dx*s*dot(qx, qw) - (dx*sum qx)*m
//   Q6_K:       (s_lo, s_hi), one scale per 16 weights  ->  dx*(s_lo*dot_lo + s_hi*dot_hi)   (for tiles whose column bn is in
//               a Q6_K segment: IS_Q6(bn), a condition nv.py defines)
// On GA102 int8 mma runs at 4x the fp16-input/fp32-accumulate rate, and no fp16 copy of the weight is written.
// The int8 fragments of mma.m16n8k32 have the same register layout as fp16 m16n8k16 ones over twice the bytes, so the tiles,
// swizzle and ldmatrix loads are nv_gemm.cu's K-contiguous ones on bytes viewed as halves (BKH = 32 halves = 64 int8 = two quant
// blocks per stage). After each block's mma the int32 dot is scaled into the fp32 accumulators.
// Epilogue: fp32 C[M][N], or with GLU (columns interleaved (gate, up) pairs) half C[M][N/2] of their SwiGLU, as nv_gemm.cu.
// nv.py prepends nv_common.cuh and #defines KNAME M_DIM N_DIM K_DIM BM BN WM WN STAGES GROUP_M SK_P IS_Q6(bn) HAS_MIN [GLU...].
#ifndef GROUP_M
#define GROUP_M 8
#endif
#ifndef SK_P
#define SK_P 0
#endif
#ifndef HAS_MIN
#define HAS_MIN 1
#endif
#ifndef IS_Q6
#define IS_Q6(bn) false
#endif
#ifndef NBLK
#define NBLK 2                 // quant blocks (32 int8 of K) per pipeline stage: 2 or 4
#endif
#define BKH (16 * NBLK)        // k tile in halves
#define KB (32 * NBLK)         // k tile in int8
#define NWARPS ((BM / WM) * (BN / WN))
#define NTHREADS (NWARPS * 32)
#define MT (WM / 16)
#define NT (WN / 8)
#define XROW (K_DIM + K_DIM / 4)  // bytes per activation row: K int8, then K/32 float2 (d, d*sum)
static_assert(NBLK == 2 || NBLK == 4, "2 or 4 quant blocks per stage");
static_assert(NT % 2 == 0 && M_DIM % BM == 0 && N_DIM % BN == 0 && K_DIM % KB == 0, "shape must be a multiple of the tile");
static_assert((BM * BKH / 8) % NTHREADS == 0 && (BN * BKH / 8) % NTHREADS == 0 && BM <= NTHREADS * 2 && BN <= NTHREADS * 2, "tile loads");

#ifdef GLU
#define C_COLS (N_DIM / 2)
typedef half out_t;
__device__ __forceinline__ float glu_act(float g, float u) {
#ifdef GLU_LIMIT
  g = fminf(g, GLU_LIMIT), u = fminf(fmaxf(u, -GLU_LIMIT), GLU_LIMIT);
#endif
  return g / (1.0f + expf(-GLU_ALPHA * g)) * (u + GLU_UP_BIAS);
}
#else
#define C_COLS N_DIM
typedef float out_t;
#endif
__device__ __forceinline__ void store2(float *p, float x, float y) { *(float2 *)p = make_float2(x, y); }
__device__ __forceinline__ void store2(half *p, float x, float y) { *(half2 *)p = __floats2half2_rn(x, y); }

__device__ __forceinline__ void mma_s8_k32(int (&c)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
      : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(0));
}
__device__ __forceinline__ void mma_s8_k16(int (&c)[4], unsigned a0, unsigned a1, unsigned b) {
  asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
      : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3]) : "r"(a0), "r"(a1), "r"(b), "r"(0));
}
// exact int32 -> fp32 for |i| < 2^22 (a block's dot is at most 32*127*127): two ALU ops instead of a quarter-rate I2F
__device__ __forceinline__ float i2f(int i) { return __int_as_float(i + 0x4B400000) - 12582912.0f; }

// K-contiguous stage rows of BKH halves, 16-byte chunks XOR-swizzled so ldmatrix's 8 rows at one chunk hit distinct banks (in halves)
#if NBLK == 2
__device__ __forceinline__ int kc_off(int row, int chunk) { return row * BKH + ((chunk ^ ((row >> 1) & 3)) << 3); }
#else
__device__ __forceinline__ int kc_off(int row, int chunk) { return row * BKH + ((chunk ^ (row & 7)) << 3); }
#endif
__device__ __forceinline__ int ld_acquire(const int *p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release(int *p, int v) { asm volatile("st.release.gpu.global.b32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory"); }

constexpr int NBX = N_DIM / BN, NBY = M_DIM / BM, TILES = NBX * NBY, KT = K_DIM / KB;
constexpr int SK_P1 = SK_P > 0 ? SK_P : 1;
constexpr int SK_TILES = (SK_P == 0 || TILES % SK_P1 == 0) ? 0 : (TILES >= 2 * SK_P1 ? TILES % SK_P1 + SK_P1 : TILES);
constexpr long long SK_ITERS = (long long)SK_TILES * KT;
constexpr int SK_WORKERS = SK_ITERS < SK_P ? (int)SK_ITERS : SK_P;
constexpr int DP_TILES = TILES - SK_TILES, GEMM_GRID = SK_WORKERS + DP_TILES;

__device__ __forceinline__ void tile_coords(int t, int &bm, int &bn) {
#if GROUP_M > 0
  const int first_m = (t / (GROUP_M * NBX)) * GROUP_M, gsz = (NBY - first_m) < GROUP_M ? (NBY - first_m) : GROUP_M;
  bm = (first_m + (t % gsz)) * BM, bn = ((t % (GROUP_M * NBX)) / gsz) * BN;
#else
  bm = (t / NBX) * BM, bn = (t % NBX) * BN;
#endif
}

// X: [M][XROW] bytes. W: [N][K] int8. WS: [N][K/32] float2. smem per stage: A BM x 64 B, B BN x 64 B, then the stage's two blocks'
// scales: A BM x 16 B, B BN x 16 B
extern "C" __global__ void __launch_bounds__(NTHREADS)
KNAME(out_t *__restrict__ C, const unsigned char *__restrict__ X, const unsigned char *__restrict__ W, const float2 *__restrict__ WS,
      float *__restrict__ ws, int *__restrict__ flags) {
  extern __shared__ __align__(128) unsigned char smem[];
  half *As = (half *)smem, *Bs = As + STAGES * BM * BKH;
  float2 *Asc = (float2 *)(Bs + STAGES * BN * BKH), *Bsc = Asc + STAGES * BM * NBLK;  // per row and block: (d, dsum) / (s, m)

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / (BN / WN), wn = warp % (BN / WN);
  int bm, bn;
  float acc[MT][NT][4];
  unsigned af[2][MT][4], bf[2][NT][2];

  auto load_tile = [&](int stage, int kt) {
    half *as = As + stage * BM * BKH, *bs = Bs + stage * BN * BKH;
#pragma unroll
    for (int i = 0; i < (BM * BKH / 8) / NTHREADS; i++) {
      const int c = tid + i * NTHREADS, row = c / (BKH / 8), ch = c % (BKH / 8);
      cp_async16(as + kc_off(row, ch), X + (size_t)(bm + row) * XROW + kt * KB + ch * 16);
    }
#pragma unroll
    for (int i = 0; i < (BN * BKH / 8) / NTHREADS; i++) {
      const int c = tid + i * NTHREADS, row = c / (BKH / 8), ch = c % (BKH / 8);
      cp_async16(bs + kc_off(row, ch), W + (size_t)(bn + row) * K_DIM + kt * KB + ch * 16);
    }
    for (int c = tid; c < (BM + BN) * (NBLK / 2); c += NTHREADS) {  // 16 bytes = two blocks' scales per copy
      const int r = c / (NBLK / 2), h = c % (NBLK / 2);
      if (r < BM) cp_async16(Asc + (stage * BM + r) * NBLK + 2 * h, X + (size_t)(bm + r) * XROW + K_DIM + kt * NBLK * 8 + h * 16);
      else cp_async16(Bsc + (stage * BN + r - BM) * NBLK + 2 * h, WS + (size_t)(bn + r - BM) * (K_DIM / 32) + kt * NBLK + 2 * h);
    }
  };
  auto load_frags = [&](int buf, int stage, int kk) {
    const half *as = As + stage * BM * BKH, *bs = Bs + stage * BN * BKH;
#pragma unroll
    for (int mt = 0; mt < MT; mt++) ldmatrix_x4(af[buf][mt], as + kc_off(wm * WM + mt * 16 + (lane & 15), kk * 2 + (lane >> 4)));
#pragma unroll
    for (int nt = 0; nt < NT; nt += 2) {
      unsigned r[4];
      ldmatrix_x4(r, bs + kc_off(wn * WN + nt * 8 + ((lane >> 4) << 3) + (lane & 7), kk * 2 + ((lane >> 3) & 1)));
      bf[buf][nt][0] = r[0]; bf[buf][nt][1] = r[1]; bf[buf][nt + 1][0] = r[2]; bf[buf][nt + 1][1] = r[3];
    }
  };
  // one 32-wide quant block (kk of the stage): int8 mma, then scale into acc
  auto block = [&](int buf, int stage, int kk, bool q6) {
    const float2 *asc = Asc + stage * BM * NBLK + kk, *bsc = Bsc + stage * BN * NBLK + kk;
    const unsigned (&a)[MT][4] = af[buf];
    const unsigned (&bb)[NT][2] = bf[buf];
    float2 xs[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
      for (int h = 0; h < 2; h++) xs[mt][h] = asc[NBLK * (wm * WM + mt * 16 + (lane >> 2) + 8 * h)];
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
      const float2 w0 = bsc[NBLK * (wn * WN + nt * 8 + (lane & 3) * 2)], w1 = bsc[NBLK * (wn * WN + nt * 8 + (lane & 3) * 2 + 1)];
#pragma unroll
      for (int mt = 0; mt < MT; mt++) {
        if (!q6) {
          int c[4];
          mma_s8_k32(c, a[mt], bb[nt]);
#pragma unroll
          for (int e = 0; e < 4; e++) {
            const float2 x = xs[mt][e >> 1], w = (e & 1) ? w1 : w0;
#if HAS_MIN
            acc[mt][nt][e] = fmaf(i2f(c[e]), x.x * w.x, fmaf(-x.y, w.y, acc[mt][nt][e]));
#else
            acc[mt][nt][e] = fmaf(i2f(c[e]), x.x * w.x, acc[mt][nt][e]);
#endif
          }
        } else {
          int lo[4], hi[4];
          mma_s8_k16(lo, a[mt][0], a[mt][1], bb[nt][0]);
          mma_s8_k16(hi, a[mt][2], a[mt][3], bb[nt][1]);
#pragma unroll
          for (int e = 0; e < 4; e++) {
            const float2 x = xs[mt][e >> 1], w = (e & 1) ? w1 : w0;
            acc[mt][nt][e] = fmaf(x.x, fmaf(i2f(lo[e]), w.x, i2f(hi[e]) * w.y), acc[mt][nt][e]);
          }
        }
      }
    }
  };

  auto mainloop = [&](int k0, int k1) {
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
      for (int j = 0; j < NT; j++)
#pragma unroll
        for (int k = 0; k < 4; k++) acc[i][j][k] = 0.0f;
    const bool q6 = IS_Q6(bn);
    const int n = k1 - k0;
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
      if (s < n) load_tile(s, k0 + s);
      cp_async_commit();
    }
    cp_async_wait<STAGES - 2>();  // tile 0 has landed
    __syncthreads();
    load_frags(0, 0, 0);
    for (int i = 0; i < n; i++) {
      // slot (i-1)%STAGES was last read before the barrier at the end of iteration i-1, so it's free to overwrite
      const int ni = i + STAGES - 1;
      if (ni < n) load_tile(ni % STAGES, k0 + ni);
      cp_async_commit();
#pragma unroll
      for (int kk = 0; kk < NBLK; kk++) {
        if (kk < NBLK - 1) {  // the next block's fragments load while this block's mmas run
          load_frags((kk + 1) & 1, i % STAGES, kk + 1);
          block(kk & 1, i % STAGES, kk, q6);
        } else {  // the last block reads this stage's scales, so it runs before the barrier that frees the stage
          block(kk & 1, i % STAGES, kk, q6);
          cp_async_wait<STAGES - 2>();  // tile i+1 has landed
          __syncthreads();              // and every warp is done with stage i
          if (i + 1 < n) load_frags(0, (i + 1) % STAGES, 0);
        }
      }
    }
    __syncthreads();
  };

  auto store_tile = [&]() {
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
      for (int nt = 0; nt < NT; nt++) {
        const int row = bm + wm * WM + mt * 16 + (lane >> 2), col = bn + wn * WN + nt * 8 + (lane & 3) * 2;
#ifdef GLU
        const float a0 = glu_act(acc[mt][nt][0], acc[mt][nt][1]), a1 = glu_act(acc[mt][nt][2], acc[mt][nt][3]);
        const float b0 = __shfl_xor_sync(0xffffffff, a0, 1), b1 = __shfl_xor_sync(0xffffffff, a1, 1);
        if (!(lane & 1)) {
          store2(&C[(size_t)row * C_COLS + col / 2], a0, b0);
          store2(&C[(size_t)(row + 8) * C_COLS + col / 2], a1, b1);
        }
#else
        store2(&C[(size_t)row * N_DIM + col], acc[mt][nt][0], acc[mt][nt][1]);
        store2(&C[(size_t)(row + 8) * N_DIM + col], acc[mt][nt][2], acc[mt][nt][3]);
#endif
      }
  };

  const int pid = blockIdx.x;
  if (pid >= SK_WORKERS) {
    tile_coords(SK_TILES + pid - SK_WORKERS, bm, bn);
    mainloop(0, KT);
    store_tile();
    return;
  }
#if SK_P > 0
  // stream-K tail, as nv_gemm.cu: partial fp32 tiles handed to the worker that did the tile's head
  auto it_begin = [](int w) { return (int)(SK_ITERS * w / SK_WORKERS); };
  const int end = it_begin(pid + 1);
  for (int it = it_begin(pid); it < end;) {
    const int t = it / KT, k0 = it % KT, k1 = min(KT, k0 + end - it);
    tile_coords(t, bm, bn);
    mainloop(k0, k1);
    if (k0 != 0) {
      float4 *dst = (float4 *)(ws + (size_t)pid * BM * BN);
#pragma unroll
      for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
          __stcg(&dst[(mt * NT + nt) * NTHREADS + tid], make_float4(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2], acc[mt][nt][3]));
      __threadfence();
      __syncthreads();
      if (tid == 0) st_release(&flags[pid], 1);
    } else {
      for (int w = pid + 1; w < SK_WORKERS && it_begin(w) < (t + 1) * KT; w++) {
        if (tid == 0) {
          while (ld_acquire(&flags[w]) == 0) __nanosleep(64);
          flags[w] = 0;
        }
        __syncthreads();
        const float4 *src = (const float4 *)(ws + (size_t)w * BM * BN);
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
          for (int nt = 0; nt < NT; nt++) {
            float4 v = __ldcg(&src[(mt * NT + nt) * NTHREADS + tid]);
            acc[mt][nt][0] += v.x; acc[mt][nt][1] += v.y; acc[mt][nt][2] += v.z; acc[mt][nt][3] += v.w;
          }
      }
      store_tile();
    }
    it += k1 - k0;
  }
#endif
}
