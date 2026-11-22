// int8 tensor core GEMM for the LLM's prefill on GGUF quant weights (llama.cpp's "MMQ" idea) for Ampere (sm_80/86):
// C[M,N] = X @ W^T with X quantized to int8 per 32 values along K (nv_quant.cu's QUANTIZE: per row K int8, then K/32 (d, d*sum)).
// W, per 32-block w = s*q - m (m = 0 for Q8_0 and Q6_K): sum_k x w = dx*s*dot(qx, qw) - (dx*sum qx)*m. The weights are read either
// as the packed Q4_K / Q8_0 blocks themselves (decoded in the fragment loads: Q4_K nibbles with two ALU ops per register, the
// 6-bit scales per warp column) or, for Q6_K, from nv_quant.cu's REQUANT int8 + float2 per block, with a scale per 16 (two k16 mmas).
// On GA102 int8 mma runs at 4x the fp16-input/fp32-accumulate rate. The int8 fragments of mma.m16n8k32 have the same register
// layout as fp16 m16n8k16 ones over twice the bytes, so the A tiles, swizzle and ldmatrix loads are nv_gemm.cu's K-contiguous ones
// on bytes viewed as halves. After each block's mma the int32 dot is scaled into the fp32 accumulators.
// Epilogue: fp32 C[M][N], or with GLU (columns interleaved (gate, up) pairs) half C[M][N/2] of their SwiGLU, as nv_gemm.cu.
// nv.py prepends nv_common.cuh and #defines KNAME M_DIM N_DIM K_DIM BM BN WM WN STAGES NBLK GROUP_M SK_P [GLU...] and the segments:
// SEG_PARAMS (their pointers) and SEG_RUN(k0, k1) (mainloop with the format and pointers of the segment that owns column bn).
#ifndef GROUP_M
#define GROUP_M 8
#endif
#ifndef SK_P
#define SK_P 0
#endif
#ifndef SK_PART
#define SK_PART 0
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

// The int32 accumulators start at 0x4B400000, the bits of 1.5 * 2^23, so the tensor core adds the int -> float trick's offset: for
// |dot| < 2^22 (a block's is at most 32*127*127) the result's bits are the float 1.5 * 2^23 + dot, and i2f is one FADD instead
// of a quarter-rate I2F.
#define I2F_MAGIC 0x4B400000
__device__ __forceinline__ void mma_s8_k32(int (&c)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
      : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(I2F_MAGIC));
}
__device__ __forceinline__ void mma_s8_k16(int (&c)[4], unsigned a0, unsigned a1, unsigned b) {
  asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
      : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3]) : "r"(a0), "r"(a1), "r"(b), "r"(I2F_MAGIC));
}
__device__ __forceinline__ float i2f(int c) { return __int_as_float(c) - 12582912.0f; }  // c = I2F_MAGIC + dot

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

// B segment formats (uniform per tile; nv.py's SEG_RUN picks one per tile column bn):
//   F_I8 / F_I8Q6: requantized int8 [N][K] + float2 [N][K/32] (REQUANT), the Q6 variant with two scales per block
//   F_Q4K / F_Q8:  the packed GGUF blocks read as they are (+ F_IL: two weights' rows alternating, gate/up for GLU)
#define F_I8 0
#define F_I8Q6 1
#define F_Q4K 2
#define F_Q8 3
#define F_IL 4

#ifndef BROW
#define BROW 176  // B stage row stride in bytes, from the launch's formats (nv.py): Q4_K 112 (28 words), Q8_0 144 (36 words): the
#endif            // fragment loads' 8 rows hit distinct banks; requantized KB + NBLK * 8 (data + scales)
#ifndef HAS_RAW
#define HAS_RAW 1  // any packed-format segment: the per-warp decoded scales are allocated
#endif
static_assert(BROW % 16 == 0 && BROW >= 80, "a B stage row: Q4_K 80 bytes, Q8_0 136, requantized KB + NBLK * 8");
__device__ __forceinline__ void cp_async8(void *smem, const void *gmem) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ unsigned lds32(const unsigned char *p) { return *(const unsigned *)p; }
__device__ __forceinline__ unsigned lds16x2(const unsigned char *p) {  // 4 bytes at a 2-byte aligned address
  return (unsigned)*(const unsigned short *)p | ((unsigned)*(const unsigned short *)(p + 2) << 16);
}

// X: [M][XROW] bytes. smem per stage: A BM x KB int8 (swizzled), B BN x BROW bytes (requantized: BN x KB swizzled int8, then BN x
// NBLK float2; packed: each row's raw bytes for the stage), A scales BM x NBLK float2. Then BN x NBLK float2 B scales decoded from
// the packed headers: each warp decodes its WN columns' (the warps sharing columns write the same values, after the same barrier).
// The per-tile work is templated on the segment format F; St holds what the kernel's steps share (inlined: registers).
struct St {
  const unsigned char *X;
  half *As;
  unsigned char *Bs;
  float2 *Asc, *wsc;
  int tid, lane, wm, wn, bm, bn;
  float acc[MT][NT][4];
  unsigned af[2][MT][4], bf[2][NT][2];  // two fragment buffers when double-buffered (FRAG_DB), else only [0] is used
};

// wa/wb: the segment's pointers (requantized: int8 rows, float2 scales; packed: the blocks, and the second weight with F_IL),
// nb: the tile's first row in the segment
template <int F> __device__ __forceinline__ void load_tile(St &t, const unsigned char *wa, const void *wb, int nb, int stage, int kt) {
  half *as = t.As + stage * BM * BKH;
  unsigned char *bs = t.Bs + stage * BN * BROW;
#pragma unroll
  for (int i = 0; i < (BM * BKH / 8) / NTHREADS; i++) {
    const int c = t.tid + i * NTHREADS, row = c / (BKH / 8), ch = c % (BKH / 8);
    cp_async16(as + kc_off(row, ch), t.X + (size_t)(t.bm + row) * XROW + kt * KB + ch * 16);
  }
  for (int c = t.tid; c < BM * (NBLK / 2); c += NTHREADS) {  // 16 bytes = two blocks' (d, dsum)
    const int r = c / (NBLK / 2), h = c % (NBLK / 2);
    cp_async16(t.Asc + (stage * BM + r) * NBLK + 2 * h, t.X + (size_t)(t.bm + r) * XROW + K_DIM + kt * NBLK * 8 + h * 16);
  }
  if constexpr ((F & 3) <= F_I8Q6) {
    half *bh = (half *)bs;
#pragma unroll
    for (int i = 0; i < (BN * BKH / 8) / NTHREADS; i++) {
      const int c = t.tid + i * NTHREADS, row = c / (BKH / 8), ch = c % (BKH / 8);
      cp_async16(bh + kc_off(row, ch), wa + (size_t)(nb + row) * K_DIM + kt * KB + ch * 16);
    }
    float2 *bsc = (float2 *)(bs + BN * KB);
    for (int c = t.tid; c < BN * (NBLK / 2); c += NTHREADS) {
      const int r = c / (NBLK / 2), h = c % (NBLK / 2);
      cp_async16(bsc + r * NBLK + 2 * h, (const float2 *)wb + (size_t)(nb + r) * (K_DIM / 32) + kt * NBLK + 2 * h);
    }
  } else {
    static_assert(NBLK == 4, "packed formats: a stage is half a Q4_K superblock / four Q8_0 blocks");
    constexpr size_t RB = (F & 3) == F_Q4K ? K_DIM / 256 * 144 : K_DIM / 32 * 34;  // bytes per packed row
    if constexpr ((F & 3) == F_Q4K) {  // header (d, dmin, scales) and this half's 64 nibble bytes: 5 x 16 bytes per row
      for (int c = t.tid; c < BN * 5; c += NTHREADS) {
        const int r = c / 5, j = c % 5;
        const unsigned char *rp = (F & F_IL) ? ((r & 1) ? (const unsigned char *)wb : wa) + (size_t)((nb + r) >> 1) * RB : wa + (size_t)(nb + r) * RB;
        cp_async16(bs + r * BROW + 16 * j, rp + (size_t)(kt >> 1) * 144 + (j ? 16 + 64 * (kt & 1) + 16 * (j - 1) : 0));
      }
    } else {  // four 34-byte blocks, 8-byte aligned: 17 x 8 bytes per row
      for (int c = t.tid; c < BN * 17; c += NTHREADS) {
        const int r = c / 17, j = c % 17;
        const unsigned char *rp = (F & F_IL) ? ((r & 1) ? (const unsigned char *)wb : wa) + (size_t)((nb + r) >> 1) * RB : wa + (size_t)(nb + r) * RB;
        cp_async8(bs + r * BROW + 8 * j, rp + (size_t)kt * 136 + 8 * j);
      }
    }
  }
}
// after a stage has landed: each lane decodes its column's scales for the stage's blocks (packed formats)
template <int F> __device__ __forceinline__ void decode_scales(St &t, int stage, int kt) {
  if constexpr ((F & 3) >= F_Q4K) {
    const unsigned char *b = t.Bs + stage * BN * BROW + (t.wn * WN + t.lane) * BROW;
#pragma unroll
    for (int g = 0; g < WN / 32; g++, b += 32 * BROW) {
      if constexpr ((F & 3) == F_Q4K) {
        const float d = __half2float(*(const half *)b), dmin = __half2float(*(const half *)(b + 2));
        const unsigned char *sc = b + 4;
#pragma unroll
        for (int kk = 0; kk < NBLK; kk++) {
          const int j = 4 * (kt & 1) + kk;
          const int sj = j < 4 ? sc[j] & 63 : (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
          const int mj = j < 4 ? sc[j + 4] & 63 : (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
          t.wsc[(g * 32 + t.lane) * NBLK + kk] = make_float2(d * sj, dmin * mj);
        }
      } else {
#pragma unroll
        for (int kk = 0; kk < NBLK; kk++) t.wsc[(g * 32 + t.lane) * NBLK + kk] = make_float2(__half2float(*(const half *)(b + 34 * kk)), 0.0f);
      }
    }
    __syncwarp();
  }
}
template <int F> __device__ __forceinline__ void load_frags(St &t, int buf, int stage, int kk) {
  const half *as = t.As + stage * BM * BKH;
  const unsigned char *bs = t.Bs + stage * BN * BROW;
  const int lane = t.lane;
#pragma unroll
  for (int mt = 0; mt < MT; mt++) ldmatrix_x4(t.af[buf][mt], as + kc_off(t.wm * WM + mt * 16 + (lane & 15), kk * 2 + (lane >> 4)));
  if constexpr ((F & 3) <= F_I8Q6) {
#pragma unroll
    for (int nt = 0; nt < NT; nt += 2) {
      unsigned r[4];
      ldmatrix_x4(r, (const half *)bs + kc_off(t.wn * WN + nt * 8 + ((lane >> 4) << 3) + (lane & 7), kk * 2 + ((lane >> 3) & 1)));
      t.bf[buf][nt][0] = r[0]; t.bf[buf][nt][1] = r[1]; t.bf[buf][nt + 1][0] = r[2]; t.bf[buf][nt + 1][1] = r[3];
    }
  } else {
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
      const unsigned char *row = bs + (t.wn * WN + nt * 8 + (lane >> 2)) * BROW;
      if constexpr ((F & 3) == F_Q4K) {  // block kk = nibbles (kk & 1) of qs chunk kk / 2: bytes 4 (lane & 3) .. and +16
        const unsigned char *q = row + 16 + 32 * (kk >> 1) + 4 * (lane & 3);
        t.bf[buf][nt][0] = (lds32(q) >> (4 * (kk & 1))) & 0x0F0F0F0F;
        t.bf[buf][nt][1] = (lds32(q + 16) >> (4 * (kk & 1))) & 0x0F0F0F0F;
      } else {
        const unsigned char *q = row + 34 * kk + 2 + 4 * (lane & 3);
        t.bf[buf][nt][0] = lds16x2(q);
        t.bf[buf][nt][1] = lds16x2(q + 16);
      }
    }
  }
}
// one 32-wide quant block (kk of the stage): int8 mma, then scale into acc
template <int F> __device__ __forceinline__ void block(St &t, int buf, int stage, int kk) {
  const int lane = t.lane;
  const float2 *asc = t.Asc + stage * BM * NBLK + kk;
  const float2 *bsc = (F & 3) <= F_I8Q6 ? (const float2 *)(t.Bs + stage * BN * BROW + BN * KB) + t.wn * WN * NBLK + kk : t.wsc + kk;
  float2 xs[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; mt++)
#pragma unroll
    for (int h = 0; h < 2; h++) xs[mt][h] = asc[NBLK * (t.wm * WM + mt * 16 + (lane >> 2) + 8 * h)];
#pragma unroll
  for (int nt = 0; nt < NT; nt++) {
    const float2 w0 = bsc[NBLK * (nt * 8 + (lane & 3) * 2)], w1 = bsc[NBLK * (nt * 8 + (lane & 3) * 2 + 1)];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
      if constexpr (F != F_I8Q6) {
        int c[4];
        mma_s8_k32(c, t.af[buf][mt], t.bf[buf][nt]);
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const float2 x = xs[mt][e >> 1], w = (e & 1) ? w1 : w0;
          if constexpr ((F & 3) == F_Q8) t.acc[mt][nt][e] = fmaf(i2f(c[e]), x.x * w.x, t.acc[mt][nt][e]);
          else t.acc[mt][nt][e] = fmaf(i2f(c[e]), x.x * w.x, fmaf(-x.y, w.y, t.acc[mt][nt][e]));
        }
      } else {
        // each half's dot is scaled as soon as it's out: 4 live int32s instead of 8 (the two at once spilled)
        int c[4];
        mma_s8_k16(c, t.af[buf][mt][0], t.af[buf][mt][1], t.bf[buf][nt][0]);
#pragma unroll
        for (int e = 0; e < 4; e++) t.acc[mt][nt][e] = fmaf(i2f(c[e]), xs[mt][e >> 1].x * ((e & 1) ? w1 : w0).x, t.acc[mt][nt][e]);
        mma_s8_k16(c, t.af[buf][mt][2], t.af[buf][mt][3], t.bf[buf][nt][1]);
#pragma unroll
        for (int e = 0; e < 4; e++) t.acc[mt][nt][e] = fmaf(i2f(c[e]), xs[mt][e >> 1].x * ((e & 1) ? w1 : w0).y, t.acc[mt][nt][e]);
      }
    }
  }
}
template <int F> __device__ __forceinline__ void mainloop(St &t, const unsigned char *wa, const void *wb, int nb, int k0, int k1) {
#pragma unroll
  for (int i = 0; i < MT; i++)
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
      for (int k = 0; k < 4; k++) t.acc[i][j][k] = 0.0f;
  const int n = k1 - k0;
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < n) load_tile<F>(t, wa, wb, nb, s, k0 + s);
    cp_async_commit();
  }
  cp_async_wait<STAGES - 2>();  // tile 0 has landed
  __syncthreads();
  decode_scales<F>(t, 0, k0);
  // double-buffered fragments (the next block's loaded during this one's mmas) are faster for the two-scale Q6_K path; the others
  // run better with one buffer: fewer registers (they spilled), the loads' latency hidden by the other warps
  constexpr bool FRAG_DB = F == F_I8Q6;
  if constexpr (FRAG_DB) load_frags<F>(t, 0, 0, 0);
  for (int i = 0; i < n; i++) {
    // slot (i-1)%STAGES was last read before the barrier at the end of iteration i-1, so it's free to overwrite
    const int ni = i + STAGES - 1;
    if (ni < n) load_tile<F>(t, wa, wb, nb, ni % STAGES, k0 + ni);
    cp_async_commit();
    if constexpr (FRAG_DB) {
#pragma unroll
    for (int kk = 0; kk < NBLK; kk++) {
      if (kk < NBLK - 1) {  // the next block's fragments load while this block's mmas run
        load_frags<F>(t, (kk + 1) & 1, i % STAGES, kk + 1);
        block<F>(t, kk & 1, i % STAGES, kk);
      } else {  // the last block reads this stage's scales, so it runs before the barrier that frees the stage
        block<F>(t, kk & 1, i % STAGES, kk);
        cp_async_wait<STAGES - 2>();  // tile i+1 has landed
        __syncthreads();              // and every warp is done with stage i
        if (i + 1 < n) {
          decode_scales<F>(t, (i + 1) % STAGES, k0 + i + 1);
          load_frags<F>(t, 0, (i + 1) % STAGES, 0);
        }
      }
    }
    } else {
#pragma unroll
    for (int kk = 0; kk < NBLK; kk++) {
      load_frags<F>(t, 0, i % STAGES, kk);
      block<F>(t, 0, i % STAGES, kk);
    }
    cp_async_wait<STAGES - 2>();  // tile i+1 has landed
    __syncthreads();              // and every warp is done with stage i
    if (i + 1 < n) decode_scales<F>(t, (i + 1) % STAGES, k0 + i + 1);
    }
  }
  __syncthreads();
}


extern "C" __global__ void __launch_bounds__(NTHREADS)
KNAME(out_t *__restrict__ C, const unsigned char *__restrict__ X, SEG_PARAMS, float *__restrict__ ws, int *__restrict__ flags) {
  extern __shared__ __align__(128) unsigned char smem[];
  St t;
  t.X = X;
  t.As = (half *)smem;
  t.Bs = (unsigned char *)(t.As + STAGES * BM * BKH);
  t.Asc = (float2 *)(t.Bs + STAGES * BN * BROW);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  t.tid = tid, t.lane = lane, t.wm = warp / (BN / WN), t.wn = warp % (BN / WN);
  t.wsc = t.Asc + STAGES * BM * NBLK + t.wn * WN * NBLK;  // the decoded B scales of this warp's columns: [WN][NBLK]
  int &bm = t.bm, &bn = t.bn;
  const int wm = t.wm, wn = t.wn;
  float (&acc)[MT][NT][4] = t.acc;

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

  // SK_PART 1 launches the whole tiles (grid DP_TILES), SK_PART 2 the stream-K workers (grid SK_WORKERS), one after the other: each
  // kernel then has one copy of the main loop (both in one kernel spilled registers in the loop). SK_PART 0: both, one launch.
#if SK_PART == 2
  const int pid = blockIdx.x;
#else
  const int pid = blockIdx.x + (SK_PART == 1 ? SK_WORKERS : 0);
  if (pid >= SK_WORKERS) {
    tile_coords(SK_TILES + pid - SK_WORKERS, bm, bn);
    SEG_RUN(0, KT)
    store_tile();
    return;
  }
#endif
#if SK_P > 0 && SK_PART != 1
  // stream-K tail, as nv_gemm.cu: partial fp32 tiles handed to the worker that did the tile's head
  auto it_begin = [](int w) { return (int)(SK_ITERS * w / SK_WORKERS); };
  const int end = it_begin(pid + 1);
  for (int it = it_begin(pid); it < end;) {
    const int tl = it / KT, k0 = it % KT, k1 = min(KT, k0 + end - it);
    tile_coords(tl, bm, bn);
    SEG_RUN(k0, k1)
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
      for (int w = pid + 1; w < SK_WORKERS && it_begin(w) < (tl + 1) * KT; w++) {
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
