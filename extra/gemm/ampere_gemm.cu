// fp16 GEMM for Ampere (sm_80/86): C[M,N] = A[M,K] @ B[K,N], row-major, fp32 accumulate, fp16 output.
// Block tile BMxBNxBK, warp tile WMxWN, STAGES-deep cp.async pipeline, swizzled shared memory,
// ldmatrix (+ .trans for B) feeding mma.sync.m16n8k16.
// The host side (extra/gemm/ampere_gemm.py) prepends #defines for KNAME, M_DIM, N_DIM, K_DIM and the tile config.
#include <cuda_fp16.h>

#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#ifndef BK
#define BK 32
#endif
#ifndef WM
#define WM 64
#endif
#ifndef WN
#define WN 64
#endif
#ifndef STAGES
#define STAGES 3
#endif
#ifndef GROUP_M
#define GROUP_M 0
#endif

#define NWARPS ((BM / WM) * (BN / WN))
#define NTHREADS (NWARPS * 32)
#define MT (WM / 16)  // m16 tiles per warp
#define NT (WN / 8)   // n8 tiles per warp

static_assert(BK == 32, "A swizzle assumes 4 x 16B chunks per row");
static_assert(BN >= 64, "B swizzle needs >= 8 x 16B chunks per row");
static_assert(NT % 2 == 0, "B fragments are loaded two n8 tiles at a time");
static_assert((BM * BK / 8) % NTHREADS == 0 && (BK * BN / 8) % NTHREADS == 0, "tile loads must divide evenly");
static_assert(M_DIM % BM == 0 && N_DIM % BN == 0 && K_DIM % BK == 0, "shape must be a multiple of the tile");

// A stage: [BM][BK] halves. Rows are 64B, so two rows share a 128B bank line: XOR chunk with (row>>1)&3.
__device__ __forceinline__ int a_off(int row, int chunk) { return row * BK + ((chunk ^ ((row >> 1) & 3)) << 3); }
// B stage: [BK][BN] halves. ldmatrix.trans reads 8 consecutive k-rows at one n-chunk: XOR chunk with row&7.
__device__ __forceinline__ int b_off(int row, int chunk) { return row * BN + ((chunk ^ (row & 7)) << 3); }

__device__ __forceinline__ void cp_async16(void *smem, const void *gmem) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(unsigned *r, const half *p) {
  unsigned s = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void ldmatrix_x4_trans(unsigned *r, const half *p) {
  unsigned s = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void mma16816(float *c, const unsigned *a, const unsigned *b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

extern "C" __global__ void __launch_bounds__(NTHREADS) KNAME(half *__restrict__ C, const half *__restrict__ A, const half *__restrict__ B) {
  // dynamic shared memory: STAGES*(BM*BK + BK*BN)*2 bytes, reserved by the runtime via KernelInfo.dyn_shared_mem (can exceed 48 KB)
  extern __shared__ __align__(128) unsigned char smem[];
  half *As = (half *)smem, *Bs = As + STAGES * BM * BK;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / (BN / WN), wn = warp % (BN / WN);
#if GROUP_M > 0
  // grouped rasterization: walk GROUP_M block-rows together so concurrently running blocks share A and B panels in L2
  constexpr int NBX = N_DIM / BN, NBY = M_DIM / BM;
  const int pid = blockIdx.y * NBX + blockIdx.x, first_m = (pid / (GROUP_M * NBX)) * GROUP_M;
  const int gsz = (NBY - first_m) < GROUP_M ? (NBY - first_m) : GROUP_M;
  const int bm = (first_m + (pid % gsz)) * BM, bn = ((pid % (GROUP_M * NBX)) / gsz) * BN;
#else
  const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;
#endif
  const half *Ag = A + (size_t)bm * K_DIM;
  const half *Bg = B + bn;

  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; i++)
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
      for (int k = 0; k < 4; k++) acc[i][j][k] = 0.0f;

  auto load_tile = [&](int stage, int kt) {
    half *as = As + stage * BM * BK, *bs = Bs + stage * BK * BN;
#pragma unroll
    for (int i = 0; i < (BM * BK / 8) / NTHREADS; i++) {
      int c = tid + i * NTHREADS, row = c / (BK / 8), ch = c % (BK / 8);
      cp_async16(as + a_off(row, ch), Ag + (size_t)row * K_DIM + kt * BK + ch * 8);
    }
#pragma unroll
    for (int i = 0; i < (BK * BN / 8) / NTHREADS; i++) {
      int c = tid + i * NTHREADS, row = c / (BN / 8), ch = c % (BN / 8);
      cp_async16(bs + b_off(row, ch), Bg + (size_t)(kt * BK + row) * N_DIM + ch * 8);
    }
  };

  // shared -> register fragments for one k16 step of this warp's tile
  unsigned af[2][MT][4], bf[2][NT][2];
  auto load_frags = [&](int buf, int stage, int kk) {
    const half *as = As + stage * BM * BK, *bs = Bs + stage * BK * BN;
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
      ldmatrix_x4(af[buf][mt], as + a_off(wm * WM + mt * 16 + (lane & 15), kk * 2 + (lane >> 4)));
#pragma unroll
    for (int nt = 0; nt < NT; nt += 2) {
      unsigned r[4];
      ldmatrix_x4_trans(r, bs + b_off(kk * 16 + (lane & 15), (wn * WN + nt * 8) / 8 + (lane >> 4)));
      bf[buf][nt][0] = r[0]; bf[buf][nt][1] = r[1]; bf[buf][nt + 1][0] = r[2]; bf[buf][nt + 1][1] = r[3];
    }
  };

  constexpr int KT = K_DIM / BK, KK = BK / 16;
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < KT) load_tile(s, s);
    cp_async_commit();
  }
  cp_async_wait<STAGES - 2>();  // tile 0 has landed
  __syncthreads();
  load_frags(0, 0, 0);

  for (int kt = 0; kt < KT; kt++) {
    // slot (kt-1)%STAGES was last read before the barrier at the end of iteration kt-1, so it's free to overwrite
    int nk = kt + STAGES - 1;
    if (nk < KT) load_tile(nk % STAGES, nk);
    cp_async_commit();
#pragma unroll
    for (int kk = 0; kk < KK; kk++) {
      if (kk == KK - 1) {
        cp_async_wait<STAGES - 2>();  // tile kt+1 has landed
        __syncthreads();              // and every warp is done reading stage kt from shared memory
      }
      // prefetch the next k16 step's fragments (from the next stage on the last step) while this step's mmas run
      if (kk < KK - 1) load_frags((kk + 1) & 1, kt % STAGES, kk + 1);
      else if (kt + 1 < KT) load_frags((kk + 1) & 1, (kt + 1) % STAGES, 0);
#pragma unroll
      for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++) mma16816(acc[mt][nt], af[kk & 1][mt], bf[kk & 1][nt]);
    }
  }

#pragma unroll
  for (int mt = 0; mt < MT; mt++)
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
      int row = bm + wm * WM + mt * 16 + (lane >> 2), col = bn + wn * WN + nt * 8 + (lane & 3) * 2;
      *(half2 *)&C[(size_t)row * N_DIM + col] = __floats2half2_rn(acc[mt][nt][0], acc[mt][nt][1]);
      *(half2 *)&C[(size_t)(row + 8) * N_DIM + col] = __floats2half2_rn(acc[mt][nt][2], acc[mt][nt][3]);
    }
}
