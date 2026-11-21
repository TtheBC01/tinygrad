// fp16 GEMM for Ampere (sm_80/86): C[M,N] = op(A) @ op(B), row-major, fp32 accumulate, fp16 output.
// A is stored [M][K], or [K][M] with TRANS_A=1 (C = A^T @ ...). B is stored [K][N], or [N][K] with TRANS_B=1 (C = ... @ B^T).
// The transposed forms are the backward pass: dA = dC @ B^T (TRANS_B), dB = A^T @ dC (TRANS_A), read without transposing first.
// Block tile BMxBNxBK, warp tile WMxWN, STAGES-deep cp.async pipeline, swizzled shared memory,
// ldmatrix (+ .trans for B) feeding mma.sync.m16n8k16.
// Stream-K tail (SK_P > 0): tiles that would leave the last wave partly empty are split along K across all SK_P block slots,
// and partial sums are merged through a fp32 workspace, so every SM stays busy to the end.
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
#ifndef TRANS_A
#define TRANS_A 0
#endif
#ifndef TRANS_B
#define TRANS_B 0
#endif
#ifndef SK_P
#define SK_P 0  // concurrent block slots on the GPU (SMs x resident blocks per SM); 0 disables stream-K
#endif

#define NWARPS ((BM / WM) * (BN / WN))
#define NTHREADS (NWARPS * 32)
#define MT (WM / 16)  // m16 tiles per warp
#define NT (WN / 8)   // n8 tiles per warp
#define KK (BK / 16)  // k16 steps per k tile

static_assert(BK == 32, "the K-contiguous swizzle assumes 4 x 16B chunks per row");
static_assert(BN >= 64 && BM >= 64, "the outer-contiguous swizzle needs >= 8 x 16B chunks per row");
static_assert(NT % 2 == 0, "B fragments are loaded two n8 tiles at a time");
static_assert((BM * BK / 8) % NTHREADS == 0 && (BK * BN / 8) % NTHREADS == 0, "tile loads must divide evenly");
static_assert(M_DIM % BM == 0 && N_DIM % BN == 0 && K_DIM % BK == 0, "shape must be a multiple of the tile");

// Work split. With T tiles and P slots, T % P tiles would run as a partial last wave. Instead the last T % P + P tiles
// (all of them if T < 2P) are shared out evenly by K iteration across P stream-K workers; the rest are whole data-parallel tiles.
// The host launches GEMM_GRID blocks: SK_WORKERS stream-K workers first, then DP_TILES data-parallel blocks.
// ampere_gemm.py mirrors this split, keep them in sync.
constexpr int NBX = N_DIM / BN, NBY = M_DIM / BM, TILES = NBX * NBY, KT = K_DIM / BK;
constexpr int SK_P1 = SK_P > 0 ? SK_P : 1;
constexpr int SK_TILES = (SK_P == 0 || TILES % SK_P1 == 0) ? 0 : (TILES >= 2 * SK_P1 ? TILES % SK_P1 + SK_P1 : TILES);
constexpr long long SK_ITERS = (long long)SK_TILES * KT;
constexpr int SK_WORKERS = SK_ITERS < SK_P ? (int)SK_ITERS : SK_P;  // (0 when SK_TILES is 0)  // every worker gets at least one K iteration
constexpr int DP_TILES = TILES - SK_TILES, GEMM_GRID = SK_WORKERS + DP_TILES;

// Shared memory stages come in two layouts, picked by how the operand is stored in global memory:
// K-contiguous [rows][BK] (A, or B with TRANS_B): rows are 64B, so two rows share a 128B bank line: XOR chunk with (row>>1)&3.
// ldmatrix reads 8 consecutive rows at one k-chunk, which hits 8 distinct 16B bank groups.
__device__ __forceinline__ int kc_off(int row, int chunk) { return row * BK + ((chunk ^ ((row >> 1) & 3)) << 3); }
// outer-contiguous [BK][W] (B, or A with TRANS_A): ldmatrix.trans reads 8 consecutive k-rows at one chunk: XOR chunk with row&7.
template <int W> __device__ __forceinline__ int oc_off(int row, int chunk) { return row * W + ((chunk ^ (row & 7)) << 3); }

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
__device__ __forceinline__ int ld_acquire(const int *p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release(int *p, int v) { asm volatile("st.release.gpu.global.b32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory"); }

// tile index -> block row/col. GROUP_M > 0 walks GROUP_M block-rows together so concurrently running blocks share A and B in L2
__device__ __forceinline__ void tile_coords(int t, int &bm, int &bn) {
#if GROUP_M > 0
  const int first_m = (t / (GROUP_M * NBX)) * GROUP_M, gsz = (NBY - first_m) < GROUP_M ? (NBY - first_m) : GROUP_M;
  bm = (first_m + (t % gsz)) * BM, bn = ((t % (GROUP_M * NBX)) / gsz) * BN;
#else
  bm = (t / NBX) * BM, bn = (t % NBX) * BN;
#endif
}

__device__ __forceinline__ void mma16816(float *c, const unsigned *a, const unsigned *b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

extern "C" __global__ void __launch_bounds__(NTHREADS)
KNAME(half *__restrict__ C, const half *__restrict__ A, const half *__restrict__ B, float *__restrict__ ws, int *__restrict__ flags) {
  // dynamic shared memory: STAGES*(BM*BK + BK*BN)*2 bytes, reserved by the runtime via KernelInfo.dyn_shared_mem (can exceed 48 KB)
  // ws: SK_WORKERS x BM*BN fp32 partial sums. flags: SK_WORKERS ints, 0 between launches (each one is reset by the worker that reads it)
  extern __shared__ __align__(128) unsigned char smem[];
  half *As = (half *)smem, *Bs = As + STAGES * BM * BK;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / (BN / WN), wn = warp % (BN / WN);
  int bm, bn;
  float acc[MT][NT][4];
  unsigned af[2][MT][4], bf[2][NT][2];

  auto load_tile = [&](int stage, int kt) {
    half *as = As + stage * BM * BK, *bs = Bs + stage * BK * BN;
#pragma unroll
    for (int i = 0; i < (BM * BK / 8) / NTHREADS; i++) {
#if TRANS_A  // A stored [K][M]: BK rows of BM
      int c = tid + i * NTHREADS, row = c / (BM / 8), ch = c % (BM / 8);
      cp_async16(as + oc_off<BM>(row, ch), A + (size_t)(kt * BK + row) * M_DIM + bm + ch * 8);
#else        // A stored [M][K]: BM rows of BK
      int c = tid + i * NTHREADS, row = c / (BK / 8), ch = c % (BK / 8);
      cp_async16(as + kc_off(row, ch), A + (size_t)(bm + row) * K_DIM + kt * BK + ch * 8);
#endif
    }
#pragma unroll
    for (int i = 0; i < (BK * BN / 8) / NTHREADS; i++) {
#if TRANS_B  // B stored [N][K]: BN rows of BK
      int c = tid + i * NTHREADS, row = c / (BK / 8), ch = c % (BK / 8);
      cp_async16(bs + kc_off(row, ch), B + (size_t)(bn + row) * K_DIM + kt * BK + ch * 8);
#else        // B stored [K][N]: BK rows of BN
      int c = tid + i * NTHREADS, row = c / (BN / 8), ch = c % (BN / 8);
      cp_async16(bs + oc_off<BN>(row, ch), B + (size_t)(kt * BK + row) * N_DIM + bn + ch * 8);
#endif
    }
  };

  // shared -> register fragments for one k16 step of this warp's tile. Lane group j = lane>>3 addresses 8x8 matrix j:
  // A fragment matrices are (m0-7,k0-7) (m8-15,k0-7) (m0-7,k8-15) (m8-15,k8-15); a pair of n8 B fragments is
  // (k0-7,n0-7) (k8-15,n0-7) (k0-7,n8-15) (k8-15,n8-15). A K-contiguous stage is read with ldmatrix, an outer-contiguous one
  // with ldmatrix.trans, so both give the fragments mma.sync expects.
  auto load_frags = [&](int buf, int stage, int kk) {
    const half *as = As + stage * BM * BK, *bs = Bs + stage * BK * BN;
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#if TRANS_A
      ldmatrix_x4_trans(af[buf][mt], as + oc_off<BM>(kk * 16 + ((lane >> 4) << 3) + (lane & 7), (wm * WM + mt * 16) / 8 + ((lane >> 3) & 1)));
#else
      ldmatrix_x4(af[buf][mt], as + kc_off(wm * WM + mt * 16 + (lane & 15), kk * 2 + (lane >> 4)));
#endif
#pragma unroll
    for (int nt = 0; nt < NT; nt += 2) {
      unsigned r[4];
#if TRANS_B
      ldmatrix_x4(r, bs + kc_off(wn * WN + nt * 8 + ((lane >> 4) << 3) + (lane & 7), kk * 2 + ((lane >> 3) & 1)));
#else
      ldmatrix_x4_trans(r, bs + oc_off<BN>(kk * 16 + (lane & 15), (wn * WN + nt * 8) / 8 + (lane >> 4)));
#endif
      bf[buf][nt][0] = r[0]; bf[buf][nt][1] = r[1]; bf[buf][nt + 1][0] = r[2]; bf[buf][nt + 1][1] = r[3];
    }
  };

  // acc = A[bm:bm+BM, k0*BK:k1*BK] @ B[k0*BK:k1*BK, bn:bn+BN]
  auto mainloop = [&](int k0, int k1) {
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
      for (int j = 0; j < NT; j++)
#pragma unroll
        for (int k = 0; k < 4; k++) acc[i][j][k] = 0.0f;
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
      int ni = i + STAGES - 1;
      if (ni < n) load_tile(ni % STAGES, k0 + ni);
      cp_async_commit();
#pragma unroll
      for (int kk = 0; kk < KK; kk++) {
        if (kk == KK - 1) {
          cp_async_wait<STAGES - 2>();  // tile i+1 has landed
          __syncthreads();              // and every warp is done reading stage i from shared memory
        }
        // prefetch the next k16 step's fragments (from the next stage on the last step) while this step's mmas run
        if (kk < KK - 1) load_frags((kk + 1) & 1, i % STAGES, kk + 1);
        else if (i + 1 < n) load_frags((kk + 1) & 1, (i + 1) % STAGES, 0);
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
          for (int nt = 0; nt < NT; nt++) mma16816(acc[mt][nt], af[kk & 1][mt], bf[kk & 1][nt]);
      }
    }
    // the last barrier above came after every shared memory read of this call, so a following call can refill the stages
  };

  auto store_tile = [&]() {
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
      for (int nt = 0; nt < NT; nt++) {
        int row = bm + wm * WM + mt * 16 + (lane >> 2), col = bn + wn * WN + nt * 8 + (lane & 3) * 2;
        *(half2 *)&C[(size_t)row * N_DIM + col] = __floats2half2_rn(acc[mt][nt][0], acc[mt][nt][1]);
        *(half2 *)&C[(size_t)(row + 8) * N_DIM + col] = __floats2half2_rn(acc[mt][nt][2], acc[mt][nt][3]);
      }
  };

  const int pid = blockIdx.x;
  if (pid >= SK_WORKERS) {  // data-parallel: one whole tile
    tile_coords(SK_TILES + pid - SK_WORKERS, bm, bn);
    mainloop(0, KT);
    store_tile();
    return;
  }

#if SK_P > 0
  // stream-K worker: K iterations [it_begin(pid), it_begin(pid+1)) of the SK tiles laid end to end. That range is
  // [tail of a tile] [whole tiles...] [head of a tile]. A tail segment hands its partial sums to the worker that did the
  // tile's head (k=0), which runs last in that worker's range, adds every partial in worker order and stores the tile.
  // Only a head segment waits, on later workers that publish their tail first, so there's no circular wait.
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
          flags[w] = 0;  // only this worker reads w's partial, so reset it for the next launch
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
