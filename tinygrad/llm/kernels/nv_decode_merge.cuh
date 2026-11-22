// Flash decoding's split layout and merge, shared by nv_decode.cu (COMBINE) and nv_quant.cu (QUANTIZE with ATTN, which merges the
// splits while quantizing the attention output for the output projection's GEMV, so the merged output is never written).
// A split's partial is [HD + 2] fp32: the output o scaled by 2^-m, then m (the max score, log2 units), then l (the softmax sum).
#ifndef NV_DECODE_MERGE
#define NV_DECODE_MERGE
#define NEG_INF __int_as_float(0xff800000)  // -inf (nvrtc --minimal has no math.h INFINITY)

// keys per split for valid_end keys over nsplit splits, at least minkeys (one full step of a block): at short lengths the later
// splits are empty and do nothing. decode_active_splits is how many have keys (>= 1, valid_end >= 1).
__device__ __forceinline__ int decode_split_keys(int valid_end, int nsplit, int minkeys) {
  return max((valid_end + nsplit - 1) / nsplit, minkeys);
}
__device__ __forceinline__ int decode_active_splits(int valid_end, int nsplit, int minkeys) {
  const int per = decode_split_keys(valid_end, nsplit, minkeys);
  return (valid_end + per - 1) / per;
}

// One warp: the attention output at head dim d (each lane its own) from the ns <= NS <= 64 non-empty partials at pp ([ns][hd + 2]).
// Every load (each lane's o over the splits, and the splits' m and l spread over the lanes) is issued before any is used: one
// round trip to L2, where the partials just written are.
template <int NS> __device__ __forceinline__ float decode_merge(const float *__restrict__ pp, int ns, int hd, int lane, int d) {
  static_assert(NS <= 64, "at most 64 splits");
  const int stride = hd + 2;
  float ov[NS];
#pragma unroll
  for (int s = 0; s < NS; s++) ov[s] = s < ns ? pp[s * stride + d] : 0.0f;
  const bool h0 = lane < ns, h1 = NS > 32 && lane + 32 < ns;
  const float m0 = h0 ? pp[lane * stride + hd] : NEG_INF, m1 = h1 ? pp[(lane + 32) * stride + hd] : NEG_INF;
  const float l0 = h0 ? pp[lane * stride + hd + 1] : 0.0f, l1 = h1 ? pp[(lane + 32) * stride + hd + 1] : 0.0f;
  float M = fmaxf(m0, m1);
#pragma unroll
  for (int off = 16; off; off >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffff, M, off));
  const float w0 = h0 ? exp2f(m0 - M) : 0.0f, w1 = h1 ? exp2f(m1 - M) : 0.0f;
  float L = l0 * w0 + l1 * w1;
#pragma unroll
  for (int off = 16; off; off >>= 1) L += __shfl_xor_sync(0xffffffff, L, off);
  float o = 0.0f;
#pragma unroll
  for (int s = 0; s < NS; s++) o += __shfl_sync(0xffffffff, s < 32 ? w0 : w1, s & 31) * ov[s];  // w = 0 past ns
  return o / L;
}
#endif
