// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cuda_runtime.h>
#include <cstdio>

#include <flashinfer/attention/sparse_mla_sm120/decode_v3_kernel.cuh>
#include <flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh>

namespace flashinfer::sparse_mla_sm120 {

#define CUDA_CHECK_BOOL(call)                                     \
  do {                                                            \
    cudaError_t e = (call);                                       \
    if (e != cudaSuccess) {                                       \
      printf("CUDA %s:%d %s\n", __FILE__, __LINE__,               \
             cudaGetErrorString(e));                              \
      return false;                                               \
    }                                                             \
  } while (0)

template <ModelType MT, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE>
static bool launch_decode_v3_impl(const bf16* Q, const uint8_t* KV_cache,
                                  const int32_t* indices, bf16* mid_out,
                                  float* mid_lse, const int* topk_length,
                                  bf16* output, float* out_lse,
                                  int num_tokens, int num_splits,
                                  float sm_scale, size_t stride_kv_block,
                                  cudaStream_t stream) {
  using KV = KVCacheTraits<MT>;
  constexpr int H_BLOCKS = NUM_HEADS / HPB;

  // Stage 1: decode-v3 partial-output kernel.
  // Dynamic smem layout (matches kernel allocation):
  //   sm_q          [HPB, D_QK]            bf16  = 16 KB
  //   sm_kv         [V3_BI, D_QK]          bf16  = 64 KB
  //   sm_warp_max   [V3_N_WARPS * HPB]     float = 0.25 KB
  //   sm_warp_sum   [V3_N_WARPS * HPB]     float = 0.25 KB
  //   sm_head_buf   [D_V]                   float = 2 KB
  // Static smem:
  //   sm_p_storage  [N_WARPS, HPB, ENTRIES_PER_WARP] float = 4 KB
  constexpr int DYN_SMEM_BYTES =
      HPB * KV::D_QK * (int)sizeof(bf16) + V3_BI * KV::D_QK * (int)sizeof(bf16) +
      2 * V3_N_WARPS * HPB * (int)sizeof(float) + KV::D_V * (int)sizeof(float);

  auto kernel = sparse_mla_decode_v3_kernel<MT, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE>;
  CUDA_CHECK_BOOL(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM_BYTES));

  dim3 grid1(num_tokens, H_BLOCKS, num_splits);
  dim3 block1(V3_BLOCK_THREADS);
  kernel<<<grid1, block1, DYN_SMEM_BYTES, stream>>>(Q, KV_cache, indices, mid_out,
                                                   mid_lse, topk_length, num_tokens,
                                                   num_splits, sm_scale, stride_kv_block);
  CUDA_CHECK_BOOL(cudaGetLastError());

  // Stage 2: merge splits → final output + LSE.
  // Grid: (num_tokens, H_BLOCKS). Block: HPB warps × 32 lanes = 512 threads.
  auto merge_kernel = sparse_mla_decode_v3_merge_kernel<NUM_HEADS, KV::D_V>;
  dim3 grid2(num_tokens, H_BLOCKS);
  dim3 block2(HPB * 32);
  merge_kernel<<<grid2, block2, 0, stream>>>(mid_out, mid_lse, output, out_lse,
                                             num_tokens, num_splits);
  CUDA_CHECK_BOOL(cudaGetLastError());
  return true;
}

// Public surface — explicit instantiation switch over the PR-body bench grid.
// MODEL1 only, page_block_size=64 only. NUM_HEADS ∈ {16, 32, 64, 128},
// TOPK ∈ {128, 512, 1024}.
bool launch_sparse_mla_decode_v3(ModelType mt, int num_heads, int topk,
                                 int page_block_size, int num_tokens,
                                 int num_splits, const bf16* Q,
                                 const uint8_t* KV_cache, const int32_t* indices,
                                 bf16* mid_out, float* mid_lse, bf16* output,
                                 float* out_lse, const int* topk_length,
                                 float sm_scale, size_t stride_kv_block,
                                 cudaStream_t stream) {
  if (mt != ModelType::MODEL1 || page_block_size != 64) return false;
#define V3_DISPATCH(H, K)                                              \
  if (num_heads == (H) && topk == (K)) {                               \
    return launch_decode_v3_impl<ModelType::MODEL1, (H), (K), 64>(     \
        Q, KV_cache, indices, mid_out, mid_lse, topk_length, output,   \
        out_lse, num_tokens, num_splits, sm_scale, stride_kv_block,    \
        stream);                                                       \
  }
  V3_DISPATCH(16, 128)
  V3_DISPATCH(16, 512)
  V3_DISPATCH(16, 1024)
  V3_DISPATCH(32, 128)
  V3_DISPATCH(32, 512)
  V3_DISPATCH(32, 1024)
  V3_DISPATCH(64, 128)
  V3_DISPATCH(64, 512)
  V3_DISPATCH(64, 1024)
  V3_DISPATCH(128, 128)
  V3_DISPATCH(128, 512)
  V3_DISPATCH(128, 1024)
#undef V3_DISPATCH
  return false;
}

}  // namespace flashinfer::sparse_mla_sm120
