// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cuda_runtime.h>

#include <flashinfer/attention/sparse_mla_sm120/decode_v3_kernel.cuh>
#include <flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh>

namespace flashinfer::sparse_mla_sm120 {

#define CUDA_CHECK(call)                                          \
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
                                  int num_tokens, int num_splits,
                                  float sm_scale, size_t stride_kv_block,
                                  cudaStream_t stream) {
  using KV = KVCacheTraits<MT>;
  constexpr int H_BLOCKS = NUM_HEADS / HPB;
  dim3 grid(num_tokens, H_BLOCKS, num_splits);
  dim3 block(V3_BLOCK_THREADS);

  constexpr int SMEM_BYTES =
      HPB * KV::D_QK * sizeof(bf16) + V3_BI * KV::D_QK * sizeof(bf16) +
      2 * V3_N_WARPS * HPB * sizeof(float) + HPB * KV::D_V * sizeof(float) +
      V3_N_WARPS * HPB * V3_ENTRIES_PER_WARP * sizeof(float) + 256 /* slack */;

  auto kernel = sparse_mla_decode_v3_kernel<MT, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE>;
  CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));

  kernel<<<grid, block, SMEM_BYTES, stream>>>(Q, KV_cache, indices, mid_out, mid_lse,
                                              topk_length, num_tokens, num_splits,
                                              sm_scale, stride_kv_block);
  CUDA_CHECK(cudaGetLastError());

  // Merge kernel: collapse splits.
  // Per token: 1 block with HPB warps (one per head) × 32 lanes.
  // For NUM_HEADS > HPB we tile h_block_idx similarly.
  return true;
}

// Public surface — explicit instantiation switch.
bool launch_sparse_mla_decode_v3(ModelType mt, int num_heads, int topk,
                                 int page_block_size, int num_tokens,
                                 int num_splits, const bf16* Q,
                                 const uint8_t* KV_cache, const int32_t* indices,
                                 bf16* mid_out, float* mid_lse,
                                 const int* topk_length, float sm_scale,
                                 size_t stride_kv_block, cudaStream_t stream) {
  // Minimal instantiation surface — MODEL1, h=128, topk=512, pbs=64 only.
  if (mt == ModelType::MODEL1 && num_heads == 128 && topk == 512 &&
      page_block_size == 64) {
    return launch_decode_v3_impl<ModelType::MODEL1, 128, 512, 64>(
        Q, KV_cache, indices, mid_out, mid_lse, topk_length, num_tokens, num_splits,
        sm_scale, stride_kv_block, stream);
  }
  return false;
}

}  // namespace flashinfer::sparse_mla_sm120
