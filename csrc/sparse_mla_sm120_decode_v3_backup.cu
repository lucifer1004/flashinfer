// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cuda_runtime.h>
#include <cstdio>

#include <flashinfer/attention/sparse_mla_sm120/decode_v3_backup_kernel.cuh>
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
static bool launch_decode_v3_backup_impl(const bf16* Q, const uint8_t* KV_cache,
                                         const int32_t* indices, bf16* mid_out,
                                         float* mid_lse, const int* topk_length,
                                         bf16* output, float* out_lse,
                                         int num_tokens, int num_splits,
                                         float sm_scale, size_t stride_kv_block,
                                         cudaStream_t stream) {
  using KV = KVCacheTraits<MT>;
  constexpr int H_BLOCKS = NUM_HEADS / HPB;

  // Smem layout matches v3 (sub-namespace's constants are identical).
  constexpr int DYN_SMEM_BYTES =
      HPB * KV::D_QK * (int)sizeof(bf16) +
      v3_backup::V3_BI * KV::D_QK * (int)sizeof(bf16) +
      2 * v3_backup::V3_N_WARPS * HPB * (int)sizeof(float) +
      KV::D_V * (int)sizeof(float);

  auto kernel =
      v3_backup::sparse_mla_decode_v3_kernel<MT, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE>;
  CUDA_CHECK_BOOL(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM_BYTES));

  dim3 grid1(num_tokens, H_BLOCKS, num_splits);
  dim3 block1(v3_backup::V3_BLOCK_THREADS);
  kernel<<<grid1, block1, DYN_SMEM_BYTES, stream>>>(Q, KV_cache, indices, mid_out,
                                                   mid_lse, topk_length, num_tokens,
                                                   num_splits, sm_scale, stride_kv_block);
  CUDA_CHECK_BOOL(cudaGetLastError());

  auto merge_kernel = v3_backup::sparse_mla_decode_v3_merge_kernel<NUM_HEADS, KV::D_V>;
  dim3 grid2(num_tokens, H_BLOCKS);
  dim3 block2(HPB * 32);
  merge_kernel<<<grid2, block2, 0, stream>>>(mid_out, mid_lse, output, out_lse,
                                             num_tokens, num_splits);
  CUDA_CHECK_BOOL(cudaGetLastError());
  return true;
}

// Public surface — same dispatch grid as v3 so the two are drop-in interchangeable.
bool launch_sparse_mla_decode_v3_backup(ModelType mt, int num_heads, int topk,
                                        int page_block_size, int num_tokens,
                                        int num_splits, const bf16* Q,
                                        const uint8_t* KV_cache, const int32_t* indices,
                                        bf16* mid_out, float* mid_lse, bf16* output,
                                        float* out_lse, const int* topk_length,
                                        float sm_scale, size_t stride_kv_block,
                                        cudaStream_t stream) {
  if (mt != ModelType::MODEL1 || page_block_size != 64) return false;
#define V3B_DISPATCH(H, K)                                                  \
  if (num_heads == (H) && topk == (K)) {                                    \
    return launch_decode_v3_backup_impl<ModelType::MODEL1, (H), (K), 64>(   \
        Q, KV_cache, indices, mid_out, mid_lse, topk_length, output,        \
        out_lse, num_tokens, num_splits, sm_scale, stride_kv_block,         \
        stream);                                                            \
  }
  V3B_DISPATCH(16, 128)
  V3B_DISPATCH(16, 512)
  V3B_DISPATCH(16, 1024)
  V3B_DISPATCH(32, 128)
  V3B_DISPATCH(32, 512)
  V3B_DISPATCH(32, 1024)
  V3B_DISPATCH(64, 128)
  V3B_DISPATCH(64, 512)
  V3B_DISPATCH(64, 1024)
  V3B_DISPATCH(128, 128)
  V3B_DISPATCH(128, 512)
  V3B_DISPATCH(128, 1024)
#undef V3B_DISPATCH
  return false;
}

}  // namespace flashinfer::sparse_mla_sm120
