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

  // Stage 1: decode-v3 (A1.2) partial-output kernel.
  // Dynamic smem layout (FP8 XV, MODEL1, V3_BI=64, double-buffered KV):
  //   sm_q_rope    HPB * D_ROPE * 2B                      =  2 KB
  //   sm_q_fp8     HPB * Q_NOPE_STRIDE                    = 7.25 KB
  //   sm_q_sc      HPB * NUM_SCALES * 4B                  = 0.44 KB
  //   sm_kv_fp8    2 * V3_BI * KV_SMEM_STRIDE             = 58 KB
  //   sm_kv_sc     2 * V3_BI * SCALE_BYTES_PER_TOKEN      =  1 KB
  //   sm_kv_rope   2 * V3_BI * D_ROPE * 2B                = 16 KB
  //   sm_reduce    2 * V3_N_WARPS * HPB * 4               = 1 KB
  //   sm_w_head_sc N_V_CHUNKS * HPB * 4                   = 448 B
  //   sm_w_fp8 ×2  2 * HPB * (V3_BI + 16)                 = 2.5 KB
  //   Total                                               ~ 88 KB
  // Static smem (kernel-side):
  //   sm_p_full    HPB * V3_BI * 2B (bf16)                =  2 KB
  // Grand total ~ 89 KB (under 100 KB SM120 carveout, 1 block/SM).
  constexpr int N_V_CHUNKS_LAUNCH = KV::D_NOPE / KV::QUANT_TILE;        // 7
  constexpr int DYN_SMEM_BYTES =
      HPB * KV::D_ROPE * (int)sizeof(bf16)                              // sm_q_rope
      + HPB * KV::Q_NOPE_STRIDE                                          // sm_q_fp8
      + HPB * KV::NUM_SCALES * (int)sizeof(float)                        // sm_q_sc
      + V3_KV_BUF_COUNT * V3_BI * KV::KV_SMEM_STRIDE                     // sm_kv_fp8 ×2
      + V3_KV_BUF_COUNT * V3_BI * KV::SCALE_BYTES_PER_TOKEN              // sm_kv_sc ×2
      + V3_KV_BUF_COUNT * V3_BI * KV::D_ROPE * (int)sizeof(bf16)         // sm_kv_rope ×2
      + 16                                                               // mbar align pad
      + 4 * (int)sizeof(uint64_t)                                        // mbar_full+empty
      + 2 * V3_N_WARPS * HPB * (int)sizeof(float)                        // sm_reduce
      + N_V_CHUNKS_LAUNCH * HPB * (int)sizeof(float)                     // sm_w_head_sc
      + 2 * HPB * (V3_BI + 16);                                          // sm_w_fp8 ×2 (vc parity)

  auto kernel = sparse_mla_decode_v3_kernel<MT, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE>;
  CUDA_CHECK_BOOL(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM_BYTES));

  // Heuristic for chunks_per_block: target ~2 waves of GPU blocks so per-
  // block work is amortized across the GPU without leaving SMs idle.
  // chunks_per_block * num_splits_eff = num_chunks_total (= num_splits
  // from caller; kept as the mid_out stride).
  int sm_count = 0;
  CUDA_CHECK_BOOL(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
  const int target_blocks = 2 * (sm_count > 0 ? sm_count : 188);
  const int per_token_head = num_tokens * H_BLOCKS;
  // chunks_per_block = ceil(num_chunks_total * per_token_head / target_blocks)
  // clamped to [1, num_chunks_total]. For grids that already saturate the
  // GPU at cpb=1 (per_token_head >= target_blocks), this is 1 — no change
  // from the prior single-chunk-per-block design.
  int cpb_raw =
      (num_splits * per_token_head + target_blocks - 1) / target_blocks;
  int chunks_per_block = (cpb_raw < 1) ? 1 : (cpb_raw > num_splits ? num_splits : cpb_raw);
  int num_splits_eff = (num_splits + chunks_per_block - 1) / chunks_per_block;

  // Launch the FULL Python-allocated num_splits grid blocks; inactive splits
  // (chunk_lo >= num_chunks_total) return early after marking LSE = -inf,
  // which is cheap. This keeps the mid_out/mid_lse stride matching Python's
  // allocation without extra coordination.
  (void)num_splits_eff;
  dim3 grid1(num_tokens, H_BLOCKS, num_splits);
  dim3 block1(V3_BLOCK_THREADS);
  kernel<<<grid1, block1, DYN_SMEM_BYTES, stream>>>(Q, KV_cache, indices, mid_out,
                                                   mid_lse, topk_length, num_tokens,
                                                   num_splits, chunks_per_block, sm_scale,
                                                   stride_kv_block);
  CUDA_CHECK_BOOL(cudaGetLastError());

  // Stage 2: merge splits → final output + LSE.
  // Grid: (num_tokens, NUM_HEADS). One block (BLOCK_THREADS=64) covers the
  // full D_V=512 via uint4 vec loads (8 bf16/thread × 64 threads = 512).
  // For h=128/T=16 this is 2048 blocks vs the prior 8192 (4× fewer).
  constexpr int MERGE_BLOCK_THREADS = 64;
  constexpr int MERGE_DIMS_PER_THREAD = KV::D_V / MERGE_BLOCK_THREADS;
  auto merge_kernel = sparse_mla_decode_v3_merge_kernel<
      NUM_HEADS, KV::D_V, MERGE_BLOCK_THREADS, MERGE_DIMS_PER_THREAD>;
  dim3 grid2(num_tokens, NUM_HEADS);
  dim3 block2(MERGE_BLOCK_THREADS);
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
