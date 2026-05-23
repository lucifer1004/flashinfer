// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Sparse-MLA SM120 prefill. Single raw-pointer entry point that dispatches:
//   - DSV3_2 / DSV4 model split
//   - SG (single-group, 16 heads/CTA) for num_heads <= 16
//   - MG (multi-group, 32 heads/CTA) for num_heads > 16
//   - Dual-cache MG variants (DSV4 only)
//
// Raw-pointer interface; framework-agnostic.

#include <cuda_runtime.h>
#include <flashinfer/attention/sparse_mla_sm120/model/model_type.h>

#include <flashinfer/attention/sparse_mla_sm120/arch/common.cuh>
#include <flashinfer/attention/sparse_mla_sm120/common/smem_layout.cuh>
#include <flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh>
#include <flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh>

namespace flashinfer::sparse_mla_sm120 {

namespace {

template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE>
void launch_prefill_sg(const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
                       const float* attn_sink, bf16* output, float* out_lse, float sm_scale,
                       int num_tokens, size_t stride_kv_block, const int* topk_length_ptr,
                       cudaStream_t stream) {
  constexpr size_t smem_bytes = SmemLayout<MT, CM>::TOTAL;
  // Ceil-div so NUM_HEADS < HPB (small-TP shards) still launches 1 CTA per token.
  constexpr int REPLICATE_H = (NUM_HEADS + HPB - 1) / HPB;
  dim3 grid(num_tokens * REPLICATE_H);
  dim3 block(BLOCK_THREADS);

  auto kernel = sparse_mla_prefill_kernel<MT, CM, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE>;
  static bool configured = false;
  if (!configured && smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    configured = true;
  }

  // SG is single-cache only.
  PrefillColdParams cold{sm_scale,
                         num_tokens,
                         stride_kv_block,
                         /*stride_kv_block_extra=*/(size_t)0,
                         attn_sink,
                         topk_length_ptr,
                         /*topk_length_extra=*/(const int*)nullptr};
  cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
  void* args[] = {(void*)&Q,      (void*)&KV_cache, (void*)&indices, (void*)&attn_sink,
                  (void*)&output, (void*)&out_lse,  (void*)&cold};
  CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
}

// MG dispatcher. TOPK_EXTRA == 0 selects single-cache; > 0 selects dual.
// MG_N_HG_T: 1 lets NUM_HEADS=16 go through MG (covers swa+dual layers
// where SG has no dual-cache support); 2 is the default for NH >= 32.
template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE,
          int TOPK_EXTRA = 0, int PAGE_BLOCK_SIZE_EXTRA = PAGE_BLOCK_SIZE,
          int MG_N_HG_T = MG_N_HG_DEFAULT>
void launch_prefill_mg(const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
                       const uint8_t* KV_cache_extra, const int32_t* indices_extra,
                       const float* attn_sink, bf16* output, float* out_lse, float sm_scale,
                       int num_tokens, size_t stride_kv_block, size_t stride_kv_block_extra,
                       const int* topk_length_ptr, const int* topk_length_extra_ptr,
                       cudaStream_t stream) {
  constexpr size_t smem_bytes = SmemLayoutMG<MT, CM>::TOTAL;
  constexpr int MG_HEADS_PER_CTA_LOCAL = MG_N_HG_T * HPB;
  static_assert(NUM_HEADS % MG_HEADS_PER_CTA_LOCAL == 0,
                "NUM_HEADS must be a multiple of MG_N_HG_T * HPB");
  constexpr int REPLICATE_H = NUM_HEADS / MG_HEADS_PER_CTA_LOCAL;
  dim3 grid(num_tokens * REPLICATE_H);
  dim3 block(BLOCK_THREADS);

  auto kernel = sparse_mla_prefill_mg_kernel<MT, CM, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE, TOPK_EXTRA,
                                             PAGE_BLOCK_SIZE_EXTRA, MG_N_HG_T>;
  static bool configured = false;
  if (!configured && smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    configured = true;
  }

  PrefillColdParams cold{sm_scale,  num_tokens,      stride_kv_block,      stride_kv_block_extra,
                         attn_sink, topk_length_ptr, topk_length_extra_ptr};
  cudaLaunchConfig_t config{grid, block, smem_bytes, stream, nullptr, 0};
  void* args[] = {(void*)&Q,
                  (void*)&KV_cache,
                  (void*)&indices,
                  (void*)&KV_cache_extra,
                  (void*)&indices_extra,
                  (void*)&output,
                  (void*)&out_lse,
                  (void*)&attn_sink,
                  (void*)&cold};
  CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
}

inline bool dispatch_v32(int num_heads, int topk, const bf16* Q, const uint8_t* KV,
                         const int32_t* indices, const float* attn_sink, bf16* output,
                         float* out_lse, float sm_scale, int num_tokens, size_t stride_kv_block,
                         const int* topk_length_ptr, cudaStream_t stream) {
  if (topk != 2048) return false;

  // PBS=64 matches the V32 decode (`decode_dsv3_2_kernel.cuh`). NH=8 covers
  // small-TP shards; the SG kernel zero-pads invalid head slots up to HPB=16
  // internally and gates write-back by VALID_HPB.
  if (num_heads <= HPB) {
    if (num_heads == 8) {
      launch_prefill_sg<ModelType::DSV3_2, ComputeMode::FP8, 8, 2048, 64>(
          Q, KV, indices, attn_sink, output, out_lse, sm_scale, num_tokens, stride_kv_block,
          topk_length_ptr, stream);
      return true;
    }
    if (num_heads != 16) return false;
    launch_prefill_sg<ModelType::DSV3_2, ComputeMode::FP8, 16, 2048, 64>(
        Q, KV, indices, attn_sink, output, out_lse, sm_scale, num_tokens, stride_kv_block,
        topk_length_ptr, stream);
    return true;
  }

#define DISPATCH_DSV3_2_MG(NH)                                                                     \
  launch_prefill_mg<ModelType::DSV3_2, ComputeMode::FP8, NH, 2048, 64>(                            \
      Q, KV, indices, /*KV_extra=*/nullptr, /*idx_extra=*/nullptr, attn_sink, output, out_lse,     \
      sm_scale, num_tokens, stride_kv_block, /*stride_kv_block_extra=*/(size_t)0, topk_length_ptr, \
      /*topk_length_extra=*/nullptr, stream)

  switch (num_heads) {
    case 32:
      DISPATCH_DSV3_2_MG(32);
      return true;
    case 64:
      DISPATCH_DSV3_2_MG(64);
      return true;
    case 128:
      DISPATCH_DSV3_2_MG(128);
      return true;
    default:
      return false;
  }
#undef DISPATCH_DSV3_2_MG
}

inline bool dispatch_dsv4_single(int num_heads, int topk, const bf16* Q, const uint8_t* KV,
                                 const int32_t* indices, const float* attn_sink, bf16* output,
                                 float* out_lse, float sm_scale, int num_tokens,
                                 size_t stride_kv_block, const int* topk_length_ptr,
                                 cudaStream_t stream) {
#define DISPATCH_SG_CM(CM, NH, TK)                                                       \
  launch_prefill_sg<ModelType::DSV4, ComputeMode::CM, NH, TK, 64>(                       \
      Q, KV, indices, attn_sink, output, out_lse, sm_scale, num_tokens, stride_kv_block, \
      topk_length_ptr, stream)

#define DISPATCH_MG_CM(CM, NH, TK)                                                                 \
  launch_prefill_mg<ModelType::DSV4, ComputeMode::CM, NH, TK, 64>(                                 \
      Q, KV, indices, /*KV_extra=*/nullptr, /*idx_extra=*/nullptr, attn_sink, output, out_lse,     \
      sm_scale, num_tokens, stride_kv_block, /*stride_kv_block_extra=*/(size_t)0, topk_length_ptr, \
      /*topk_length_extra=*/nullptr, stream)

#define DISPATCH_BY_NH_CM(CM, TK)    \
  do {                               \
    switch (num_heads) {             \
      case 16:                       \
        DISPATCH_SG_CM(CM, 16, TK);  \
        return true;                 \
      case 32:                       \
        DISPATCH_MG_CM(CM, 32, TK);  \
        return true;                 \
      case 64:                       \
        DISPATCH_MG_CM(CM, 64, TK);  \
        return true;                 \
      case 128:                      \
        DISPATCH_MG_CM(CM, 128, TK); \
        return true;                 \
      default:                       \
        return false;                \
    }                                \
  } while (0)

  // Small K-loop: BF16 QK skips the FP8 Q-quantize prologue. Larger K
  // amortises FP8's higher Tensor-Core throughput.
  if (topk == 128)
    DISPATCH_BY_NH_CM(BF16, 128);
  else if (topk == 512)
    DISPATCH_BY_NH_CM(FP8, 512);
  else if (topk == 1024)
    DISPATCH_BY_NH_CM(FP8, 1024);
  else if (topk == 2048)
    DISPATCH_BY_NH_CM(FP8, 2048);
  else
    return false;

#undef DISPATCH_BY_NH_CM
#undef DISPATCH_MG_CM
#undef DISPATCH_SG_CM
  return false;  // unreachable
}

inline bool dispatch_dsv4_dual(int num_heads, int topk, int topk_extra, int extra_page_block_size,
                               const bf16* Q, const uint8_t* KV, const int32_t* indices,
                               const uint8_t* KV_extra, const int32_t* idx_extra,
                               const float* attn_sink, bf16* output, float* out_lse, float sm_scale,
                               int num_tokens, size_t stride_kv_block, size_t stride_kv_block_extra,
                               const int* topk_length_ptr, const int* topk_length_extra_ptr,
                               cudaStream_t stream) {
// NH=16 dispatches through MG with MG_N_HG_T=1 so callers can pad TP=4/TP=8
// without falling into SG (SG has no dual-cache support).
#define DISPATCH_DUAL_MG_CM(CM, NH, TK, TK_EX, PBSX, NHG)                                    \
  launch_prefill_mg<ModelType::DSV4, ComputeMode::CM, NH, TK, 64, TK_EX, PBSX, NHG>(         \
      Q, KV, indices, KV_extra, idx_extra, attn_sink, output, out_lse, sm_scale, num_tokens, \
      stride_kv_block, stride_kv_block_extra, topk_length_ptr, topk_length_extra_ptr, stream)

  if (topk == 128 && topk_extra == 128 && extra_page_block_size == 64) {
    switch (num_heads) {
      case 16:
        DISPATCH_DUAL_MG_CM(BF16, 16, 128, 128, 64, 1);
        return true;
      case 32:
        DISPATCH_DUAL_MG_CM(BF16, 32, 128, 128, 64, 2);
        return true;
      case 64:
        DISPATCH_DUAL_MG_CM(BF16, 64, 128, 128, 64, 2);
        return true;
      case 128:
        DISPATCH_DUAL_MG_CM(BF16, 128, 128, 128, 64, 2);
        return true;
      default:
        return false;
    }
  } else if (topk == 128 && topk_extra == 512 && extra_page_block_size == 64) {
    // C4A: SWA window=128, indexer top_k=512, compress_ratio=4.
    switch (num_heads) {
      case 16:
        DISPATCH_DUAL_MG_CM(BF16, 16, 128, 512, 64, 1);
        return true;
      case 32:
        DISPATCH_DUAL_MG_CM(BF16, 32, 128, 512, 64, 2);
        return true;
      case 64:
        DISPATCH_DUAL_MG_CM(BF16, 64, 128, 512, 64, 2);
        return true;
      case 128:
        DISPATCH_DUAL_MG_CM(BF16, 128, 128, 512, 64, 2);
        return true;
      default:
        return false;
    }
  } else if (topk == 128 && topk_extra == 512 && extra_page_block_size == 2) {
    // C128A: SWA window=128, indexer top_k=512, compress_ratio=128.
    switch (num_heads) {
      case 16:
        DISPATCH_DUAL_MG_CM(BF16, 16, 128, 512, 2, 1);
        return true;
      case 32:
        DISPATCH_DUAL_MG_CM(BF16, 32, 128, 512, 2, 2);
        return true;
      case 64:
        DISPATCH_DUAL_MG_CM(BF16, 64, 128, 512, 2, 2);
        return true;
      case 128:
        DISPATCH_DUAL_MG_CM(BF16, 128, 128, 512, 2, 2);
        return true;
      default:
        return false;
    }
  }
  return false;
#undef DISPATCH_DUAL_MG_CM
}

}  // namespace

// Public dispatcher. Returns false if no template variant matches
// (model_type, num_heads, topk, [extra_*]). Caller raises with the
// supported envelope.
//
// Dual-cache mode is triggered iff extra_KV_cache != nullptr.
// Dual-cache is DSV4-only; passing extra_KV_cache != nullptr with mt=DSV3_2
// returns false.
bool sparse_mla_prefill_dispatch(ModelType mt, int num_heads, int topk, int page_block_size,
                                 int topk_extra, int extra_page_block_size, const bf16* Q,
                                 const uint8_t* KV_cache, const int32_t* indices,
                                 const uint8_t* extra_KV_cache, const int32_t* extra_indices,
                                 bf16* output, float* out_lse, float sm_scale, int num_tokens,
                                 int stride_kv_row, int extra_stride_kv_row, const float* attn_sink,
                                 const int* topk_length, const int* extra_topk_length,
                                 cudaStream_t stream) {
  const size_t stride_kv_block = (size_t)page_block_size * (size_t)stride_kv_row;
  const size_t stride_kv_block_extra =
      (extra_KV_cache != nullptr) ? (size_t)extra_page_block_size * (size_t)extra_stride_kv_row : 0;

  if (extra_KV_cache != nullptr) {
    if (mt != ModelType::DSV4) return false;
    return dispatch_dsv4_dual(num_heads, topk, topk_extra, extra_page_block_size, Q, KV_cache,
                              indices, extra_KV_cache, extra_indices, attn_sink, output, out_lse,
                              sm_scale, num_tokens, stride_kv_block, stride_kv_block_extra,
                              topk_length, extra_topk_length, stream);
  }

  switch (mt) {
    case ModelType::DSV3_2:
      return dispatch_v32(num_heads, topk, Q, KV_cache, indices, attn_sink, output, out_lse,
                          sm_scale, num_tokens, stride_kv_block, topk_length, stream);
    case ModelType::DSV4:
      return dispatch_dsv4_single(num_heads, topk, Q, KV_cache, indices, attn_sink, output, out_lse,
                                  sm_scale, num_tokens, stride_kv_block, topk_length, stream);
  }
  return false;
}

}  // namespace flashinfer::sparse_mla_sm120
