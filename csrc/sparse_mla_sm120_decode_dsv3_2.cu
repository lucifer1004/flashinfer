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

// Sparse-MLA SM120 decode (v2, scheduler-driven). Single raw-pointer entry
// point that dispatches DSV3_2 / DSV4 + num_heads × topk × extra-page-size
// template variants. Drops the v1 (split-KV + manual combine) path.
//
// Raw-pointer interface; framework-agnostic.

#include <cuda_runtime.h>
#include <flashinfer/attention/sparse_mla_sm120/common/sched_params.h>
#include <flashinfer/attention/sparse_mla_sm120/model/model_type.h>

#include <flashinfer/attention/sparse_mla_sm120/arch/common.cuh>
#include <flashinfer/attention/sparse_mla_sm120/common/smem_layout.cuh>
#include <flashinfer/attention/sparse_mla_sm120/decode_dsv3_2_kernel.cuh>
#include <flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh>

namespace flashinfer::sparse_mla_sm120 {

namespace {

template <ModelType MT, ComputeMode CM, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE>
void launch_decode_dsv3_2(const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
                      float* o_accum, float* lse_accum, bf16* output, float* out_lse,
                      const DecodingSchedMeta* sched_meta, const int* num_splits_ptr,
                      float sm_scale, int num_batches, int s_q, int topk, size_t stride_kv_block,
                      int num_sm_parts, size_t stride_oa_split, size_t stride_oa_sq,
                      size_t stride_la_split, size_t stride_la_sq, const float* attn_sink,
                      const int* topk_length, cudaStream_t stream) {
  constexpr size_t smem_bytes = SmemLayout<MT, CM>::TOTAL;
  constexpr int REPLICATE_H = (NUM_HEADS + HPB - 1) / HPB;

  auto kernel =
      sparse_mla_decode_dsv3_2_kernel<MT, CM, NUM_HEADS, TOPK, PAGE_BLOCK_SIZE>;
  static bool configured = false;
  if (!configured && smem_bytes > 48 * 1024) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    configured = true;
  }

  dim3 grid(REPLICATE_H, s_q, num_sm_parts);
  dim3 block(BLOCK_THREADS);

  DecodeDsv3_2ColdParams cold{sm_scale,
                          num_batches,
                          s_q,
                          stride_kv_block,
                          topk,
                          stride_oa_split,
                          stride_oa_sq,
                          stride_la_split,
                          stride_la_sq,
                          attn_sink,
                          topk_length};

  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{grid, block, smem_bytes, stream, attrs, 1};
  void* args[] = {(void*)&Q,
                  (void*)&KV_cache,
                  (void*)&indices,
                  (void*)&o_accum,
                  (void*)&lse_accum,
                  (void*)&output,
                  (void*)&out_lse,
                  (void*)&sched_meta,
                  (void*)&num_splits_ptr,
                  (void*)&cold};
  CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
}

// Return false if no template variant matches (caller raises).
inline bool dispatch_dsv3_2(int num_heads, int topk, const bf16* Q, const uint8_t* KV,
                         const int32_t* indices, float* o_accum, float* lse_accum,
                         bf16* output, float* out_lse, const DecodingSchedMeta* sched_meta,
                         const int* num_splits_ptr, float sm_scale, int num_batches, int s_q,
                         size_t stride_kv_block, int num_sm_parts, size_t stride_oa_split,
                         size_t stride_oa_sq, size_t stride_la_split, size_t stride_la_sq,
                         const float* attn_sink, const int* topk_length,
                         cudaStream_t stream) {
  if (topk != 2048) return false;

#define DISPATCH_DSV3_2(NH)                                                                    \
  launch_decode_dsv3_2<ModelType::DSV3_2, ComputeMode::FP8, NH, 2048, 1>(                      \
      Q, KV, indices, o_accum, lse_accum, output, out_lse, sched_meta, num_splits_ptr,         \
      sm_scale, num_batches, s_q, topk, stride_kv_block, num_sm_parts, stride_oa_split,        \
      stride_oa_sq, stride_la_split, stride_la_sq, attn_sink, topk_length, stream)

  switch (num_heads) {
    case 8:
      DISPATCH_DSV3_2(8);
      return true;
    case 16:
      DISPATCH_DSV3_2(16);
      return true;
    case 32:
      DISPATCH_DSV3_2(32);
      return true;
    case 64:
      DISPATCH_DSV3_2(64);
      return true;
    case 128:
      DISPATCH_DSV3_2(128);
      return true;
    default:
      return false;
  }
#undef DISPATCH_DSV3_2
}

}  // namespace

// Public dispatcher. Returns false if no template variant matches the
// (num_heads, topk) combination, so the orchestrator can produce a precise
// error message at the framework boundary. This kernel is V32-only: DSv4
// decode is serviced by decode-dsv4 (with its own dual-cache support).
bool sparse_mla_decode_dsv3_2_dispatch(ModelType mt, int num_heads, int topk, int page_block_size,
                                   const bf16* Q, const uint8_t* KV_cache, const int32_t* indices,
                                   float* o_accum, float* lse_accum, bf16* output, float* out_lse,
                                   const DecodingSchedMeta* sched_meta, const int* num_splits_ptr,
                                   float sm_scale, int num_batches, int s_q, int stride_kv_row,
                                   int num_sm_parts, const float* attn_sink, const int* topk_length,
                                   cudaStream_t stream) {
  if (mt != ModelType::DSV3_2) return false;
  const size_t stride_kv_block = (size_t)page_block_size * (size_t)stride_kv_row;
  const size_t stride_oa_split = (size_t)s_q * num_heads * D_V;
  const size_t stride_oa_sq = (size_t)num_heads * D_V;
  const size_t stride_la_split = (size_t)s_q * num_heads;
  const size_t stride_la_sq = (size_t)num_heads;
  return dispatch_dsv3_2(num_heads, topk, Q, KV_cache, indices, o_accum, lse_accum, output,
                         out_lse, sched_meta, num_splits_ptr, sm_scale, num_batches, s_q,
                         stride_kv_block, num_sm_parts, stride_oa_split, stride_oa_sq,
                         stride_la_split, stride_la_sq, attn_sink, topk_length, stream);
}

}  // namespace flashinfer::sparse_mla_sm120
