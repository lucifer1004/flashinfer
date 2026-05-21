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

// Sparse-MLA SM120 paged attention orchestrator.
//
// Single TVM-FFI entry point that:
//   - Validates inputs (dtype / shape / contiguity)
//   - Detects model type from d_qk (V32: 576, MODEL1: 512)
//   - Dispatches decode-v2 (num_tokens <= 64) vs prefill (larger)
//   - For decode: launches sched + decode + combine
//   - For prefill: launches prefill (single or dual cache)
//
// Workspace partitioning is handled by the Python wrapper; this orchestrator
// takes pre-partitioned tensors directly to match the cutlass_mla precedent.

#include <cuda_runtime.h>
#include <flashinfer/attention/sparse_mla_sm120/common/sched_params.h>
#include <flashinfer/attention/sparse_mla_sm120/model/model_type.h>

#include <flashinfer/attention/sparse_mla_sm120/arch/common.cuh>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace flashinfer::sparse_mla_sm120 {

// Forward declarations (defined in sibling .cu files).
void launch_get_sched_meta(int b, int topk, int extra_topk, int block_size_n, int fixed_overhead,
                           int num_sm_parts, const int* topk_length, const int* extra_topk_length,
                           DecodingSchedMeta* sched_meta, int* num_splits, cudaStream_t stream);

bool sparse_mla_decode_v2_dispatch(
    ModelType mt, int num_heads, int topk, int page_block_size, int extra_page_block_size,
    const bf16* Q, const uint8_t* KV_cache, const int32_t* indices, const uint8_t* extra_KV_cache,
    const int32_t* extra_indices, float* o_accum, float* lse_accum, bf16* output, float* out_lse,
    const DecodingSchedMeta* sched_meta, const int* num_splits_ptr, float sm_scale, int num_batches,
    int s_q, int stride_kv_row, int num_sm_parts, const float* attn_sink, const int* topk_length,
    int extra_topk, const int* extra_topk_length, int extra_stride_kv_row, cudaStream_t stream);

bool sparse_mla_prefill_dispatch(ModelType mt, int num_heads, int topk, int page_block_size,
                                 int topk_extra, int extra_page_block_size, const bf16* Q,
                                 const uint8_t* KV_cache, const int32_t* indices,
                                 const uint8_t* extra_KV_cache, const int32_t* extra_indices,
                                 bf16* output, float* out_lse, float sm_scale, int num_tokens,
                                 int stride_kv_row, int extra_stride_kv_row, const float* attn_sink,
                                 const int* topk_length, const int* extra_topk_length,
                                 cudaStream_t stream);

bool launch_combine_v2(const float* o_accum, const float* lse_accum, bf16* output, float* out_lse,
                       const int* num_splits_ptr, int batch, int s_q, int num_heads,
                       int max_nsplits, const float* attn_sink, cudaStream_t stream);

namespace {

inline ModelType infer_model_type(int d_qk) {
  if (d_qk == 576) return ModelType::V32;
  TVM_FFI_ICHECK_EQ(d_qk, 512) << "Unsupported d_qk=" << d_qk
                               << "; expected 576 (V32) or 512 (MODEL1)";
  return ModelType::MODEL1;
}

// Convert padded-block KV strides to a per-row override matching the kernel's
// `page_block_size * stride_kv_row` block stride. When vLLM pads block stride
// for alignment, the natural per-token stride times page_block_size doesn't
// equal the actual block-to-block stride; encode the padding via this row
// override. Mirrors the upstream `effective_stride_kv_row` helper.
inline int effective_stride_kv_row(const TensorView& kv) {
  const int natural_row_bytes = static_cast<int>(kv.stride(-2) * (kv.dtype().bits / 8));
  const int block_stride_bytes = static_cast<int>(kv.stride(0) * (kv.dtype().bits / 8));
  const int page_block_size = static_cast<int>(kv.size(-3));
  if (block_stride_bytes == page_block_size * natural_row_bytes) {
    return natural_row_bytes;
  }
  TVM_FFI_ICHECK_EQ(block_stride_bytes % page_block_size, 0)
      << "kv_cache block stride " << block_stride_bytes << " not divisible by page_block_size "
      << page_block_size << "; cannot encode padding via stride_kv_row override";
  return block_stride_bytes / page_block_size;
}

}  // namespace

void SparseMlaSm120PagedAttention(
    TensorView q,           // [num_tokens, num_heads, d_qk] bf16
    TensorView kv_cache,    // [num_pages, page_block_size, ...] paged FP8
    TensorView indices,     // [num_tokens, topk] int32 (-1 = skip)
    TensorView output,      // [num_tokens, num_heads, d_v] bf16 — in-place
    TensorView out_lse,     // [num_tokens, num_heads] f32 — in-place
    TensorView o_accum,     // [num_sm_parts, s_q, num_heads, d_v] f32
    TensorView lse_accum,   // [num_sm_parts, s_q, num_heads] f32
    TensorView sched_meta,  // [num_sm_parts] DecodingSchedMeta — bytes-typed
    TensorView num_splits,  // [batch + 1] int32
    double sm_scale, int64_t num_sm_parts,
    Optional<TensorView> topk_length,        // [num_tokens] int32, optional
    Optional<TensorView> attn_sink,          // [num_heads] f32, optional
    Optional<TensorView> extra_kv_cache,     // optional dual cache
    Optional<TensorView> extra_indices,      // optional dual cache indices
    Optional<TensorView> extra_topk_length)  // [num_tokens] int32, optional
{
  // ── Input validation ───────────────────────────────────────────────
  CHECK_INPUT_AND_TYPE(q, dl_bfloat16);
  // kv_cache: CUDA + last-dim contiguous only. The block stride may be
  // padded for alignment (vLLM convention); the kernel handles that via
  // `effective_stride_kv_row` below.
  CHECK_CUDA(kv_cache);
  CHECK_LAST_DIM_CONTIGUOUS(kv_cache);
  CHECK_INPUT_AND_TYPE(indices, dl_int32);
  CHECK_INPUT_AND_TYPE(output, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(out_lse, dl_float32);

  CHECK_DIM(3, q);

  const int num_tokens = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int d_qk = static_cast<int>(q.size(2));
  const int topk = static_cast<int>(indices.size(-1));
  const int page_block_size = static_cast<int>(kv_cache.size(-3));

  TVM_FFI_ICHECK_GT(num_heads, 0);
  TVM_FFI_ICHECK_LE(num_heads, 128);
  TVM_FFI_ICHECK_GT(page_block_size, 0);

  const ModelType mt = infer_model_type(d_qk);
  const int stride_kv_row = effective_stride_kv_row(kv_cache);

  // attn_sink (per-head bias added pre-softmax).
  const float* attn_sink_ptr = nullptr;
  if (attn_sink.has_value()) {
    const auto& s = attn_sink.value();
    CHECK_INPUT_AND_TYPE(s, dl_float32);
    TVM_FFI_ICHECK_EQ(s.ndim(), 1);
    TVM_FFI_ICHECK_EQ(s.size(0), num_heads) << "attn_sink must be [num_heads]";
    attn_sink_ptr = static_cast<const float*>(s.data_ptr());
  }

  // Optional dual-cache extras.
  const uint8_t* extra_kv_ptr = nullptr;
  const int32_t* extra_idx_ptr = nullptr;
  int extra_page_block_size = 0;
  int extra_stride_kv_row = 0;
  int extra_topk = 0;
  if (extra_kv_cache.has_value()) {
    TVM_FFI_ICHECK(extra_indices.has_value()) << "extra_kv_cache requires extra_indices";
    const auto& ekv = extra_kv_cache.value();
    const auto& eidx = extra_indices.value();
    // Same relaxation as the main kv_cache: padded block stride is OK.
    CHECK_CUDA(ekv);
    CHECK_LAST_DIM_CONTIGUOUS(ekv);
    CHECK_INPUT_AND_TYPE(eidx, dl_int32);
    extra_kv_ptr = static_cast<const uint8_t*>(ekv.data_ptr());
    extra_idx_ptr = static_cast<const int32_t*>(eidx.data_ptr());
    extra_page_block_size = static_cast<int>(ekv.size(-3));
    extra_stride_kv_row = effective_stride_kv_row(ekv);
    extra_topk = static_cast<int>(eidx.size(-1));
  }

  const int* tl_ptr =
      topk_length.has_value() ? static_cast<const int*>(topk_length.value().data_ptr()) : nullptr;
  const int* etl_ptr = extra_topk_length.has_value()
                           ? static_cast<const int*>(extra_topk_length.value().data_ptr())
                           : nullptr;

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());

  const auto Q_ptr = static_cast<const bf16*>(q.data_ptr());
  const auto KV_ptr = static_cast<const uint8_t*>(kv_cache.data_ptr());
  const auto idx_ptr = static_cast<const int32_t*>(indices.data_ptr());
  const auto O_ptr = static_cast<bf16*>(output.data_ptr());
  const auto LSE_ptr = static_cast<float*>(out_lse.data_ptr());

  // ── Prefill path (num_tokens > 64) ────────────────────────────────
  // Prefill writes output and out_lse directly; o_accum/lse_accum/sched_meta/
  // num_splits are unused but still passed in (Python wrapper allocates a
  // shared workspace regardless).
  if (num_tokens > 64) {
    const bool ok = sparse_mla_prefill_dispatch(
        mt, num_heads, topk, page_block_size, extra_topk, extra_page_block_size, Q_ptr, KV_ptr,
        idx_ptr, extra_kv_ptr, extra_idx_ptr, O_ptr, LSE_ptr, static_cast<float>(sm_scale),
        num_tokens, stride_kv_row, extra_stride_kv_row, attn_sink_ptr, tl_ptr, etl_ptr, stream);
    TVM_FFI_ICHECK(ok) << "Unsupported sparse-MLA prefill configuration: "
                       << "model=" << (mt == ModelType::V32 ? "V32" : "MODEL1")
                       << " num_heads=" << num_heads << " topk=" << topk
                       << " page_block_size=" << page_block_size << " topk_extra=" << extra_topk
                       << " extra_page_block_size=" << extra_page_block_size;
    return;
  }

  // ── Decode-v2 path (num_tokens <= 64) ──────────────────────────────
  const int s_q = 1;
  const int num_batches = num_tokens;  // s_q == 1 ⇒ tokens == batches
  constexpr int BI = 64;
  constexpr int FIXED_OVERHEAD = 5;

  CHECK_INPUT_AND_TYPE(o_accum, dl_float32);
  CHECK_INPUT_AND_TYPE(lse_accum, dl_float32);
  CHECK_INPUT(sched_meta);
  CHECK_INPUT_AND_TYPE(num_splits, dl_int32);

  // 1) Scheduler: emit sched_meta + num_splits prefix sum.
  auto* meta_ptr = reinterpret_cast<DecodingSchedMeta*>(sched_meta.data_ptr());
  auto* ns_ptr = static_cast<int*>(num_splits.data_ptr());
  launch_get_sched_meta(num_batches, topk, extra_topk, BI, FIXED_OVERHEAD,
                        static_cast<int>(num_sm_parts), tl_ptr, etl_ptr, meta_ptr, ns_ptr, stream);

  // 2) Decode-v2 dispatch.
  auto* oa_ptr = static_cast<float*>(o_accum.data_ptr());
  auto* la_ptr = static_cast<float*>(lse_accum.data_ptr());
  const bool ok = sparse_mla_decode_v2_dispatch(
      mt, num_heads, topk, page_block_size, extra_page_block_size, Q_ptr, KV_ptr, idx_ptr,
      extra_kv_ptr, extra_idx_ptr, oa_ptr, la_ptr, O_ptr, LSE_ptr, meta_ptr, ns_ptr,
      static_cast<float>(sm_scale), num_batches, s_q, stride_kv_row, static_cast<int>(num_sm_parts),
      attn_sink_ptr, tl_ptr, extra_topk, etl_ptr, extra_stride_kv_row, stream);
  TVM_FFI_ICHECK(ok) << "Unsupported sparse-MLA decode configuration: "
                     << "model=" << (mt == ModelType::V32 ? "V32" : "MODEL1")
                     << " num_heads=" << num_heads << " topk=" << topk
                     << " page_block_size=" << page_block_size << " extra_topk=" << extra_topk
                     << " extra_page_block_size=" << extra_page_block_size;

  // 3) Combine: merge per-split partials when num_sm_parts > 1.
  // Skip the combine launch entirely for batch=1, num_sm_parts=1 (no splits
  // to merge); the decode kernel writes the final output directly when
  // num_splits == 1, identical to upstream's v1-bypass behavior.
  if (num_sm_parts > 1) {
    const bool combine_ok =
        launch_combine_v2(oa_ptr, la_ptr, O_ptr, LSE_ptr, ns_ptr, num_batches, s_q, num_heads,
                          static_cast<int>(num_sm_parts), attn_sink_ptr, stream);
    TVM_FFI_ICHECK(combine_ok) << "combine-v2 max_nsplits=" << num_sm_parts
                               << " exceeds the compiled MAX_SPLITS=256 ceiling";
  }
}

}  // namespace flashinfer::sparse_mla_sm120
