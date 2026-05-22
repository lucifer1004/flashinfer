// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// TVM-FFI binding for sparse-MLA SM120 paged attention.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <flashinfer/attention/sparse_mla_sm120/model/model_type.h>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace flashinfer::sparse_mla_sm120 {

using bf16 = __nv_bfloat16;

void SparseMlaSm120PagedAttention(TensorView q, TensorView kv_cache, TensorView indices,
                                  TensorView output, TensorView out_lse, TensorView o_accum,
                                  TensorView lse_accum, TensorView sched_meta,
                                  TensorView num_splits, double sm_scale, int64_t num_sm_parts,
                                  Optional<TensorView> topk_length, Optional<TensorView> attn_sink,
                                  Optional<TensorView> extra_kv_cache,
                                  Optional<TensorView> extra_indices,
                                  Optional<TensorView> extra_topk_length);

bool launch_sparse_mla_decode_v3(ModelType mt, int num_heads, int topk,
                                 int page_block_size, int num_tokens,
                                 int num_splits, const bf16* Q,
                                 const uint8_t* KV_cache, const int32_t* indices,
                                 bf16* mid_out, float* mid_lse, bf16* output,
                                 float* out_lse, const int* topk_length,
                                 const float* attn_sink,
                                 const uint8_t* extra_KV_cache,
                                 const int32_t* extra_indices,
                                 const int* extra_topk_length,
                                 int extra_topk, int pbs_extra,
                                 size_t stride_extra_kv_block,
                                 int chunks_per_block_override,
                                 float sm_scale, size_t stride_kv_block,
                                 cudaStream_t stream);

bool launch_sparse_mla_decode_v3_backup(ModelType mt, int num_heads, int topk,
                                        int page_block_size, int num_tokens,
                                        int num_splits, const bf16* Q,
                                        const uint8_t* KV_cache, const int32_t* indices,
                                        bf16* mid_out, float* mid_lse, bf16* output,
                                        float* out_lse, const int* topk_length,
                                        float sm_scale, size_t stride_kv_block,
                                        cudaStream_t stream);

// Thin TVM-FFI wrapper for the decode-v3 standalone path. The caller passes
// already-sized scratch tensors mid_out + mid_lse plus the output and lse.
// Currently only handles MODEL1 h=128 topk=512 pbs=64.
void SparseMlaSm120DecodeV3(TensorView q, TensorView kv_cache, TensorView indices,
                            TensorView mid_out, TensorView mid_lse, TensorView output,
                            TensorView out_lse, int64_t num_splits, double sm_scale,
                            Optional<TensorView> topk_length,
                            Optional<TensorView> attn_sink,
                            Optional<TensorView> extra_kv_cache,
                            Optional<TensorView> extra_indices,
                            Optional<TensorView> extra_topk_length,
                            int64_t chunks_per_block_override) {
  TVM_FFI_ICHECK_EQ(q.ndim(), 3) << "q must be [T, H, D_QK]";
  TVM_FFI_ICHECK_EQ(kv_cache.ndim(), 2) << "kv_cache must be [num_blocks, page_bytes]";
  TVM_FFI_ICHECK_EQ(indices.ndim(), 2) << "indices must be [T, topk]";

  const int num_tokens = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int topk = static_cast<int>(indices.size(1));
  const int d_qk = static_cast<int>(q.size(2));
  ModelType mt = (d_qk == 512) ? ModelType::MODEL1 : ModelType::V32;
  // Currently the kernel only supports MODEL1.
  TVM_FFI_ICHECK_EQ(static_cast<int>(mt), static_cast<int>(ModelType::MODEL1))
      << "decode-v3 currently MODEL1-only";

  const size_t stride_kv_block = static_cast<size_t>(kv_cache.size(1));
  // page_block_size is implicit: 36864 bytes data + 512 bytes scales / 584 per
  // token = 64 tokens/block for MODEL1.
  constexpr int page_block_size = 64;

  const int* topk_len_ptr =
      topk_length.has_value() ? static_cast<const int*>(topk_length.value().data_ptr()) : nullptr;
  const float* attn_sink_ptr =
      attn_sink.has_value() ? static_cast<const float*>(attn_sink.value().data_ptr()) : nullptr;
  const uint8_t* extra_kv_ptr = extra_kv_cache.has_value()
                                    ? static_cast<const uint8_t*>(extra_kv_cache.value().data_ptr())
                                    : nullptr;
  const int32_t* extra_indices_ptr =
      extra_indices.has_value()
          ? static_cast<const int32_t*>(extra_indices.value().data_ptr())
          : nullptr;
  const int* extra_topk_len_ptr =
      extra_topk_length.has_value()
          ? static_cast<const int*>(extra_topk_length.value().data_ptr())
          : nullptr;
  // extra_topk and stride_extra_kv_block derived from the optional tensors.
  // pbs_extra: when extra cache is 4D [num_blocks, pbs_extra, 1, bpt] take
  // pbs_extra from dim -3; when caller passes 2D [num_blocks, pbs_extra * bpt]
  // we infer from the total row width / BPT_MODEL1 (= 584).
  int extra_topk_arg = 0;
  int pbs_extra_arg = 0;
  size_t stride_extra_kv_block = 0;
  if (extra_kv_cache.has_value()) {
    const auto& ekv = extra_kv_cache.value();
    extra_topk_arg = static_cast<int>(extra_indices.value().size(1));
    if (ekv.ndim() >= 3) {
      pbs_extra_arg = static_cast<int>(ekv.size(-3));
      // row stride = pbs * bpt, derive from total trailing size
      size_t row_bytes = 1;
      for (int d = 1; d < ekv.ndim(); ++d) row_bytes *= static_cast<size_t>(ekv.size(d));
      stride_extra_kv_block = row_bytes;
    } else {
      // 2D fallback: assume MODEL1 bpt = 584. Infer pbs from row width.
      constexpr int BPT_MODEL1 = 584;
      stride_extra_kv_block = static_cast<size_t>(ekv.size(1));
      pbs_extra_arg = static_cast<int>(stride_extra_kv_block / BPT_MODEL1);
    }
  }

  cudaStream_t stream = get_stream(q.device());
  bool ok = launch_sparse_mla_decode_v3(
      mt, num_heads, topk, page_block_size, num_tokens, static_cast<int>(num_splits),
      static_cast<const bf16*>(q.data_ptr()),
      static_cast<const uint8_t*>(kv_cache.data_ptr()),
      static_cast<const int32_t*>(indices.data_ptr()),
      static_cast<bf16*>(mid_out.data_ptr()), static_cast<float*>(mid_lse.data_ptr()),
      static_cast<bf16*>(output.data_ptr()), static_cast<float*>(out_lse.data_ptr()),
      topk_len_ptr, attn_sink_ptr,
      extra_kv_ptr, extra_indices_ptr, extra_topk_len_ptr,
      extra_topk_arg, pbs_extra_arg, stride_extra_kv_block,
      static_cast<int>(chunks_per_block_override),
      static_cast<float>(sm_scale), stride_kv_block, stream);
  TVM_FFI_ICHECK(ok) << "decode-v3 launch failed (unsupported shape or kernel error)";
}

// Frozen snapshot of decode-v3 at commit 41ac1687 (16/16 vs v2, 12/16 vs jasl).
// Same Python-facing surface as SparseMlaSm120DecodeV3, used as the regression
// baseline for future decode-v3 tuning.
void SparseMlaSm120DecodeV3Backup(TensorView q, TensorView kv_cache, TensorView indices,
                                  TensorView mid_out, TensorView mid_lse, TensorView output,
                                  TensorView out_lse, int64_t num_splits, double sm_scale,
                                  Optional<TensorView> topk_length) {
  TVM_FFI_ICHECK_EQ(q.ndim(), 3) << "q must be [T, H, D_QK]";
  TVM_FFI_ICHECK_EQ(kv_cache.ndim(), 2) << "kv_cache must be [num_blocks, page_bytes]";
  TVM_FFI_ICHECK_EQ(indices.ndim(), 2) << "indices must be [T, topk]";

  const int num_tokens = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int topk = static_cast<int>(indices.size(1));
  const int d_qk = static_cast<int>(q.size(2));
  ModelType mt = (d_qk == 512) ? ModelType::MODEL1 : ModelType::V32;
  TVM_FFI_ICHECK_EQ(static_cast<int>(mt), static_cast<int>(ModelType::MODEL1))
      << "decode-v3-backup currently MODEL1-only";

  const size_t stride_kv_block = static_cast<size_t>(kv_cache.size(1));
  constexpr int page_block_size = 64;

  const int* topk_len_ptr =
      topk_length.has_value() ? static_cast<const int*>(topk_length.value().data_ptr()) : nullptr;

  cudaStream_t stream = get_stream(q.device());
  bool ok = launch_sparse_mla_decode_v3_backup(
      mt, num_heads, topk, page_block_size, num_tokens, static_cast<int>(num_splits),
      static_cast<const bf16*>(q.data_ptr()),
      static_cast<const uint8_t*>(kv_cache.data_ptr()),
      static_cast<const int32_t*>(indices.data_ptr()),
      static_cast<bf16*>(mid_out.data_ptr()), static_cast<float*>(mid_lse.data_ptr()),
      static_cast<bf16*>(output.data_ptr()), static_cast<float*>(out_lse.data_ptr()),
      topk_len_ptr, static_cast<float>(sm_scale), stride_kv_block, stream);
  TVM_FFI_ICHECK(ok) << "decode-v3-backup launch failed (unsupported shape or kernel error)";
}

}  // namespace flashinfer::sparse_mla_sm120

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_paged_attention,
                              flashinfer::sparse_mla_sm120::SparseMlaSm120PagedAttention);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_decode_v3,
                              flashinfer::sparse_mla_sm120::SparseMlaSm120DecodeV3);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sparse_mla_sm120_decode_v3_backup,
                              flashinfer::sparse_mla_sm120::SparseMlaSm120DecodeV3Backup);
