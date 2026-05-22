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

// Sparse-MLA SM120 combine kernel (v2): per-batch split indexing via
// num_splits_ptr. Merges decode-dsv3_2 partial outputs into final bf16 output
// and float32 LSE. Vectorized float4 loads with split-level prefetch.
//
// Raw-pointer interface; framework-agnostic.

#include <cuda_runtime.h>

#include <flashinfer/attention/sparse_mla_sm120/arch/common.cuh>
#include <flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh>

namespace flashinfer::sparse_mla_sm120 {

namespace {

constexpr int COMBINE_BLOCK_H = 8;
constexpr int COMBINE_THREADS = COMBINE_BLOCK_H * 32;     // 256
constexpr int COMBINE_ELEMS_PER_THREAD = D_V / (32 * 4);  // 512/(32*4) = 4

struct CombineV2Params {
  const float* o_accum;
  const float* lse_accum;
  bf16* output;
  float* out_lse;
  const int* num_splits_ptr;  // [batch + 1] prefix sum
  int num_heads;
  int s_q;
  size_t stride_oa_split;  // s_q * num_heads * D_V
  size_t stride_la_split;  // s_q * num_heads
  const float* attn_sink;  // [num_heads] or nullptr
};

template <int MAX_SPLITS>
__global__ void __launch_bounds__(COMBINE_THREADS)
    sparse_mla_combine_v2_kernel(__grid_constant__ const CombineV2Params params) {
  cudaGridDependencySynchronize();

  const int batch_sq_idx = blockIdx.x;
  const int batch_idx = batch_sq_idx / params.s_q;
  const int s_q_idx = batch_sq_idx % params.s_q;
  const int h_block = blockIdx.z;
  const int warp_idx = threadIdx.x / 32;
  const int lane_idx = threadIdx.x % 32;
  const int h = h_block * COMBINE_BLOCK_H + warp_idx;
  if (h >= params.num_heads) return;

  const int start_split = __ldg(params.num_splits_ptr + batch_idx);
  const int end_split = __ldg(params.num_splits_ptr + batch_idx + 1);
  const int my_nsplits = end_split - start_split;

  if (my_nsplits <= 1) return;

  const float* __restrict__ oaccum_ptr = params.o_accum +
                                         (size_t)start_split * params.stride_oa_split +
                                         (size_t)s_q_idx * params.num_heads * D_V + (size_t)h * D_V;
  const size_t oa_split_stride = params.stride_oa_split;

  const float* __restrict__ lse_ptr = params.lse_accum +
                                      (size_t)start_split * params.stride_la_split +
                                      (size_t)s_q_idx * params.num_heads + h;
  const size_t la_split_stride = params.stride_la_split;

  __shared__ float smem_buf[COMBINE_BLOCK_H][MAX_SPLITS];

  constexpr int NUM_LSE_PER_THREAD = (MAX_SPLITS + 31) / 32;
  float local_lse[NUM_LSE_PER_THREAD];

#pragma unroll
  for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
    int sp = i * 32 + lane_idx;
    local_lse[i] = (sp < my_nsplits) ? lse_ptr[sp * la_split_stride] : -1e30f;
  }

  float max_lse = -1e30f;
#pragma unroll
  for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) max_lse = fmaxf(max_lse, local_lse[i]);
  max_lse = warp_reduce_max(max_lse);
  if (max_lse == -1e30f) max_lse = 0.f;

  float sum_lse = 0.f;
#pragma unroll
  for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) sum_lse += exp2f(local_lse[i] - max_lse);
  sum_lse = warp_reduce_sum(sum_lse);

  float global_lse = (sum_lse > 0.f) ? (log2f(sum_lse) + max_lse) : -1e30f;

  if (params.attn_sink != nullptr) {
    float sink_log2 = __ldg(params.attn_sink + h) * LOG2E;
    if (global_lse != -1e30f)
      global_lse += log2f(1.f + exp2f(sink_log2 - global_lse));
    else
      global_lse = sink_log2;
  }

  if (lane_idx == 0) {
    size_t lse_out_idx = (size_t)batch_sq_idx * params.num_heads + h;
    params.out_lse[lse_out_idx] = global_lse;
  }

#pragma unroll
  for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
    int sp = i * 32 + lane_idx;
    if (sp < MAX_SPLITS) smem_buf[warp_idx][sp] = exp2f(local_lse[i] - global_lse);
  }
  __syncwarp();

  float4 datas[COMBINE_ELEMS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < COMBINE_ELEMS_PER_THREAD; ++i)
    datas[i] = *(const float4*)(oaccum_ptr + lane_idx * 4 + i * 128);

  float4 result[COMBINE_ELEMS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < COMBINE_ELEMS_PER_THREAD; ++i) result[i] = {0.f, 0.f, 0.f, 0.f};

#pragma unroll 1
  for (int sp = 0; sp < my_nsplits; ++sp) {
    float lse_scale = smem_buf[warp_idx][sp];
#pragma unroll
    for (int i = 0; i < COMBINE_ELEMS_PER_THREAD; ++i) {
      result[i].x += lse_scale * datas[i].x;
      result[i].y += lse_scale * datas[i].y;
      result[i].z += lse_scale * datas[i].z;
      result[i].w += lse_scale * datas[i].w;
      if (sp != my_nsplits - 1) {
        datas[i] = *(const float4*)(oaccum_ptr + (size_t)(sp + 1) * oa_split_stride + lane_idx * 4 +
                                    i * 128);
      }
    }
  }

  bf16* o_ptr = params.output + (size_t)batch_sq_idx * params.num_heads * D_V + (size_t)h * D_V;

#pragma unroll
  for (int i = 0; i < COMBINE_ELEMS_PER_THREAD; ++i) {
    bf16 b[4];
    b[0] = __float2bfloat16(result[i].x);
    b[1] = __float2bfloat16(result[i].y);
    b[2] = __float2bfloat16(result[i].z);
    b[3] = __float2bfloat16(result[i].w);
    *(uint64_t*)(o_ptr + lane_idx * 4 + i * 128) = *(const uint64_t*)b;
  }
}

#define COMBINE_SPLITS_SWITCH(NSPLITS, NAME, ...) \
  [&] {                                           \
    if ((NSPLITS) <= 32) {                        \
      constexpr int NAME = 32;                    \
      return __VA_ARGS__();                       \
    } else if ((NSPLITS) <= 64) {                 \
      constexpr int NAME = 64;                    \
      return __VA_ARGS__();                       \
    } else if ((NSPLITS) <= 128) {                \
      constexpr int NAME = 128;                   \
      return __VA_ARGS__();                       \
    } else if ((NSPLITS) <= 256) {                \
      constexpr int NAME = 256;                   \
      return __VA_ARGS__();                       \
    } else {                                      \
      return false;                               \
    }                                             \
  }()

}  // namespace

// Launch combine-dsv3_2.
// MAX_SPLITS=256 covers the maximum num_sm_parts the sm120 scheduler emits
// (RTX PRO 6000 Blackwell = 188 SMs). Returns true on dispatch, false if
// max_nsplits exceeds the compiled ceiling (caller should error out).
bool launch_combine_dsv3_2(const float* o_accum, const float* lse_accum, bf16* output, float* out_lse,
                       const int* num_splits_ptr, int batch, int s_q, int num_heads,
                       int max_nsplits, const float* attn_sink, cudaStream_t stream) {
  size_t stride_oa_split = (size_t)s_q * num_heads * D_V;
  size_t stride_la_split = (size_t)s_q * num_heads;

  return COMBINE_SPLITS_SWITCH(max_nsplits, MAX_SPLITS, [&] {
    dim3 grid(batch * s_q, 1, (num_heads + COMBINE_BLOCK_H - 1) / COMBINE_BLOCK_H);
    dim3 block(COMBINE_THREADS);
    size_t smem_bytes = COMBINE_BLOCK_H * MAX_SPLITS * sizeof(float);

    auto kernel = sparse_mla_combine_v2_kernel<MAX_SPLITS>;
    if (smem_bytes > 48 * 1024) {
      CUDA_CHECK(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }

    CombineV2Params params{o_accum,   lse_accum, output,          out_lse,         num_splits_ptr,
                           num_heads, s_q,       stride_oa_split, stride_la_split, attn_sink};

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{grid, block, smem_bytes, stream, attrs, 1};
    void* args[] = {(void*)&params};
    CUDA_CHECK(cudaLaunchKernelExC(&config, (const void*)kernel, args));
    return true;
  });
}

#undef COMBINE_SPLITS_SWITCH

}  // namespace flashinfer::sparse_mla_sm120
