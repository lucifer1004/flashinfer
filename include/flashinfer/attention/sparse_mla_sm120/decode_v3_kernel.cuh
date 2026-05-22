// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// (license header — full text mirrors the other files in this directory)

#pragma once

#include "arch/barrier.cuh"
#include "arch/cp_async.cuh"
#include "arch/ldmatrix_sm120.cuh"
#include "arch/mma_sm120.cuh"
#include "common/d2_load_b.cuh"
#include "common/fp8_quant.cuh"
#include "common/online_softmax.cuh"
#include "model/kv_cache_traits.cuh"
#include "model/scale_convert.cuh"

namespace flashinfer::sparse_mla_sm120 {

// Decode-v3 (A1.1): per-block online-FA softmax over `chunks_per_block`
// V3_CAND_WINDOW chunks. Grid: (num_tokens, num_head_blocks, num_splits).
// Each block writes its (per-split) normalized partial output + LSE to
// mid_out / mid_lse; the merge kernel collapses splits as before.
//
// chunks_per_block lets the launcher trade grid-block count against per-
// block work (= waves). Streaming-acc behaviour (one chunk per block)
// is recovered with chunks_per_block=1.

constexpr int V3_BLOCK_THREADS = 128;
constexpr int V3_N_WARPS = 4;
constexpr int V3_CAND_WINDOW = 128;
constexpr int V3_BI = V3_CAND_WINDOW;
constexpr int V3_ENTRIES_PER_WARP = V3_BI / V3_N_WARPS;     // 32
constexpr int V3_QK_N_TILES = V3_ENTRIES_PER_WARP / 8;      // 4

template <ModelType MT, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE>
__global__ void __launch_bounds__(V3_BLOCK_THREADS, 1) sparse_mla_decode_v3_kernel(
    const bf16* __restrict__ Q,            // [num_tokens, num_heads, d_qk] bf16
    const uint8_t* __restrict__ KV_cache,  // FP8 paged (MODEL1 footer layout)
    const int32_t* __restrict__ indices,   // [num_tokens, topk] int32
    bf16* __restrict__ mid_out,            // [num_tokens, num_heads, num_splits, d_v] bf16
    float* __restrict__ mid_lse,           // [num_tokens, num_heads, num_splits] f32
    const int* __restrict__ topk_length_ptr,  // [num_tokens] or null
    int num_tokens, int num_splits, int chunks_per_block,
    float sm_scale, size_t stride_kv_block) {
  using KV = KVCacheTraits<MT>;
  static_assert(MT == ModelType::MODEL1, "decode-v3 currently MODEL1-only");
  constexpr int D_NOPE = KV::D_NOPE;                              // 448
  constexpr int D_ROPE_C = KV::D_ROPE;                            // 64
  constexpr int D_QK = KV::D_QK;                                  // 512
  constexpr int D_V_C = KV::D_V;                                  // 512
  constexpr int QUANT_TILE = KV::QUANT_TILE;                      // 64
  constexpr int NUM_SCALES = KV::NUM_SCALES;                      // 7
  constexpr int Q_NOPE_STRIDE = KV::Q_NOPE_STRIDE;                // 464
  constexpr int KV_SMEM_STRIDE = KV::KV_SMEM_STRIDE;              // 464
  constexpr int SCALE_BYTES_PER_TOKEN = KV::SCALE_BYTES_PER_TOKEN;  // 8
  constexpr int IO_STRIDE = D_NOPE + D_ROPE_C * 2;                // 576
  constexpr int pbs = PAGE_BLOCK_SIZE;

  const int t_idx = blockIdx.x;
  const int h_block_idx = blockIdx.y;
  const int split_idx = blockIdx.z;
  if (t_idx >= num_tokens) return;

  const int h_start = h_block_idx * HPB;
  const int topk_len = topk_length_ptr ? __ldg(topk_length_ptr + t_idx) : TOPK;

  // Chunk range this block owns.
  const int num_chunks_total = (topk_len + V3_CAND_WINDOW - 1) / V3_CAND_WINDOW;
  const int chunk_lo = split_idx * chunks_per_block;
  const int chunk_hi = min(chunk_lo + chunks_per_block, num_chunks_total);
  if (chunk_lo >= num_chunks_total) {
    if (threadIdx.x < HPB) {
      const int h = h_start + threadIdx.x;
      const size_t lse_off =
          (size_t)t_idx * NUM_HEADS * num_splits + (size_t)h * num_splits + split_idx;
      mid_lse[lse_off] = -1e30f;
    }
    return;
  }

  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int gid = lane >> 2;
  const int tid = lane & 3;

  constexpr int V_CHUNK = QUANT_TILE;                              // 64
  constexpr int N_V_CHUNKS = D_NOPE / V_CHUNK;                     // 7
  constexpr int NT_PER_WARP_XV = V_CHUNK / 8 / V3_N_WARPS;         // 2
  constexpr int XV_KSTEPS = V3_BI / 32;                            // 4
  constexpr int W_FP8_STRIDE = V3_BI + 16;                         // 144
  constexpr int ROPE_DIMS_PER_WARP = D_ROPE_C / V3_N_WARPS;        // 16
  constexpr int ROPE_N_TILES = ROPE_DIMS_PER_WARP / 8;             // 2
  constexpr int ROPE_K_ITERS = V3_BI / 16;                         // 8

  // Dynamic smem layout: same as the prior single-chunk version. Total
  // ~88KB dyn + 4KB static sm_p_full = ~92KB (1 block/SM).
  extern __shared__ __align__(16) char smem_raw[];
  size_t off = 0;
  bf16* sm_q_rope = reinterpret_cast<bf16*>(smem_raw + off);
  off += (size_t)HPB * D_ROPE_C * sizeof(bf16);
  uint8_t* sm_q_fp8 = reinterpret_cast<uint8_t*>(smem_raw + off);
  off += (size_t)HPB * Q_NOPE_STRIDE;
  float* sm_q_sc = reinterpret_cast<float*>(smem_raw + off);
  off += (size_t)HPB * NUM_SCALES * sizeof(float);
  uint8_t* sm_kv_fp8 = reinterpret_cast<uint8_t*>(smem_raw + off);
  off += (size_t)V3_BI * KV_SMEM_STRIDE;
  uint8_t* sm_kv_sc = reinterpret_cast<uint8_t*>(smem_raw + off);
  off += (size_t)V3_BI * SCALE_BYTES_PER_TOKEN;
  bf16* sm_kv_rope = reinterpret_cast<bf16*>(smem_raw + off);
  off += (size_t)V3_BI * D_ROPE_C * sizeof(bf16);
  float* sm_reduce = reinterpret_cast<float*>(smem_raw + off);
  off += (size_t)(2 * V3_N_WARPS * HPB) * sizeof(float);
  float* sm_w_head_sc = reinterpret_cast<float*>(smem_raw + off);
  off += (size_t)N_V_CHUNKS * HPB * sizeof(float);
  uint8_t* sm_w_fp8 = reinterpret_cast<uint8_t*>(smem_raw + off);

  float* sm_warp_max = sm_reduce;
  float* sm_warp_sum = sm_reduce + V3_N_WARPS * HPB;

  // ── Stage 0: Q quantization (once, before chunk loop) ──────────
  const bf16* q_base = Q + (size_t)t_idx * NUM_HEADS * D_QK + (size_t)h_start * D_QK;
  quantize_q_to_smem<MT, V3_BLOCK_THREADS>(sm_q_fp8, sm_q_sc, sm_q_rope, q_base, sm_reduce);

  // Persistent state across chunks (per-thread registers).
  float acc_nope[N_V_CHUNKS][NT_PER_WARP_XV][4] = {0};
  float acc_rope[ROPE_N_TILES][4] = {0};
  float global_max[2] = {-1e30f, -1e30f};
  float global_sum[2] = {0.f, 0.f};

  __shared__ bf16 sm_p_full[HPB][V3_BI];  // 4 KB static
  const int32_t* idx_base = indices + (size_t)t_idx * TOPK;

  // ── Chunk loop ──────────────────────────────────────────────────
  for (int chunk_idx = chunk_lo; chunk_idx < chunk_hi; ++chunk_idx) {
    const int split_cand_start = chunk_idx * V3_CAND_WINDOW;
    const int split_cand_end = min(split_cand_start + V3_CAND_WINDOW, topk_len);

    // ── Stage 1: gather KV bytes ────────────────────────────────
    {
      const int warp_first_entry = warp_id * V3_ENTRIES_PER_WARP;
#pragma unroll
      for (int eo = 0; eo < V3_ENTRIES_PER_WARP; eo++) {
        const int entry_idx = warp_first_entry + eo;
        const int cand_pos = split_cand_start + entry_idx;
        const bool is_valid_cand = (cand_pos < split_cand_end);
        int idx_raw = is_valid_cand ? idx_base[cand_pos] : -1;
        const int idx = (idx_raw >= 0) ? idx_raw : 0;
        const int block_idx_g = idx / pbs;
        const int local_idx_g = idx - block_idx_g * pbs;

        const uint8_t* data_base =
            KV_cache + (size_t)block_idx_g * stride_kv_block + (size_t)local_idx_g * IO_STRIDE;
        const uint8_t* scale_base = KV_cache + (size_t)block_idx_g * stride_kv_block +
                                    (size_t)pbs * IO_STRIDE +
                                    (size_t)local_idx_g * SCALE_BYTES_PER_TOKEN;

        if (lane < 28) {
          cp_async_16B(sm_kv_fp8 + (size_t)entry_idx * KV_SMEM_STRIDE + lane * 16,
                       data_base + lane * 16);
        }
        if (lane < 8) {
          cp_async_16B(sm_kv_rope + (size_t)entry_idx * D_ROPE_C + lane * 8,
                       data_base + D_NOPE + lane * 16);
        }
        if (lane == 28) {
          *reinterpret_cast<uint64_t*>(sm_kv_sc + (size_t)entry_idx * SCALE_BYTES_PER_TOKEN) =
              *reinterpret_cast<const uint64_t*>(scale_base);
        }
      }
    }
    cp_async_commit();
    cp_async_wait_group<0>();
    __syncthreads();

    // Zero-stomp invalid rows.
    {
      const int warp_first_entry = warp_id * V3_ENTRIES_PER_WARP;
#pragma unroll
      for (int eo = 0; eo < V3_ENTRIES_PER_WARP; eo++) {
        const int entry_idx = warp_first_entry + eo;
        const int cand_pos = split_cand_start + entry_idx;
        const bool is_valid = (cand_pos < split_cand_end) && (idx_base[cand_pos] >= 0);
        if (!is_valid) {
          uint8_t* fp8_row = sm_kv_fp8 + (size_t)entry_idx * KV_SMEM_STRIDE;
#pragma unroll
          for (int b = 0; b < (D_NOPE + 31) / 32; b++) {
            int off_b = lane * ((D_NOPE + 31) / 32) + b;
            if (off_b < D_NOPE) fp8_row[off_b] = 0;
          }
          if (lane < 8) {
            uint4 zero = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(sm_kv_rope + (size_t)entry_idx * D_ROPE_C + lane * 8) = zero;
          }
        }
      }
    }
    __syncthreads();

    // ── Stage 2 QK ──────────────────────────────────────────────
    float qk[V3_QK_N_TILES][4] = {0};
    {
      const int warp_first_cand = warp_id * V3_ENTRIES_PER_WARP;
#pragma unroll
      for (int blk = 0; blk < NUM_SCALES; blk++) {
        uint8_t sfa = fp32_to_ue8m0(sm_q_sc[(gid + (lane & 1) * 8) * NUM_SCALES + blk]);
#pragma unroll
        for (int ks = 0; ks < QUANT_TILE / 32; ks++) {
          const int ko = blk * QUANT_TILE + ks * 32;
          uint32_t a0, a1, a2, a3;
          ldmatrix_load_A_fp8(a0, a1, a2, a3, sm_q_fp8 + ko, Q_NOPE_STRIDE, lane);
#pragma unroll
          for (int nt = 0; nt < V3_QK_N_TILES; nt++) {
            const int cand_row_base = warp_first_cand + nt * 8;
            uint8_t sfb = sm_kv_sc[(cand_row_base + gid) * SCALE_BYTES_PER_TOKEN + blk];
            uint32_t b0, b1;
            ldmatrix_load_B_fp8(b0, b1,
                                sm_kv_fp8 + (size_t)cand_row_base * KV_SMEM_STRIDE + ko,
                                KV_SMEM_STRIDE, lane);
            MmaFp8Result r = mma_fp8_block_scaled_m16n8k32(
                a0, a1, a2, a3, b0, b1, qk[nt][0], qk[nt][1], qk[nt][2], qk[nt][3], sfa, sfb);
            qk[nt][0] = r.d0;
            qk[nt][1] = r.d1;
            qk[nt][2] = r.d2;
            qk[nt][3] = r.d3;
          }
        }
      }
    }
    {
      const int warp_first_cand = warp_id * V3_ENTRIES_PER_WARP;
#pragma unroll
      for (int ks = 0; ks < D_ROPE_C / 16; ks++) {
        uint32_t a0, a1, a2, a3;
        ldmatrix_load_A_bf16(a0, a1, a2, a3, sm_q_rope + ks * 16, D_ROPE_C, lane);
#pragma unroll
        for (int nt = 0; nt < V3_QK_N_TILES; nt++) {
          const int cand_row_base = warp_first_cand + nt * 8;
          uint32_t b0, b1;
          ldmatrix_load_B_bf16(b0, b1,
                               sm_kv_rope + (size_t)cand_row_base * D_ROPE_C + ks * 16,
                               D_ROPE_C, lane);
          MmaBf16Result r = mma_bf16_m16n8k16(
              a0, a1, a2, a3, b0, b1, qk[nt][0], qk[nt][1], qk[nt][2], qk[nt][3]);
          qk[nt][0] = r.d0;
          qk[nt][1] = r.d1;
          qk[nt][2] = r.d2;
          qk[nt][3] = r.d3;
        }
      }
    }

    // Mask invalid cands + sm_scale × LOG2E.
    const int warp_first_cand = warp_id * V3_ENTRIES_PER_WARP;
#pragma unroll
    for (int nt = 0; nt < V3_QK_N_TILES; nt++) {
      const int c0 = warp_first_cand + nt * 8 + tid * 2;
      const int c1 = c0 + 1;
      if (c0 + split_cand_start >= split_cand_end) {
        qk[nt][0] = -1e30f;
        qk[nt][2] = -1e30f;
      }
      if (c1 + split_cand_start >= split_cand_end) {
        qk[nt][1] = -1e30f;
        qk[nt][3] = -1e30f;
      }
      qk[nt][0] *= sm_scale * LOG2E;
      qk[nt][1] *= sm_scale * LOG2E;
      qk[nt][2] *= sm_scale * LOG2E;
      qk[nt][3] *= sm_scale * LOG2E;
    }

    // Per-warp local max/sum.
    float local_max[2] = {-1e30f, -1e30f};
#pragma unroll
    for (int nt = 0; nt < V3_QK_N_TILES; nt++) {
      local_max[0] = fmaxf(local_max[0], fmaxf(qk[nt][0], qk[nt][1]));
      local_max[1] = fmaxf(local_max[1], fmaxf(qk[nt][2], qk[nt][3]));
    }
#pragma unroll
    for (int s = 2; s >= 1; s >>= 1) {
      local_max[0] = fmaxf(local_max[0], __shfl_xor_sync(0xffffffff, local_max[0], s));
      local_max[1] = fmaxf(local_max[1], __shfl_xor_sync(0xffffffff, local_max[1], s));
    }
    float local_sum[2] = {0.f, 0.f};
    float p[V3_QK_N_TILES][4];
#pragma unroll
    for (int nt = 0; nt < V3_QK_N_TILES; nt++) {
      p[nt][0] = exp2f(qk[nt][0] - local_max[0]);
      p[nt][1] = exp2f(qk[nt][1] - local_max[0]);
      p[nt][2] = exp2f(qk[nt][2] - local_max[1]);
      p[nt][3] = exp2f(qk[nt][3] - local_max[1]);
      local_sum[0] += p[nt][0] + p[nt][1];
      local_sum[1] += p[nt][2] + p[nt][3];
    }
#pragma unroll
    for (int s = 2; s >= 1; s >>= 1) {
      local_sum[0] += __shfl_xor_sync(0xffffffff, local_sum[0], s);
      local_sum[1] += __shfl_xor_sync(0xffffffff, local_sum[1], s);
    }

    // Cross-warp reduce.
    if (tid == 0) {
      sm_warp_max[warp_id * HPB + gid] = local_max[0];
      sm_warp_max[warp_id * HPB + gid + 8] = local_max[1];
      sm_warp_sum[warp_id * HPB + gid] = local_sum[0];
      sm_warp_sum[warp_id * HPB + gid + 8] = local_sum[1];
    }
    __syncthreads();
    if (threadIdx.x < HPB) {
      const int h = threadIdx.x;
      float wmax[V3_N_WARPS], wsum[V3_N_WARPS];
#pragma unroll
      for (int w = 0; w < V3_N_WARPS; w++) {
        wmax[w] = sm_warp_max[w * HPB + h];
        wsum[w] = sm_warp_sum[w * HPB + h];
      }
      float bmax = -1e30f;
#pragma unroll
      for (int w = 0; w < V3_N_WARPS; w++) bmax = fmaxf(bmax, wmax[w]);
      float bsum = 0.f;
#pragma unroll
      for (int w = 0; w < V3_N_WARPS; w++) bsum += wsum[w] * exp2f(wmax[w] - bmax);
      sm_warp_max[h] = bmax;
      sm_warp_sum[h] = bsum;
    }
    __syncthreads();

    const float block_local_max0 = sm_warp_max[gid];
    const float block_local_max1 = sm_warp_max[gid + 8];
    const float block_local_sum0 = sm_warp_sum[gid];
    const float block_local_sum1 = sm_warp_sum[gid + 8];

    // Online softmax update.
    float new_gmax0 = fmaxf(global_max[0], block_local_max0);
    float new_gmax1 = fmaxf(global_max[1], block_local_max1);
    const float alpha0 =
        (global_max[0] > -1e29f) ? exp2f(global_max[0] - new_gmax0) : 0.f;
    const float alpha1 =
        (global_max[1] > -1e29f) ? exp2f(global_max[1] - new_gmax1) : 0.f;
    const float local_rescale0 = exp2f(block_local_max0 - new_gmax0);
    const float local_rescale1 = exp2f(block_local_max1 - new_gmax1);

    if (chunk_idx > chunk_lo) {
#pragma unroll
      for (int vc = 0; vc < N_V_CHUNKS; vc++) {
#pragma unroll
        for (int nt = 0; nt < NT_PER_WARP_XV; nt++) {
          acc_nope[vc][nt][0] *= alpha0;
          acc_nope[vc][nt][1] *= alpha0;
          acc_nope[vc][nt][2] *= alpha1;
          acc_nope[vc][nt][3] *= alpha1;
        }
      }
#pragma unroll
      for (int nt = 0; nt < ROPE_N_TILES; nt++) {
        acc_rope[nt][0] *= alpha0;
        acc_rope[nt][1] *= alpha0;
        acc_rope[nt][2] *= alpha1;
        acc_rope[nt][3] *= alpha1;
      }
      global_sum[0] = global_sum[0] * alpha0 + block_local_sum0 * local_rescale0;
      global_sum[1] = global_sum[1] * alpha1 + block_local_sum1 * local_rescale1;
    } else {
      global_sum[0] = block_local_sum0 * local_rescale0;
      global_sum[1] = block_local_sum1 * local_rescale1;
    }
    global_max[0] = new_gmax0;
    global_max[1] = new_gmax1;

    // Stage 2.75: sm_p_full = p * local_rescale (rescaled-only; not / gsum).
    // w_pre also feeds Stage 3 NoPE Phase 1 (amax) + Phase 3 (quant).
    float w_pre[V3_QK_N_TILES][4];
#pragma unroll
    for (int nt = 0; nt < V3_QK_N_TILES; nt++) {
      w_pre[nt][0] = p[nt][0] * local_rescale0;
      w_pre[nt][1] = p[nt][1] * local_rescale0;
      w_pre[nt][2] = p[nt][2] * local_rescale1;
      w_pre[nt][3] = p[nt][3] * local_rescale1;
    }
    const int cand_col_base = warp_id * V3_ENTRIES_PER_WARP;
#pragma unroll
    for (int nt = 0; nt < V3_QK_N_TILES; nt++) {
      const int c0 = nt * 8 + tid * 2;
      const int c1 = c0 + 1;
      sm_p_full[gid][cand_col_base + c0] = __float2bfloat16(w_pre[nt][0]);
      sm_p_full[gid][cand_col_base + c1] = __float2bfloat16(w_pre[nt][1]);
      sm_p_full[gid + 8][cand_col_base + c0] = __float2bfloat16(w_pre[nt][2]);
      sm_p_full[gid + 8][cand_col_base + c1] = __float2bfloat16(w_pre[nt][3]);
    }
    __syncthreads();

    // ── Stage 3 NoPE FP8 (accumulates into acc_nope) ───────────
    for (int i = threadIdx.x; i < N_V_CHUNKS * HPB; i += V3_BLOCK_THREADS) {
      sm_w_head_sc[i] = 0.f;
    }
    __syncthreads();
    {
      const int warp_first_cand_xv = warp_id * V3_ENTRIES_PER_WARP;
#pragma unroll
      for (int nt = 0; nt < V3_QK_N_TILES; nt++) {
        const int cand_e0 = warp_first_cand_xv + nt * 8 + tid * 2;
        const int cand_e1 = cand_e0 + 1;
#pragma unroll
        for (int vc = 0; vc < N_V_CHUNKS; vc++) {
          const float vsc0 = ue8m0_to_fp32(
              sm_kv_sc[(size_t)cand_e0 * SCALE_BYTES_PER_TOKEN + vc]);
          const float vsc1 = ue8m0_to_fp32(
              sm_kv_sc[(size_t)cand_e1 * SCALE_BYTES_PER_TOKEN + vc]);
          atomicMax(reinterpret_cast<int*>(&sm_w_head_sc[vc * HPB + gid]),
                    __float_as_int(fmaxf(fabsf(w_pre[nt][0] * vsc0),
                                         fabsf(w_pre[nt][1] * vsc1))));
          atomicMax(reinterpret_cast<int*>(&sm_w_head_sc[vc * HPB + gid + 8]),
                    __float_as_int(fmaxf(fabsf(w_pre[nt][2] * vsc0),
                                         fabsf(w_pre[nt][3] * vsc1))));
        }
      }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < N_V_CHUNKS * HPB; i += V3_BLOCK_THREADS) {
      sm_w_head_sc[i] = fmaxf(sm_w_head_sc[i], 1e-10f) / FP8_MAX;
    }
    __syncthreads();

#pragma unroll
    for (int vc = 0; vc < N_V_CHUNKS; vc++) {
      if (vc > 0) __syncthreads();
      // Phase 3 quant.
      {
        const int warp_first_cand_xv = warp_id * V3_ENTRIES_PER_WARP;
        const float si0 = 1.f / sm_w_head_sc[vc * HPB + gid];
        const float si1 = 1.f / sm_w_head_sc[vc * HPB + gid + 8];
#pragma unroll
        for (int nt = 0; nt < V3_QK_N_TILES; nt++) {
          const int cand_e0 = warp_first_cand_xv + nt * 8 + tid * 2;
          const int cand_e1 = cand_e0 + 1;
          const float vsc0 = ue8m0_to_fp32(
              sm_kv_sc[(size_t)cand_e0 * SCALE_BYTES_PER_TOKEN + vc]);
          const float vsc1 = ue8m0_to_fp32(
              sm_kv_sc[(size_t)cand_e1 * SCALE_BYTES_PER_TOKEN + vc]);
          __nv_fp8_e4m3 f00(fmaxf(FP8_MIN, fminf(FP8_MAX, w_pre[nt][0] * vsc0 * si0)));
          __nv_fp8_e4m3 f01(fmaxf(FP8_MIN, fminf(FP8_MAX, w_pre[nt][1] * vsc1 * si0)));
          __nv_fp8_e4m3 f10(fmaxf(FP8_MIN, fminf(FP8_MAX, w_pre[nt][2] * vsc0 * si1)));
          __nv_fp8_e4m3 f11(fmaxf(FP8_MIN, fminf(FP8_MAX, w_pre[nt][3] * vsc1 * si1)));
          sm_w_fp8[(size_t)gid * W_FP8_STRIDE + cand_e0] = f00.__x;
          sm_w_fp8[(size_t)gid * W_FP8_STRIDE + cand_e1] = f01.__x;
          sm_w_fp8[(size_t)(gid + 8) * W_FP8_STRIDE + cand_e0] = f10.__x;
          sm_w_fp8[(size_t)(gid + 8) * W_FP8_STRIDE + cand_e1] = f11.__x;
        }
      }
      __syncthreads();
      // Phase 4 FP8 MMA. Accumulate into persistent acc_nope[vc][nt][k].
      const float sc0 = sm_w_head_sc[vc * HPB + gid];
      const float sc1 = sm_w_head_sc[vc * HPB + gid + 8];
#pragma unroll
      for (int nt = 0; nt < NT_PER_WARP_XV; nt++) {
        const int dim = vc * V_CHUNK + warp_id * (NT_PER_WARP_XV * 8) + nt * 8;
        float xv[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kstep = 0; kstep < XV_KSTEPS; kstep++) {
          const int ko = kstep * 32;
          uint32_t a0, a1, a2, a3, b0, b1;
          ldmatrix_load_A_fp8(a0, a1, a2, a3, sm_w_fp8 + ko, W_FP8_STRIDE, lane);
          d2_load_b_fp8<KV_SMEM_STRIDE>(b0, b1, sm_kv_fp8, kstep * 32, dim, lane);
          MmaFp8Result r =
              mma_fp8_m16n8k32(a0, a1, a2, a3, b0, b1, xv[0], xv[1], xv[2], xv[3]);
          xv[0] = r.d0;
          xv[1] = r.d1;
          xv[2] = r.d2;
          xv[3] = r.d3;
        }
        acc_nope[vc][nt][0] += xv[0] * sc0;
        acc_nope[vc][nt][1] += xv[1] * sc0;
        acc_nope[vc][nt][2] += xv[2] * sc1;
        acc_nope[vc][nt][3] += xv[3] * sc1;
      }
    }

    // ── Stage 3 RoPE bf16 (accumulates into acc_rope) ──────────
    {
      const int rope_dim_base = warp_id * ROPE_DIMS_PER_WARP;
#pragma unroll
      for (int ks = 0; ks < ROPE_K_ITERS; ks++) {
        uint32_t a0, a1, a2, a3;
        ldmatrix_load_A_bf16(a0, a1, a2, a3,
                             reinterpret_cast<const bf16*>(&sm_p_full[0][ks * 16]),
                             V3_BI, lane);
#pragma unroll
        for (int nt = 0; nt < ROPE_N_TILES; nt++) {
          const int n_col = rope_dim_base + nt * 8;
          const int k_base = ks * 16;
          const int ent0 = k_base + tid * 2;
          const int ent1 = ent0 + 1;
          const int ent8 = ent0 + 8;
          const int ent9 = ent0 + 9;
          const int col = n_col + gid;
          uint16_t v0 = *reinterpret_cast<const uint16_t*>(
              sm_kv_rope + (size_t)ent0 * D_ROPE_C + col);
          uint16_t v1 = *reinterpret_cast<const uint16_t*>(
              sm_kv_rope + (size_t)ent1 * D_ROPE_C + col);
          uint16_t v8 = *reinterpret_cast<const uint16_t*>(
              sm_kv_rope + (size_t)ent8 * D_ROPE_C + col);
          uint16_t v9 = *reinterpret_cast<const uint16_t*>(
              sm_kv_rope + (size_t)ent9 * D_ROPE_C + col);
          uint32_t b0 = (uint32_t)v0 | ((uint32_t)v1 << 16);
          uint32_t b1 = (uint32_t)v8 | ((uint32_t)v9 << 16);
          MmaBf16Result r = mma_bf16_m16n8k16(
              a0, a1, a2, a3, b0, b1, acc_rope[nt][0], acc_rope[nt][1],
              acc_rope[nt][2], acc_rope[nt][3]);
          acc_rope[nt][0] = r.d0;
          acc_rope[nt][1] = r.d1;
          acc_rope[nt][2] = r.d2;
          acc_rope[nt][3] = r.d3;
        }
      }
    }
    __syncthreads();
  }  // chunk loop

  // ── Write per-split partial output + LSE to mid_out / mid_lse ───
  // Partial output = acc / global_sum (normalized over this block's chunks).
  // LSE = log2(global_sum) + global_max — merge kernel combines splits.
  const float inv_g0 = (global_sum[0] > 0.f) ? (1.f / global_sum[0]) : 0.f;
  const float inv_g1 = (global_sum[1] > 0.f) ? (1.f / global_sum[1]) : 0.f;

  const size_t mid_o_base =
      ((size_t)t_idx * NUM_HEADS + h_start) * (size_t)num_splits * D_V_C +
      (size_t)split_idx * D_V_C;

#pragma unroll
  for (int vc = 0; vc < N_V_CHUNKS; vc++) {
#pragma unroll
    for (int nt = 0; nt < NT_PER_WARP_XV; nt++) {
      const int d0 = vc * V_CHUNK + warp_id * (NT_PER_WARP_XV * 8) + nt * 8 + tid * 2;
      const int d1 = d0 + 1;
      mid_out[mid_o_base + (size_t)gid * num_splits * D_V_C + d0] =
          __float2bfloat16(acc_nope[vc][nt][0] * inv_g0);
      mid_out[mid_o_base + (size_t)gid * num_splits * D_V_C + d1] =
          __float2bfloat16(acc_nope[vc][nt][1] * inv_g0);
      mid_out[mid_o_base + (size_t)(gid + 8) * num_splits * D_V_C + d0] =
          __float2bfloat16(acc_nope[vc][nt][2] * inv_g1);
      mid_out[mid_o_base + (size_t)(gid + 8) * num_splits * D_V_C + d1] =
          __float2bfloat16(acc_nope[vc][nt][3] * inv_g1);
    }
  }
  {
    const int rope_dim_base = warp_id * ROPE_DIMS_PER_WARP;
#pragma unroll
    for (int nt = 0; nt < ROPE_N_TILES; nt++) {
      const int d0 = D_NOPE + rope_dim_base + nt * 8 + tid * 2;
      const int d1 = d0 + 1;
      mid_out[mid_o_base + (size_t)gid * num_splits * D_V_C + d0] =
          __float2bfloat16(acc_rope[nt][0] * inv_g0);
      mid_out[mid_o_base + (size_t)gid * num_splits * D_V_C + d1] =
          __float2bfloat16(acc_rope[nt][1] * inv_g0);
      mid_out[mid_o_base + (size_t)(gid + 8) * num_splits * D_V_C + d0] =
          __float2bfloat16(acc_rope[nt][2] * inv_g1);
      mid_out[mid_o_base + (size_t)(gid + 8) * num_splits * D_V_C + d1] =
          __float2bfloat16(acc_rope[nt][3] * inv_g1);
    }
  }
  // Per-head LSE write: warp 0 lanes 0,4,..,28 (tid==0) own all 16 heads.
  if (warp_id == 0 && tid == 0) {
    const float lse0 =
        (global_sum[0] > 0.f) ? (log2f(global_sum[0]) + global_max[0]) : -1e30f;
    const float lse1 =
        (global_sum[1] > 0.f) ? (log2f(global_sum[1]) + global_max[1]) : -1e30f;
    const size_t lse_base =
        (size_t)t_idx * NUM_HEADS * num_splits + (size_t)h_start * num_splits;
    mid_lse[lse_base + (size_t)gid * num_splits + split_idx] = lse0;
    mid_lse[lse_base + (size_t)(gid + 8) * num_splits + split_idx] = lse1;
  }
}

// Merge kernel: collapse splits → final output + LSE. Same as prior S3.
// Grid: (num_tokens, num_heads, D_V_PARTS). Block: 32 threads (1 warp).
template <int NUM_HEADS, int D_V_VAL, int D_V_PARTS>
__global__ void __launch_bounds__(32, 16) sparse_mla_decode_v3_merge_kernel(
    const bf16* __restrict__ mid_out,
    const float* __restrict__ mid_lse,
    bf16* __restrict__ output,
    float* __restrict__ out_lse,
    int num_tokens, int num_splits) {
  static_assert(D_V_VAL % D_V_PARTS == 0, "D_V must divide evenly by D_V_PARTS");
  constexpr int DIMS_PER_PART = D_V_VAL / D_V_PARTS;
  static_assert(DIMS_PER_PART % 32 == 0, "DIMS_PER_PART must be a multiple of 32");
  constexpr int DIMS_PER_LANE = DIMS_PER_PART / 32;

  const int t_idx = blockIdx.x;
  const int h = blockIdx.y;
  const int dim_part = blockIdx.z;
  if (t_idx >= num_tokens) return;
  if (h >= NUM_HEADS) return;
  const int lane = threadIdx.x;
  const int part_base = dim_part * DIMS_PER_PART;

  float my_lse[16];
  const float* lse_ptr = mid_lse + (size_t)t_idx * NUM_HEADS * num_splits + (size_t)h * num_splits;
  const int splits_per_thread = (num_splits + 31) / 32;
  float local_max = -1e30f;
#pragma unroll
  for (int i = 0; i < 16; i++) my_lse[i] = -1e30f;
  for (int i = 0; i < splits_per_thread; i++) {
    int sp = i * 32 + lane;
    if (sp < num_splits) {
      my_lse[i] = lse_ptr[sp];
      local_max = fmaxf(local_max, my_lse[i]);
    }
  }
#pragma unroll
  for (int s = 16; s >= 1; s >>= 1) {
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, s));
  }
  float global_max = local_max;
  if (global_max <= -1e29f) global_max = 0.f;

  float local_sum = 0.f;
  for (int i = 0; i < splits_per_thread; i++) {
    if (my_lse[i] > -1e29f) local_sum += exp2f(my_lse[i] - global_max);
  }
#pragma unroll
  for (int s = 16; s >= 1; s >>= 1) {
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, s);
  }
  const float global_sum = local_sum;
  const float global_lse =
      (global_sum > 0.f) ? (log2f(global_sum) + global_max) : -1e30f;
  const float inv_global_sum = (global_sum > 0.f) ? (1.f / global_sum) : 0.f;

  const bf16* mid_base =
      mid_out + (size_t)t_idx * NUM_HEADS * num_splits * D_V_VAL + (size_t)h * num_splits * D_V_VAL;
  bf16* out_ptr = output + (size_t)t_idx * NUM_HEADS * D_V_VAL + (size_t)h * D_V_VAL;

  float acc[DIMS_PER_LANE];
#pragma unroll
  for (int d = 0; d < DIMS_PER_LANE; d++) acc[d] = 0.f;

  for (int sp = 0; sp < num_splits; sp++) {
    float lse_sp = lse_ptr[sp];
    if (lse_sp <= -1e29f) continue;
    float weight = exp2f(lse_sp - global_max);
#pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) {
      int local_d = lane * DIMS_PER_LANE + d;
      float v = __bfloat162float(mid_base[(size_t)sp * D_V_VAL + part_base + local_d]);
      acc[d] += weight * v;
    }
  }

#pragma unroll
  for (int d = 0; d < DIMS_PER_LANE; d++) {
    int local_d = lane * DIMS_PER_LANE + d;
    out_ptr[part_base + local_d] = __float2bfloat16(acc[d] * inv_global_sum);
  }
  if (out_lse != nullptr && lane == 0 && dim_part == 0) {
    out_lse[(size_t)t_idx * NUM_HEADS + h] = global_lse;
  }
}

}  // namespace flashinfer::sparse_mla_sm120
