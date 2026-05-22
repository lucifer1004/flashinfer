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

// Decode-v3: small-block sparse-MLA decode optimised for the high-topk
// contested-shape regime.
//
// Grid: (num_tokens, num_head_blocks, num_splits)
//   num_head_blocks = num_heads / HPB
//   num_splits     = topk / CAND_WINDOW (= 64 in S1)
//
// Block: 128 threads (4 warps), uniform — no warp specialisation.
//
// FP8 path (S1):
//   sm_q_fp8     [HPB,  Q_NOPE_STRIDE] u8        (FP8 e4m3, padded for ldmatrix)
//   sm_q_sc      [HPB,  NUM_SCALES]    f32       (per-tile power-of-2 scale; ue8m0 derived)
//   sm_q_rope    [HPB,  D_ROPE]        bf16
//   sm_kv_fp8    [V3_BI, KV_SMEM_STRIDE] u8      (FP8 e4m3, padded for ldmatrix)
//   sm_kv_sc     [V3_BI, SCALE_BYTES_PER_TOKEN] u8 (UE8M0 bytes, first NUM_SCALES used)
//   sm_kv_rope   [V3_BI, D_ROPE]       bf16
//   sm_reduce    [max(HPB*NUM_SCALES, 2*N_WARPS*HPB)] f32  (alias: amax during quant,
//                                                          warp_max/warp_sum during softmax)
//   sm_p_full    [HPB,  V3_BI]         bf16      (static; cross-warp normalized softmax)
//
// Stage 2 QK:
//   NoPE k-dim 0..447 — block-scaled FP8 m16n8k32 (14 k-iters at k=32)
//   RoPE k-dim 448..511 — bf16 m16n8k16 (4 k-iters at k=16)
//
// Stage 3 XV (bf16 MMA, B-operand FP8-dequant in registers per MMA tile):
//   warps 0..2 cover D_V dims [0..384) — all NoPE, FP8 → bf16 register dequant
//   warp 3 covers D_V dims [384..512) — first 8 N-tiles NoPE, last 8 RoPE (from sm_kv_rope)
//
// First cut: MODEL1 only. No extra/dual cache. No attn_sink. No padded heads.

constexpr int V3_BLOCK_THREADS = 128;
constexpr int V3_N_WARPS = 4;
constexpr int V3_CAND_WINDOW = 128;
constexpr int V3_BI = V3_CAND_WINDOW;
constexpr int V3_ENTRIES_PER_WARP = V3_BI / V3_N_WARPS;     // 32
constexpr int V3_QK_N_TILES = V3_ENTRIES_PER_WARP / 8;      // 4 (m16n8 N-tiles per warp)

template <ModelType MT, int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE>
__global__ void __launch_bounds__(V3_BLOCK_THREADS, 1) sparse_mla_decode_v3_kernel(
    const bf16* __restrict__ Q,           // [num_tokens, num_heads, d_qk] bf16
    const uint8_t* __restrict__ KV_cache, // FP8 paged (MODEL1 footer layout)
    const int32_t* __restrict__ indices,  // [num_tokens, topk] int32
    bf16* __restrict__ mid_out,           // [num_tokens, num_heads, num_splits, d_v] bf16
    float* __restrict__ mid_lse,          // [num_tokens, num_heads, num_splits] f32
    const int* __restrict__ topk_length_ptr,  // [num_tokens] or null
    int num_tokens, int num_splits,
    float sm_scale, size_t stride_kv_block) {
  using KV = KVCacheTraits<MT>;
  static_assert(MT == ModelType::MODEL1, "decode-v3 currently MODEL1-only");
  constexpr int D_NOPE = KV::D_NOPE;                      // 448
  constexpr int D_ROPE_C = KV::D_ROPE;                    // 64
  constexpr int D_QK = KV::D_QK;                          // 512
  constexpr int D_V_C = KV::D_V;                          // 512
  constexpr int QUANT_TILE = KV::QUANT_TILE;              // 64
  constexpr int NUM_SCALES = KV::NUM_SCALES;              // 7
  constexpr int Q_NOPE_STRIDE = KV::Q_NOPE_STRIDE;        // 464 (D_NOPE + 16 pad)
  constexpr int KV_SMEM_STRIDE = KV::KV_SMEM_STRIDE;      // 464
  constexpr int SCALE_BYTES_PER_TOKEN = KV::SCALE_BYTES_PER_TOKEN;  // 8
  constexpr int IO_STRIDE = D_NOPE + D_ROPE_C * 2;        // 576 — per-token DATA stride
  constexpr int pbs = PAGE_BLOCK_SIZE;

  const int t_idx = blockIdx.x;
  const int h_block_idx = blockIdx.y;
  const int split_idx = blockIdx.z;
  if (t_idx >= num_tokens) return;

  const int h_start = h_block_idx * HPB;
  const int topk_len = topk_length_ptr ? __ldg(topk_length_ptr + t_idx) : TOPK;
  const int split_cand_start = split_idx * V3_CAND_WINDOW;
  if (split_cand_start >= topk_len) {
    // Mark this split as inactive via -inf LSE; merge will skip it.
    if (threadIdx.x < HPB) {
      const int h = h_start + threadIdx.x;
      const size_t lse_off =
          (size_t)t_idx * NUM_HEADS * num_splits + (size_t)h * num_splits + split_idx;
      mid_lse[lse_off] = -1e30f;
    }
    return;
  }
  const int split_cand_end = min(split_cand_start + V3_CAND_WINDOW, topk_len);

  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int gid = lane >> 2;
  const int tid = lane & 3;

  // V chunk constants for FP8 XV (v2 pattern).
  constexpr int V_CHUNK = QUANT_TILE;                              // 64
  constexpr int N_V_CHUNKS = D_NOPE / V_CHUNK;                     // 7
  constexpr int NT_PER_WARP_XV = V_CHUNK / 8 / V3_N_WARPS;         // 2
  constexpr int XV_KSTEPS = V3_BI / 32;                            // 4 (FP8 k=32)
  constexpr int W_FP8_STRIDE = V3_BI + 16;                         // 144

  // ── Dynamic smem layout (S2.5; CW=128, MODEL1, FP8 XV) ──────────
  //   sm_q_rope    HPB * D_ROPE * 2B            =   2 KB
  //   sm_q_fp8     HPB * Q_NOPE_STRIDE          = 7.25 KB
  //   sm_q_sc      HPB * NUM_SCALES * 4         = 0.44 KB
  //   sm_kv_fp8    V3_BI * KV_SMEM_STRIDE       =  58 KB
  //   sm_kv_sc     V3_BI * SCALE_BYTES_PER_TOKEN =  1 KB
  //   sm_kv_rope   V3_BI * D_ROPE * 2B          =  16 KB
  //   sm_reduce    max(HPB*NUM_SCALES, 2*N_WARPS*HPB) f32 =  512 B
  //   sm_w_head_sc N_V_CHUNKS * HPB * 4         =  448 B
  //   sm_w_fp8     HPB * W_FP8_STRIDE           = 2.25 KB (reused across 7 chunks)
  //   ----
  //   Total dyn   ~ 88 KB
  //   Static sm_p_full = HPB * V3_BI * 2B = 4 KB (used by RoPE Stage 3)
  //   Grand total  ~ 92 KB (under 100 KB SM120 carveout, 1 block/SM).
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
  // off += (size_t)HPB * W_FP8_STRIDE;  // (terminator; reused across chunks)

  // Aliases over sm_reduce. Used in two non-overlapping phases:
  //   Phase 0 (Stage 0 quantize_q_to_smem): amax[HPB * NUM_SCALES = 112]
  //   Phase 1 (Stage 2.5 cross-warp softmax): warp_max[64] + warp_sum[64] = 128 floats
  float* sm_warp_max = sm_reduce;
  float* sm_warp_sum = sm_reduce + V3_N_WARPS * HPB;

  // ── Stage 0: Q quantization (bf16 → FP8 e4m3 + UE8M0 scales) ────
  // Produces sm_q_fp8 (FP8 NoPE, padded to Q_NOPE_STRIDE), sm_q_sc (FP32 power-
  // of-2 scales, NUM_SCALES per head), sm_q_rope (bf16 RoPE).
  const bf16* q_base = Q + (size_t)t_idx * NUM_HEADS * D_QK + (size_t)h_start * D_QK;
  quantize_q_to_smem<MT, V3_BLOCK_THREADS>(sm_q_fp8, sm_q_sc, sm_q_rope, q_base, sm_reduce);
  // quantize_q_to_smem ends with a bar_sync_t<2, V3_BLOCK_THREADS> after the
  // FP8 store loop, so all 128 threads see the finished Q buffers.

  // ── Stage 1: gather KV bytes raw (no dequant). ──────────────────
  // Each warp handles V3_ENTRIES_PER_WARP=16 candidates. Layout per cand in
  // gmem (MODEL1 footer):
  //   [0:448)      FP8 NoPE
  //   [448:576)    bf16 RoPE (64 elements × 2B)
  //   footer[0:8)  UE8M0 scale bytes (first 7 used)
  //
  // Lanes 0..27 cooperatively copy the 28 16B NoPE chunks (= 448B).
  // Lanes 0..7 cooperatively copy the 8 16B RoPE chunks (= 128B).
  // Lane 28 copies the 8B scale footer.
  // Invalid cands (cand_pos >= split_cand_end OR idx < 0) get explicit zero
  // stores after the cp.async lands, so the FP8 MMA sees exact-zero V.
  const int32_t* idx_base = indices + (size_t)t_idx * TOPK;
  {
    const int warp_first_entry = warp_id * V3_ENTRIES_PER_WARP;
#pragma unroll
    for (int eo = 0; eo < V3_ENTRIES_PER_WARP; eo++) {
      const int entry_idx = warp_first_entry + eo;
      const int cand_pos = split_cand_start + entry_idx;
      const bool is_valid_cand = (cand_pos < split_cand_end);
      int idx_raw = is_valid_cand ? idx_base[cand_pos] : -1;
      const bool valid_idx = (idx_raw >= 0);
      const int idx = valid_idx ? idx_raw : 0;
      const int block_idx_g = idx / pbs;
      const int local_idx_g = idx - block_idx_g * pbs;

      const uint8_t* data_base =
          KV_cache + (size_t)block_idx_g * stride_kv_block + (size_t)local_idx_g * IO_STRIDE;
      const uint8_t* scale_base = KV_cache + (size_t)block_idx_g * stride_kv_block +
                                  (size_t)pbs * IO_STRIDE +
                                  (size_t)local_idx_g * SCALE_BYTES_PER_TOKEN;

      // 28 lanes × 16B = 448B NoPE FP8.
      if (lane < 28) {
        cp_async_16B(sm_kv_fp8 + (size_t)entry_idx * KV_SMEM_STRIDE + lane * 16,
                     data_base + lane * 16);
      }
      // 8 lanes × 16B (= 8 bf16 each) = 128B RoPE bf16.
      if (lane < 8) {
        cp_async_16B(sm_kv_rope + (size_t)entry_idx * D_ROPE_C + lane * 8,
                     data_base + D_NOPE + lane * 16);
      }
      // 8B scale footer.
      if (lane == 28) {
        *reinterpret_cast<uint64_t*>(sm_kv_sc + (size_t)entry_idx * SCALE_BYTES_PER_TOKEN) =
            *reinterpret_cast<const uint64_t*>(scale_base);
      }
    }
  }
  cp_async_commit();
  cp_async_wait_group<0>();
  __syncthreads();

  // Zero-stomp invalid candidate rows so the FP8 MMA can't pick up NaN bytes
  // (mirrors v2's "zero kv_smem rows for invalid (-1) entries" guard).
  // Stage 1 already populated sm_kv_fp8/_rope from idx=0 for those rows; we
  // overwrite the FP8 bytes with zeros (FP8 +0.0 = byte 0). RoPE bf16 +0.0
  // is also bytewise zero. Scales don't matter for an all-zero row.
  {
    const int warp_first_entry = warp_id * V3_ENTRIES_PER_WARP;
#pragma unroll
    for (int eo = 0; eo < V3_ENTRIES_PER_WARP; eo++) {
      const int entry_idx = warp_first_entry + eo;
      const int cand_pos = split_cand_start + entry_idx;
      const bool is_valid = (cand_pos < split_cand_end) && (idx_base[cand_pos] >= 0);
      if (!is_valid) {
        // 32 lanes zero the FP8 row (448 / 32 = 14 bytes per lane).
        uint8_t* fp8_row = sm_kv_fp8 + (size_t)entry_idx * KV_SMEM_STRIDE;
#pragma unroll
        for (int b = 0; b < (D_NOPE + 31) / 32; b++) {
          int off_b = lane * ((D_NOPE + 31) / 32) + b;
          if (off_b < D_NOPE) fp8_row[off_b] = 0;
        }
        // 8 lanes zero the RoPE row (64 bf16 / 8 = 8 bf16 each).
        if (lane < 8) {
          uint4 zero = make_uint4(0, 0, 0, 0);
          *reinterpret_cast<uint4*>(sm_kv_rope + (size_t)entry_idx * D_ROPE_C + lane * 8) = zero;
        }
      }
    }
  }
  __syncthreads();

  // ── Stage 2: QK = Q @ K^T, sm_scale, then warp-level softmax ────
  // Per warp processes V3_ENTRIES_PER_WARP candidates × HPB=16 heads.
  // Output per warp: V3_QK_N_TILES N-tiles of 8 cands each, each tile
  // produces a 16×8 fragment distributed across 32 lanes as 4 floats / lane.
  float qk[V3_QK_N_TILES][4] = {0};

  // NoPE FP8 MMA: 14 k-iters at k=32, organized as 7 scale tiles × 2 k-iters
  // each (QUANT_TILE=64 = 2 × k=32). A operand from sm_q_fp8 (16 heads × 32
  // k-bytes), B operand from sm_kv_fp8 (8 cands × 32 k-bytes per N-tile).
  // Block-scaled MMA absorbs the UE8M0 scales (one per scale-tile per row).
  {
    const int warp_first_cand = warp_id * V3_ENTRIES_PER_WARP;
#pragma unroll
    for (int blk = 0; blk < NUM_SCALES; blk++) {
      // sfa: Q-side UE8M0 scale (NUM_SCALES per head, one per k-tile of 64).
      // ldmatrix_load_A_fp8 distributes lanes such that lane's (gid, tid)
      // maps to head rows. The MMA m16n8k32 block_scale convention takes
      // ONE scale byte per A-row pair; we feed the head-row's scale.
      // The block_scale instruction picks the scale from a 2-byte slot
      // depending on the lane: lanes 0..3 → head gid, lanes 16..19 → head
      // gid+8 (per ldmatrix.x4 mapping). Each lane reads its own head's
      // scale via gid + (lane & 1) * 8 (matches v2's pattern).
      uint8_t sfa = fp32_to_ue8m0(sm_q_sc[(gid + (lane & 1) * 8) * NUM_SCALES + blk]);

#pragma unroll
      for (int ks = 0; ks < QUANT_TILE / 32; ks++) {
        const int ko = blk * QUANT_TILE + ks * 32;
        uint32_t a0, a1, a2, a3;
        ldmatrix_load_A_fp8(a0, a1, a2, a3, sm_q_fp8 + ko, Q_NOPE_STRIDE, lane);

#pragma unroll
        for (int nt = 0; nt < V3_QK_N_TILES; nt++) {
          const int cand_row_base = warp_first_cand + nt * 8;
          // sfb: K-side UE8M0 scale for this N-tile's 8 cands. ldmatrix_load_B_fp8
          // maps lane's row index to cand within tile via (lane & 7). The
          // block-scaled MMA needs one scale byte per B-row; each lane feeds
          // its own cand's scale at this k-block.
          // B-side scale lane mapping matches v2: lane provides scale for
          // cand row (cand_row_base + gid). Per the block_scale convention,
          // each cand's scale is broadcast across 4 lanes (lanes with the
          // same gid); hardware reads from the right lane for each B row.
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

  // RoPE bf16 MMA: 4 k-iters at k=16, covering D_ROPE=64 dims. A from
  // sm_q_rope, B from sm_kv_rope. No swizzle on sm_kv_rope (row-major bf16).
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

  // Mask invalid candidates (beyond split_cand_end - split_cand_start) and
  // apply sm_scale × LOG2E.
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

  // Per-head max across this warp's candidates.
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

  // ── Stage 2.5: cross-warp reduce per head ───────────────────────
  if (tid == 0) {
    sm_warp_max[warp_id * HPB + gid] = local_max[0];
    sm_warp_max[warp_id * HPB + gid + 8] = local_max[1];
    sm_warp_sum[warp_id * HPB + gid] = local_sum[0];
    sm_warp_sum[warp_id * HPB + gid + 8] = local_sum[1];
  }
  __syncthreads();

  if (threadIdx.x < HPB) {
    const int h = threadIdx.x;
    float wmax[V3_N_WARPS];
    float wsum[V3_N_WARPS];
#pragma unroll
    for (int w = 0; w < V3_N_WARPS; w++) {
      wmax[w] = sm_warp_max[w * HPB + h];
      wsum[w] = sm_warp_sum[w * HPB + h];
    }
    float gmax = -1e30f;
#pragma unroll
    for (int w = 0; w < V3_N_WARPS; w++) gmax = fmaxf(gmax, wmax[w]);
    float gsum = 0.f;
#pragma unroll
    for (int w = 0; w < V3_N_WARPS; w++) gsum += wsum[w] * exp2f(wmax[w] - gmax);
    sm_warp_max[h] = gmax;
    sm_warp_sum[h] = gsum;
  }
  __syncthreads();

  // ── Stage 2.75: globally-normalized softmax weights as bf16 ─────
  // Per-warp p is rescaled by exp2(local_max - gmax) / gsum so sm_p_full
  // holds the fully normalized softmax weights; Stage 3 XV produces the
  // final output, no further normalization needed.
  __shared__ bf16 sm_p_full[HPB][V3_BI];  // 2 KB static

  const float gmax0 = sm_warp_max[gid];
  const float gmax1 = sm_warp_max[gid + 8];
  const float gsum0 = sm_warp_sum[gid];
  const float gsum1 = sm_warp_sum[gid + 8];
  const float resc_h0 = (gsum0 > 0.f) ? (exp2f(local_max[0] - gmax0) / gsum0) : 0.f;
  const float resc_h1 = (gsum1 > 0.f) ? (exp2f(local_max[1] - gmax1) / gsum1) : 0.f;

  const int cand_col_base = warp_id * V3_ENTRIES_PER_WARP;
#pragma unroll
  for (int nt = 0; nt < V3_QK_N_TILES; nt++) {
    const int c0 = nt * 8 + tid * 2;
    const int c1 = c0 + 1;
    sm_p_full[gid][cand_col_base + c0] = __float2bfloat16(p[nt][0] * resc_h0);
    sm_p_full[gid][cand_col_base + c1] = __float2bfloat16(p[nt][1] * resc_h0);
    sm_p_full[gid + 8][cand_col_base + c0] = __float2bfloat16(p[nt][2] * resc_h1);
    sm_p_full[gid + 8][cand_col_base + c1] = __float2bfloat16(p[nt][3] * resc_h1);
  }
  __syncthreads();

  // ── Stage 3: XV (v2-pattern FP8 NoPE + bf16 RoPE) ───────────────
  //
  // NoPE (D_NOPE=448 dims = 7 V chunks of V_CHUNK=64):
  //   Per chunk vc: amax(|p * vsc|) per (vc, head) → Q-scale → quantize
  //   W = p*vsc/Q-scale → FP8 in sm_w_fp8; FP8 m16n8k32 MMA reads
  //   sm_kv_fp8 straight into the B operand via d2_load_b_fp8.
  //   Dim partitioning (matches v2 SmemLayout): warp w owns dims
  //   [vc*V_CHUNK + w*16, vc*V_CHUNK + w*16 + 16) per chunk (16 dims, 2
  //   N-tiles of 8 each). 7 chunks × 16 = 112 NoPE dims per warp.
  //
  // RoPE (D_ROPE=64 dims = 16 per warp, contiguous): bf16 m16n8k16 MMA
  //   reads sm_p_full as A operand and sm_kv_rope as B operand.
  //
  // sm_w_fp8 buffer is reused across the 7 chunks via barrier syncs.
  const size_t mid_o_base_ll = ((size_t)t_idx * NUM_HEADS + h_start) * (size_t)num_splits * D_V_C +
                               (size_t)split_idx * D_V_C;
  const size_t mid_lse_base_ll =
      (size_t)t_idx * NUM_HEADS * num_splits + (size_t)h_start * num_splits;

  // Phase 1: atomicMax-based amax computation per (vc, head).
  // Initialize sm_w_head_sc to 0.
  for (int i = threadIdx.x; i < N_V_CHUNKS * HPB; i += V3_BLOCK_THREADS) {
    sm_w_head_sc[i] = 0.f;
  }
  __syncthreads();

  // Each thread holds p[V3_QK_N_TILES][4] = its share of the global
  // softmax weights, where p[nt][k] = (head h, cand c) maps to:
  //   p[nt][0]: head gid,    cand warp_first_cand + nt*8 + tid*2
  //   p[nt][1]: head gid,    cand warp_first_cand + nt*8 + tid*2 + 1
  //   p[nt][2]: head gid+8,  cand warp_first_cand + nt*8 + tid*2
  //   p[nt][3]: head gid+8,  cand warp_first_cand + nt*8 + tid*2 + 1
  // (p was rescaled in Stage 2.75 by exp2(local_max - gmax) / gsum.)
  {
    const int warp_first_cand_xv = warp_id * V3_ENTRIES_PER_WARP;
#pragma unroll
    for (int nt = 0; nt < V3_QK_N_TILES; nt++) {
      const int cand_e0 = warp_first_cand_xv + nt * 8 + tid * 2;
      const int cand_e1 = cand_e0 + 1;
      const float w0 = p[nt][0] * resc_h0;  // head gid, cand e0
      const float w1 = p[nt][1] * resc_h0;  // head gid, cand e1
      const float w2 = p[nt][2] * resc_h1;  // head gid+8, cand e0
      const float w3 = p[nt][3] * resc_h1;  // head gid+8, cand e1
#pragma unroll
      for (int vc = 0; vc < N_V_CHUNKS; vc++) {
        const float vsc0 = ue8m0_to_fp32(
            sm_kv_sc[(size_t)cand_e0 * SCALE_BYTES_PER_TOKEN + vc]);
        const float vsc1 = ue8m0_to_fp32(
            sm_kv_sc[(size_t)cand_e1 * SCALE_BYTES_PER_TOKEN + vc]);
        const float ws00 = w0 * vsc0;
        const float ws01 = w1 * vsc1;
        const float ws10 = w2 * vsc0;
        const float ws11 = w3 * vsc1;
        atomicMax(reinterpret_cast<int*>(&sm_w_head_sc[vc * HPB + gid]),
                  __float_as_int(fmaxf(fabsf(ws00), fabsf(ws01))));
        atomicMax(reinterpret_cast<int*>(&sm_w_head_sc[vc * HPB + gid + 8]),
                  __float_as_int(fmaxf(fabsf(ws10), fabsf(ws11))));
      }
    }
  }
  __syncthreads();

  // Phase 2: convert amax → Q-scale = amax / FP8_MAX
  for (int i = threadIdx.x; i < N_V_CHUNKS * HPB; i += V3_BLOCK_THREADS) {
    sm_w_head_sc[i] = fmaxf(sm_w_head_sc[i], 1e-10f) / FP8_MAX;
  }
  __syncthreads();

  // NoPE accumulator: per warp, N_V_CHUNKS × NT_PER_WARP_XV × 4 floats.
  // Each (vc, nt) covers 8 dims; warp's total NoPE dims = 7 × 2 × 8 = 112.
  float acc_nope[N_V_CHUNKS][NT_PER_WARP_XV][4] = {0};

  // Per-chunk FP8 quant + MMA. sm_w_fp8 is reused across chunks via the
  // barrier at the top of each chunk iter (after the prior chunk's MMAs
  // have read from it).
#pragma unroll
  for (int vc = 0; vc < N_V_CHUNKS; vc++) {
    if (vc > 0) __syncthreads();  // prior chunk's MMAs done reading sm_w_fp8

    // Phase 3: each thread quantizes its 4 ws values per nt → FP8 → sm_w_fp8.
    {
      const int warp_first_cand_xv = warp_id * V3_ENTRIES_PER_WARP;
      const float si0 = 1.f / sm_w_head_sc[vc * HPB + gid];
      const float si1 = 1.f / sm_w_head_sc[vc * HPB + gid + 8];
#pragma unroll
      for (int nt = 0; nt < V3_QK_N_TILES; nt++) {
        const int cand_e0 = warp_first_cand_xv + nt * 8 + tid * 2;
        const int cand_e1 = cand_e0 + 1;
        const float w0 = p[nt][0] * resc_h0;
        const float w1 = p[nt][1] * resc_h0;
        const float w2 = p[nt][2] * resc_h1;
        const float w3 = p[nt][3] * resc_h1;
        const float vsc0 = ue8m0_to_fp32(
            sm_kv_sc[(size_t)cand_e0 * SCALE_BYTES_PER_TOKEN + vc]);
        const float vsc1 = ue8m0_to_fp32(
            sm_kv_sc[(size_t)cand_e1 * SCALE_BYTES_PER_TOKEN + vc]);
        const float ws00 = w0 * vsc0;
        const float ws01 = w1 * vsc1;
        const float ws10 = w2 * vsc0;
        const float ws11 = w3 * vsc1;
        __nv_fp8_e4m3 f00(fmaxf(FP8_MIN, fminf(FP8_MAX, ws00 * si0)));
        __nv_fp8_e4m3 f01(fmaxf(FP8_MIN, fminf(FP8_MAX, ws01 * si0)));
        __nv_fp8_e4m3 f10(fmaxf(FP8_MIN, fminf(FP8_MAX, ws10 * si1)));
        __nv_fp8_e4m3 f11(fmaxf(FP8_MIN, fminf(FP8_MAX, ws11 * si1)));
        sm_w_fp8[(size_t)gid * W_FP8_STRIDE + cand_e0] = f00.__x;
        sm_w_fp8[(size_t)gid * W_FP8_STRIDE + cand_e1] = f01.__x;
        sm_w_fp8[(size_t)(gid + 8) * W_FP8_STRIDE + cand_e0] = f10.__x;
        sm_w_fp8[(size_t)(gid + 8) * W_FP8_STRIDE + cand_e1] = f11.__x;
      }
    }
    __syncthreads();

    // Phase 4: FP8 m16n8k32 MMA per warp for this chunk.
    // Per warp: NT_PER_WARP_XV N-tiles × XV_KSTEPS K-iters.
    // A from sm_w_fp8 (16h × 32k), B from sm_kv_fp8 via d2_load_b_fp8.
#pragma unroll
    for (int nt = 0; nt < NT_PER_WARP_XV; nt++) {
      const int dim = vc * V_CHUNK + warp_id * (NT_PER_WARP_XV * 8) + nt * 8;
#pragma unroll
      for (int kstep = 0; kstep < XV_KSTEPS; kstep++) {
        const int ko = kstep * 32;
        uint32_t a0, a1, a2, a3, b0, b1;
        ldmatrix_load_A_fp8(a0, a1, a2, a3, sm_w_fp8 + ko, W_FP8_STRIDE, lane);
        d2_load_b_fp8<KV_SMEM_STRIDE>(b0, b1, sm_kv_fp8, kstep * 32, dim, lane);
        MmaFp8Result r =
            mma_fp8_m16n8k32(a0, a1, a2, a3, b0, b1, acc_nope[vc][nt][0],
                             acc_nope[vc][nt][1], acc_nope[vc][nt][2],
                             acc_nope[vc][nt][3]);
        acc_nope[vc][nt][0] = r.d0;
        acc_nope[vc][nt][1] = r.d1;
        acc_nope[vc][nt][2] = r.d2;
        acc_nope[vc][nt][3] = r.d3;
      }
    }
  }

  // Apply per-(vc, head) Q-scale to NoPE acc, then write to mid_out.
#pragma unroll
  for (int vc = 0; vc < N_V_CHUNKS; vc++) {
    const float sc0 = sm_w_head_sc[vc * HPB + gid];
    const float sc1 = sm_w_head_sc[vc * HPB + gid + 8];
#pragma unroll
    for (int nt = 0; nt < NT_PER_WARP_XV; nt++) {
      const int d0 = vc * V_CHUNK + warp_id * (NT_PER_WARP_XV * 8) + nt * 8 + tid * 2;
      const int d1 = d0 + 1;
      mid_out[mid_o_base_ll + (size_t)gid * num_splits * D_V_C + d0] =
          __float2bfloat16(acc_nope[vc][nt][0] * sc0);
      mid_out[mid_o_base_ll + (size_t)gid * num_splits * D_V_C + d1] =
          __float2bfloat16(acc_nope[vc][nt][1] * sc0);
      mid_out[mid_o_base_ll + (size_t)(gid + 8) * num_splits * D_V_C + d0] =
          __float2bfloat16(acc_nope[vc][nt][2] * sc1);
      mid_out[mid_o_base_ll + (size_t)(gid + 8) * num_splits * D_V_C + d1] =
          __float2bfloat16(acc_nope[vc][nt][3] * sc1);
    }
  }

  // ── Stage 3 RoPE: bf16 m16n8k16 over 16 RoPE dims per warp ──────
  // Warp w owns RoPE dims [w*16, (w+1)*16) within the RoPE block of D_V.
  // Output dim in mid_out: D_NOPE + warp_id*16 + nt*8 + (gid|gid+8 lane).
  {
    constexpr int ROPE_DIMS_PER_WARP = D_ROPE_C / V3_N_WARPS;     // 16
    constexpr int ROPE_N_TILES = ROPE_DIMS_PER_WARP / 8;          // 2
    constexpr int ROPE_K_ITERS = V3_BI / 16;                      // 8 (k=16 bf16)
    const int rope_dim_base = warp_id * ROPE_DIMS_PER_WARP;

    float acc_rope[ROPE_N_TILES][4] = {0};
#pragma unroll
    for (int ks = 0; ks < ROPE_K_ITERS; ks++) {
      // A operand: bf16 W = sm_p_full[16h, 16k=cands ks*16..ks*16+15]
      uint32_t a0, a1, a2, a3;
      ldmatrix_load_A_bf16(a0, a1, a2, a3,
                           reinterpret_cast<const bf16*>(&sm_p_full[0][ks * 16]),
                           V3_BI, lane);
#pragma unroll
      for (int nt = 0; nt < ROPE_N_TILES; nt++) {
        const int n_col = rope_dim_base + nt * 8;
        // B operand: bf16 from sm_kv_rope, layout [V3_BI cands, D_ROPE dims].
        // ldmatrix.x2.trans reads 8 rows × 8 cols, transposing per-tile so
        // each thread sees K-rows in its register slots for one N-col.
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

    // Write RoPE acc to mid_out at offset D_NOPE + rope_dim_base + nt*8 + tid*2.
#pragma unroll
    for (int nt = 0; nt < ROPE_N_TILES; nt++) {
      const int d0 = D_NOPE + rope_dim_base + nt * 8 + tid * 2;
      const int d1 = d0 + 1;
      mid_out[mid_o_base_ll + (size_t)gid * num_splits * D_V_C + d0] =
          __float2bfloat16(acc_rope[nt][0]);
      mid_out[mid_o_base_ll + (size_t)gid * num_splits * D_V_C + d1] =
          __float2bfloat16(acc_rope[nt][1]);
      mid_out[mid_o_base_ll + (size_t)(gid + 8) * num_splits * D_V_C + d0] =
          __float2bfloat16(acc_rope[nt][2]);
      mid_out[mid_o_base_ll + (size_t)(gid + 8) * num_splits * D_V_C + d1] =
          __float2bfloat16(acc_rope[nt][3]);
    }
  }

  // Phase D: write LSE per head.
  if (threadIdx.x < HPB) {
    const int h = threadIdx.x;
    const float g_max = sm_warp_max[h];
    const float g_sum = sm_warp_sum[h];
    const float lse = (g_sum > 0.f) ? (log2f(g_sum) + g_max) : -1e30f;
    mid_lse[mid_lse_base_ll + (size_t)h * num_splits + split_idx] = lse;
  }
}

// Merge kernel: collapse splits into final output and LSE.
// (Unchanged from v3_backup — operates on mid_out/mid_lse, not smem layout.)
template <int NUM_HEADS, int D_V_VAL>
__global__ void __launch_bounds__(HPB * 32, 4) sparse_mla_decode_v3_merge_kernel(
    const bf16* __restrict__ mid_out,   // [num_tokens, num_heads, num_splits, D_V] bf16
    const float* __restrict__ mid_lse,  // [num_tokens, num_heads, num_splits] f32
    bf16* __restrict__ output,          // [num_tokens, num_heads, D_V] bf16
    float* __restrict__ out_lse,        // [num_tokens, num_heads] f32, nullable
    int num_tokens, int num_splits) {
  const int t_idx = blockIdx.x;
  const int h_block_idx = blockIdx.y;
  if (t_idx >= num_tokens) return;
  const int h = h_block_idx * HPB + (threadIdx.x / 32);
  if (h >= NUM_HEADS) return;
  const int lane = threadIdx.x & 31;

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

  constexpr int DIMS_PER_LANE = D_V_VAL / 32;
  float acc[DIMS_PER_LANE];
#pragma unroll
  for (int d = 0; d < DIMS_PER_LANE; d++) acc[d] = 0.f;

  for (int sp = 0; sp < num_splits; sp++) {
    float lse_sp = lse_ptr[sp];
    if (lse_sp <= -1e29f) continue;
    float weight = exp2f(lse_sp - global_max);
#pragma unroll
    for (int d = 0; d < DIMS_PER_LANE; d++) {
      float v = __bfloat162float(mid_base[(size_t)sp * D_V_VAL + lane * DIMS_PER_LANE + d]);
      acc[d] += weight * v;
    }
  }

#pragma unroll
  for (int d = 0; d < DIMS_PER_LANE; d++) {
    out_ptr[lane * DIMS_PER_LANE + d] = __float2bfloat16(acc[d] * inv_global_sum);
  }
  if (out_lse != nullptr && lane == 0) {
    out_lse[(size_t)t_idx * NUM_HEADS + h] = global_lse;
  }
}

}  // namespace flashinfer::sparse_mla_sm120
