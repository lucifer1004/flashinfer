// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// (license header — full text mirrors the other files in this directory)

#pragma once

#include "arch/cp_async.cuh"
#include "arch/ldmatrix_sm120.cuh"
#include "arch/mma_sm120.cuh"
#include "common/online_softmax.cuh"
#include "model/kv_cache_traits.cuh"
#include "model/scale_convert.cuh"

namespace flashinfer::sparse_mla_sm120 {

// Decode-v3: small-block sparse-MLA decode optimised for the high-topk
// contested-shape regime.
//
// Grid: (num_tokens, num_head_blocks, num_splits)
//   num_head_blocks = num_heads / HPB
//   num_splits     = topk / CAND_WINDOW (= 64)
//
// Block: 128 threads (4 warps), uniform — no warp specialisation.
//
// Each block processes one (token, head_block) cell over one
// CAND_WINDOW-sized window of candidates: gathers FP8 KV → dequants in
// shared memory → bf16 MMA QK + softmax + XV → writes a partial bf16
// output and an f32 LSE to a per-(token, head, split) mid buffer.
//
// A separate merge kernel (decode_v3_merge_kernel) collapses splits with
// LSE-weighted reduction.
//
// First cut: MODEL1 only. No extra/dual cache. No attn_sink. No padded
// heads. These can fold in later without changing the kernel structure.

constexpr int V3_BLOCK_THREADS = 128;
constexpr int V3_N_WARPS = 4;
constexpr int V3_CAND_WINDOW = 64;  // candidates handled by one block
constexpr int V3_BI = V3_CAND_WINDOW;
constexpr int V3_ENTRIES_PER_WARP = V3_BI / V3_N_WARPS;  // 16

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
  constexpr int D_NOPE = KV::D_NOPE;          // 448
  constexpr int D_ROPE = KV::D_ROPE;          // 64
  constexpr int D_QK = KV::D_QK;              // 512
  constexpr int D_V = KV::D_V;                // 512
  constexpr int QUANT_TILE = KV::QUANT_TILE;  // 64
  constexpr int NUM_SCALES = KV::NUM_SCALES;  // 7
  constexpr int IO_STRIDE = D_NOPE + D_ROPE * 2;          // 576 — per-token DATA stride
  constexpr int SCALE_BYTES_PER_TOKEN = NUM_SCALES + 1;   // 8 (incl. one unused)
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
  const int n_valid_in_split = split_cand_end - split_cand_start;

  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int gid = lane >> 2;
  const int tid = lane & 3;

  extern __shared__ char smem_raw[];
  // Smem layout:
  //   q_bf16    [HPB, D_QK]                bf16  =  16 KB
  //   kv_bf16   [V3_BI, D_QK]              bf16  =  64 KB
  //   warp_max  [V3_N_WARPS * HPB]         float = small
  //   warp_sum  [V3_N_WARPS * HPB]         float = small
  bf16* sm_q = reinterpret_cast<bf16*>(smem_raw);
  bf16* sm_kv = sm_q + (size_t)HPB * D_QK;
  float* sm_warp_max = reinterpret_cast<float*>(sm_kv + (size_t)V3_BI * D_QK);
  float* sm_warp_sum = sm_warp_max + V3_N_WARPS * HPB;

  // ── Stage 0: load Q for (t_idx, h_block) ─────────────────────────
  // Q layout: [num_tokens, num_heads, D_QK] bf16. We load HPB heads
  // worth = HPB * D_QK = 16 * 512 = 8192 bf16 = 16 KB.
  {
    const bf16* q_base = Q + (size_t)t_idx * NUM_HEADS * D_QK + (size_t)h_start * D_QK;
    constexpr int Q_ELEMS = HPB * D_QK;
    // 128 threads, 8 elements per thread per chunk = 16 chunks total.
    for (int i = threadIdx.x; i < Q_ELEMS / 8; i += V3_BLOCK_THREADS) {
      uint4 v = *reinterpret_cast<const uint4*>(q_base + i * 8);
      *reinterpret_cast<uint4*>(sm_q + i * 8) = v;
    }
  }
  __syncthreads();

  // ── Stage 1: gather + dequantize CAND_WINDOW candidates ──────────
  // Per warp owns V3_ENTRIES_PER_WARP=16 entries. Within a warp, 32
  // lanes split 16 entries × (D_QK / 16 lanes-per-entry-row)...
  // simpler: each thread handles a vector chunk of one entry.
  //
  // Per-entry: read D_NOPE=448 bytes FP8 + D_ROPE*2=128 bytes bf16
  // RoPE + scale bytes. Dequantize FP8 → bf16, concatenate with RoPE
  // bf16, write D_QK=512 bf16 to sm_kv[entry_idx][:].
  //
  // FP8 dequant: per QUANT_TILE=64 FP8 values, one UE8M0 scale byte.
  // For D_NOPE=448 = 7 tiles. Scales live in the page-block FOOTER
  // (offset pbs * IO_STRIDE within the block).
  const int32_t* idx_base = indices + (size_t)t_idx * TOPK;
  {
    // Each warp handles V3_ENTRIES_PER_WARP=16 entries.
    // Lanes within a warp split a single entry's payload.
    // For simplicity: 32 lanes split 16 entries → 2 lanes per entry.
    // Each lane handles half the FP8 NoPE (224 bytes) + half rope.
    const int warp_first_entry = warp_id * V3_ENTRIES_PER_WARP;
#pragma unroll 1
    for (int eo = 0; eo < V3_ENTRIES_PER_WARP; eo++) {
      const int entry_idx = warp_first_entry + eo;
      const int cand_pos = split_cand_start + entry_idx;
      const bool is_valid_cand = (cand_pos < split_cand_end);
      int idx = is_valid_cand ? idx_base[cand_pos] : 0;
      idx = (idx >= 0) ? idx : 0;
      const int block_idx_g = idx / pbs;
      const int local_idx_g = idx % pbs;

      const uint8_t* data_base =
          KV_cache + (size_t)block_idx_g * stride_kv_block + (size_t)local_idx_g * IO_STRIDE;
      const uint8_t* scale_base = KV_cache + (size_t)block_idx_g * stride_kv_block +
                                  (size_t)pbs * IO_STRIDE + (size_t)local_idx_g * SCALE_BYTES_PER_TOKEN;

      bf16* kv_dst = sm_kv + (size_t)entry_idx * D_QK;

      // Load the 8 scale bytes (one per QUANT_TILE; only first 7 used).
      // Cooperative load via the first lane; broadcast via shuffle.
      uint64_t scale_packed;
      if (lane == 0) {
        scale_packed = *reinterpret_cast<const uint64_t*>(scale_base);
      }
      scale_packed = __shfl_sync(0xffffffff, scale_packed, 0);
      // Dequant the FP8 NoPE half: each thread processes 14 bytes
      // (≈ 448/32, with 16 leftover for the last 16 threads).
      // Simpler: 32 lanes × 16 bytes each = 512 covers D_NOPE+padding.
      // But D_NOPE=448 = 32 × 14. Let's do 16 lanes × 32 bytes each:
      //   16 lanes cover 16 × 32 = 512 (> 448, last 64 discarded).
      // Skipping; use a simpler 8B-per-lane formulation. We need
      // each lane to load 16 FP8 bytes, dequant to 16 bf16 (32 bytes
      // to write to smem).
      constexpr int FP8_BYTES_PER_LANE = (D_NOPE + 31) / 32;  // 14 (round-up)
      // Use the simpler "lane handles its slice" pattern: each of the
      // 32 lanes writes a 16-byte FP8 chunk, but lanes 28..31 handle
      // the partial tail.
      // Actually, do it the straightforward way: lane i loads bytes
      // [i*14, (i+1)*14) of the NoPE data and dequants.
      const int lane_byte_start = lane * FP8_BYTES_PER_LANE;
      const int lane_byte_end = min(lane_byte_start + FP8_BYTES_PER_LANE, D_NOPE);
      for (int b = lane_byte_start; b < lane_byte_end; b++) {
        const int qt = b / QUANT_TILE;  // which scale tile
        const uint8_t scale_byte = static_cast<uint8_t>((scale_packed >> (qt * 8)) & 0xFF);
        const float scale = ue8m0_to_fp32(scale_byte);
        const __nv_fp8_e4m3 raw{
            static_cast<__nv_fp8_storage_t>(data_base[b])};
        const float v = static_cast<float>(raw);
        kv_dst[b] = __float2bfloat16(is_valid_cand ? (v * scale) : 0.0f);
      }
      // RoPE half: already bf16 in gmem, 128 bytes = 64 bf16, copy verbatim.
      // 32 lanes × 2 bf16 each = 64 bf16.
      const bf16* rope_src = reinterpret_cast<const bf16*>(data_base + D_NOPE);
      bf16* rope_dst = kv_dst + D_NOPE;
      if (lane * 2 < D_ROPE) {
        rope_dst[lane * 2] = is_valid_cand ? rope_src[lane * 2] : __float2bfloat16(0.0f);
        rope_dst[lane * 2 + 1] = is_valid_cand ? rope_src[lane * 2 + 1] : __float2bfloat16(0.0f);
      }
    }
  }
  __syncthreads();

  // ── Stage 2: QK = Q @ K^T, sm_scale, then warp-level softmax ────
  // Per warp processes V3_ENTRIES_PER_WARP=16 candidates × HPB=16 heads.
  // bf16 MMA m16n8k16: A is 16m × 16k, B is 16k × 8n, C is 16m × 8n.
  // For 16 candidates per warp: 2 N-tiles of 8 each.
  // For D_QK=512 inner dim: 32 K-iter (k-step 16).
  //
  // Output per warp: 16 heads × 16 candidates = 256 floats distributed
  // across 32 lanes = 8 floats per thread (4 per N-tile).
  float qk[2][4] = {0};  // [N-tile][m16n8 fragment]
  {
    const bf16* sm_kv_warp = sm_kv + (size_t)warp_id * V3_ENTRIES_PER_WARP * D_QK;
    // Per-thread B-operand for bf16 mma_m16n8k16 holds 4 bf16 values arranged
    // as K-rows {tid*2, tid*2+1, tid*2+8, tid*2+9} for the per-thread N-col.
    // Mirrors xv_rope_mma.cuh's manual scalar-load pattern; ldmatrix.x2.trans
    // produces a different layout we don't want.
#pragma unroll
    for (int ks = 0; ks < D_QK / 16; ks++) {
      uint32_t a0, a1, a2, a3;
      ldmatrix_load_A_bf16(a0, a1, a2, a3, sm_q + ks * 16, D_QK, lane);
#pragma unroll
      for (int nt = 0; nt < 2; nt++) {
        const int cand_idx = nt * 8 + gid;  // N-col owned by this thread
        const bf16* k_row_ptr = sm_kv_warp + (size_t)cand_idx * D_QK + ks * 16;
        uint16_t v0 =
            *reinterpret_cast<const uint16_t*>(k_row_ptr + tid * 2);
        uint16_t v1 =
            *reinterpret_cast<const uint16_t*>(k_row_ptr + tid * 2 + 1);
        uint16_t v8 =
            *reinterpret_cast<const uint16_t*>(k_row_ptr + tid * 2 + 8);
        uint16_t v9 =
            *reinterpret_cast<const uint16_t*>(k_row_ptr + tid * 2 + 9);
        uint32_t b0 = static_cast<uint32_t>(v0) | (static_cast<uint32_t>(v1) << 16);
        uint32_t b1 = static_cast<uint32_t>(v8) | (static_cast<uint32_t>(v9) << 16);
        MmaBf16Result r =
            mma_bf16_m16n8k16(a0, a1, a2, a3, b0, b1, qk[nt][0], qk[nt][1], qk[nt][2], qk[nt][3]);
        qk[nt][0] = r.d0;
        qk[nt][1] = r.d1;
        qk[nt][2] = r.d2;
        qk[nt][3] = r.d3;
      }
    }
  }
  // Mask invalid candidates (beyond split_cand_end - split_cand_start).
  // Lane (gid, tid) maps to MMA output coord: each thread holds 4 floats
  // for one N-tile: heads (gid, gid+8) × candidates (tid*2, tid*2+1).
  //
  // For N-tile nt, the candidate indices within the block are:
  //   warp_id * V3_ENTRIES_PER_WARP + nt * 8 + tid * 2 + {0, 1}
  // Mask those whose global cand_pos >= split_cand_end.
  const int warp_first_cand = warp_id * V3_ENTRIES_PER_WARP;
#pragma unroll
  for (int nt = 0; nt < 2; nt++) {
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

  // Per-head max across this warp's candidates: each thread holds
  // values for 2 heads (gid, gid+8) × 16 candidates. Reduce within
  // the warp.
  float local_max[2] = {-1e30f, -1e30f};
#pragma unroll
  for (int nt = 0; nt < 2; nt++) {
    local_max[0] = fmaxf(local_max[0], fmaxf(qk[nt][0], qk[nt][1]));
    local_max[1] = fmaxf(local_max[1], fmaxf(qk[nt][2], qk[nt][3]));
  }
  // Reduce across tid for each gid (4 lanes per gid).
#pragma unroll
  for (int s = 2; s >= 1; s >>= 1) {
    local_max[0] = fmaxf(local_max[0], __shfl_xor_sync(0xffffffff, local_max[0], s));
    local_max[1] = fmaxf(local_max[1], __shfl_xor_sync(0xffffffff, local_max[1], s));
  }
  // Now lanes within the same gid hold the same max for their 2 heads.

  float local_sum[2] = {0.f, 0.f};
  float p[2][4];
#pragma unroll
  for (int nt = 0; nt < 2; nt++) {
    p[nt][0] = exp2f(qk[nt][0] - local_max[0]);
    p[nt][1] = exp2f(qk[nt][1] - local_max[0]);
    p[nt][2] = exp2f(qk[nt][2] - local_max[1]);
    p[nt][3] = exp2f(qk[nt][3] - local_max[1]);
    local_sum[0] += p[nt][0] + p[nt][1];
    local_sum[1] += p[nt][2] + p[nt][3];
  }
  // Reduce sum across tid.
#pragma unroll
  for (int s = 2; s >= 1; s >>= 1) {
    local_sum[0] += __shfl_xor_sync(0xffffffff, local_sum[0], s);
    local_sum[1] += __shfl_xor_sync(0xffffffff, local_sum[1], s);
  }

  // ── Stage 3: XV = P @ V (V is the same dequantized KV in sm_kv) ──
  // For each head, sum_c (p[c] * V[c, :]) over 16 candidates.
  // Use bf16 MMA: A = P (16 heads × 16 candidates), B = V (16 candidates
  // × 8 dim per N-tile). C = output (16 heads × 8 dim).
  //
  // But P is in registers as float; we'd need to convert to bf16 and
  // pack as A-matrix for MMA. Simpler first pass: do a manual loop
  // (no MMA for XV). This is slower but easier to verify.
  //
  // Output: HPB=16 heads × D_V=512 dim per warp's 16-candidate slice.
  // Each thread holds the partial output for 2 heads × some dim slice.
  //
  // Simpler still: every thread accumulates the FULL output for its
  // 2 heads (gid, gid+8) across DIM_PER_THREAD slice. Then we reduce
  // across the warp at the end.
  //
  // For HPB=16, D_V=512: per warp we need 16 × 512 floats = 8192 floats.
  // Per thread: 8192 / 32 = 256 floats. Too many to keep in registers.
  //
  // Better: distribute D_V across lanes. With 32 lanes covering 512
  // dim: 16 dim per lane. Per thread: 2 heads × 16 dim = 32 floats.
  // OK. Each thread accumulates a (2 heads × 16 dim) slice.

  constexpr int DIM_PER_LANE = D_V / 32;  // 16

  // ── Stage 2.5: cross-warp reduce of (local_max, local_sum) per head ─
  // Stash per-warp local max/sum into shared memory (one writer per head
  // per warp: lanes with tid==0 cover gid ∈ [0..7], writing both gid and
  // gid+8 slots).
  if (tid == 0) {
    sm_warp_max[warp_id * HPB + gid] = local_max[0];
    sm_warp_max[warp_id * HPB + gid + 8] = local_max[1];
    sm_warp_sum[warp_id * HPB + gid] = local_sum[0];
    sm_warp_sum[warp_id * HPB + gid + 8] = local_sum[1];
  }
  __syncthreads();

  // First HPB threads compute global max/sum per head and overwrite the
  // [0..HPB) slots of sm_warp_max/sm_warp_sum with the global values.
  if (threadIdx.x < HPB) {
    const int h = threadIdx.x;
    float gmax = -1e30f;
#pragma unroll
    for (int w = 0; w < V3_N_WARPS; w++)
      gmax = fmaxf(gmax, sm_warp_max[w * HPB + h]);
    float gsum = 0.f;
#pragma unroll
    for (int w = 0; w < V3_N_WARPS; w++)
      gsum += sm_warp_sum[w * HPB + h] * exp2f(sm_warp_max[w * HPB + h] - gmax);
    sm_warp_max[h] = gmax;
    sm_warp_sum[h] = gsum;
  }
  __syncthreads();

  // Stage 3 setup: stash per-thread p values into shared memory so the
  // outer loop over heads (below) can read sm_p_storage[warp_id][h][c]
  // for any h and c.
  //
  // Layout: sm_p_storage[warp][HPB heads][V3_ENTRIES_PER_WARP candidates].
  // 4 × 16 × 16 × 4 = 4 KB static smem.
  __shared__ float sm_p_storage[V3_N_WARPS][HPB][V3_ENTRIES_PER_WARP];
#pragma unroll
  for (int nt = 0; nt < 2; nt++) {
    const int c0 = nt * 8 + tid * 2;
    const int c1 = c0 + 1;
    sm_p_storage[warp_id][gid][c0] = p[nt][0];
    sm_p_storage[warp_id][gid][c1] = p[nt][1];
    sm_p_storage[warp_id][gid + 8][c0] = p[nt][2];
    sm_p_storage[warp_id][gid + 8][c1] = p[nt][3];
  }
  __syncwarp();

  // Cross-warp-merged g_max/g_sum already in sm_warp_max[0..HPB) and
  // sm_warp_sum[0..HPB). Per-warp local_max also lives in sm_warp_max
  // at slot [warp_id*HPB + h] for h < HPB — but the global-reduce step
  // above overwrote the first HPB slots. We need the per-warp local_max
  // for ALL heads to compute the rescale factor below, so we save it
  // first BEFORE the global reduce in shared.
  //
  // The current code path has already done the reduction; we cannot
  // recover the per-warp local_max without re-storing it. So we treat
  // this as a known limitation: re-write sm_warp_max to hold per-warp
  // local_max for the rescale step, then recompute g_max/g_sum where
  // needed in the head loop.
  //
  // Simpler restructure: re-stash per-warp local_max into a separate
  // scratch region of sm_warp_max[HPB .. V3_N_WARPS*HPB).
  // Actually that region is still intact — the global reduce only
  // overwrites [0, HPB). Use it.

  // ── Stage 3: XV per head ────────────────────────────────────────────
  // Outer loop over the HPB=16 heads owned by this block. For each head:
  //   - Per-warp: compute warp_acc[16] = sum_c p[h,c] * V[c, lane*16+d]
  //     across this warp's 16 candidates.
  //   - Rescale by exp2(local_max_w_h - g_max_h) / g_sum_h.
  //   - Atomic-add across warps into a 2 KB per-head buffer in sm_kv
  //     (reused after the XV pass).
  //   - Write the buffer to mid_out for this (token, head, split).
  //
  // Per-lane register state: warp_acc[16] = 64 bytes. Trivial.
  // The cross-warp merge is via shared memory atomic-add.
  float* sm_head_buf = reinterpret_cast<float*>(sm_kv);  // reuse 64 KB
  const bf16* sm_kv_warp_x = sm_kv + (size_t)warp_id * V3_ENTRIES_PER_WARP * D_QK;
  const size_t mid_o_base_ll = ((size_t)t_idx * NUM_HEADS + h_start) * (size_t)num_splits * D_V +
                               (size_t)split_idx * D_V;
  const size_t mid_lse_base_ll =
      (size_t)t_idx * NUM_HEADS * num_splits + (size_t)h_start * num_splits;

#pragma unroll 1
  for (int h_idx = 0; h_idx < HPB; h_idx++) {
    // Per-warp local max for head h_idx. The global reduce overwrote
    // sm_warp_max[0..HPB), so we need to read from per-warp slots
    // [HPB, V3_N_WARPS*HPB). Wait — we ONLY overwrote [0..HPB). The
    // remaining slots [HPB .. V3_N_WARPS*HPB) still hold warps 1..3's
    // local max. Warp 0's local max for h_idx is now g_max — lost.
    //
    // To handle this cleanly, recompute local_max for h_idx by reading
    // from this warp's per-thread `local_max` register. But that's
    // per-thread state for h_idx=gid (when h_idx<8) or gid+8 (>=8).
    //
    // Simplest fix: have warp 0 not participate in the global reduce
    // overwrite. Or pre-stash per-warp local_max into a fresh smem
    // region before the global reduce.
    //
    // We do the latter: read from sm_kv_warp's footer area. Actually,
    // just use the warp's per-thread local_max via shuffle:
    //   - Lane (gid', tid') in warp w has local_max[0] for head gid'
    //     and local_max[1] for head gid'+8.
    //   - To get the warp's local_max for head h_idx, broadcast from
    //     the lane that owns it.
    int src_lane_for_h;
    int which_local;  // 0 or 1
    if (h_idx < 8) {
      src_lane_for_h = h_idx * 4;  // gid = h_idx, tid = 0
      which_local = 0;
    } else {
      src_lane_for_h = (h_idx - 8) * 4;  // gid = h_idx - 8, tid = 0
      which_local = 1;
    }
    float local_max_h =
        __shfl_sync(0xffffffff, which_local == 0 ? local_max[0] : local_max[1], src_lane_for_h);

    // Read global max/sum for h_idx (broadcast from one read).
    const float g_max_h = sm_warp_max[h_idx];
    const float g_sum_h = sm_warp_sum[h_idx];
    const float resc =
        (g_sum_h > 0.f) ? (exp2f(local_max_h - g_max_h) / g_sum_h) : 0.f;

    // Phase A: per-warp acc.
    float warp_acc[DIM_PER_LANE];
#pragma unroll
    for (int d = 0; d < DIM_PER_LANE; d++) warp_acc[d] = 0.f;
#pragma unroll
    for (int c = 0; c < V3_ENTRIES_PER_WARP; c++) {
      const float p_hc = sm_p_storage[warp_id][h_idx][c];
      const bf16* v_row = sm_kv_warp_x + (size_t)c * D_QK + lane * DIM_PER_LANE;
#pragma unroll
      for (int d = 0; d < DIM_PER_LANE; d++) {
        const float vf = __bfloat162float(v_row[d]);
        warp_acc[d] += p_hc * vf;
      }
    }
#pragma unroll
    for (int d = 0; d < DIM_PER_LANE; d++) warp_acc[d] *= resc;

    // Phase B: cross-warp merge in sm_head_buf via atomic-add.
    // Zero the buf first (only the D_V slots we use).
    for (int i = threadIdx.x; i < D_V; i += V3_BLOCK_THREADS) sm_head_buf[i] = 0.f;
    __syncthreads();
#pragma unroll
    for (int d = 0; d < DIM_PER_LANE; d++) {
      atomicAdd(&sm_head_buf[lane * DIM_PER_LANE + d], warp_acc[d]);
    }
    __syncthreads();

    // Phase C: write mid_out for this head.
    if (warp_id == 0) {
#pragma unroll
      for (int d = 0; d < DIM_PER_LANE; d++) {
        const int dim_idx = lane * DIM_PER_LANE + d;
        bf16 bv = __float2bfloat16(sm_head_buf[dim_idx]);
        mid_out[mid_o_base_ll + (size_t)h_idx * num_splits * D_V + dim_idx] = bv;
      }
    }
    __syncthreads();
  }

  // Phase D: write LSE per head (HPB threads).
  if (threadIdx.x < HPB) {
    const int h = threadIdx.x;
    const float g_max = sm_warp_max[h];
    const float g_sum = sm_warp_sum[h];
    const float lse = (g_sum > 0.f) ? (log2f(g_sum) + g_max) : -1e30f;
    mid_lse[mid_lse_base_ll + (size_t)h * num_splits + split_idx] = lse;
  }
}

// Merge kernel: collapse splits into final output and LSE.
//
// Grid: (num_tokens, num_head_blocks_for_merge), block: 32 × HPB.
// Each block handles HPB heads for one token; each warp handles one
// head, looping over num_splits to compute LSE-weighted sum.
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

  // Find global LSE across splits.
  float my_lse[16];  // up to 16 splits per thread (256 / 16 = 16)
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

  // Output: sum_split (weight * mid_out[t, h, split, :])
  // where weight = exp2(lse[split] - global_max).
  const bf16* mid_base =
      mid_out + (size_t)t_idx * NUM_HEADS * num_splits * D_V_VAL + (size_t)h * num_splits * D_V_VAL;
  bf16* out_ptr = output + (size_t)t_idx * NUM_HEADS * D_V_VAL + (size_t)h * D_V_VAL;

  constexpr int DIMS_PER_LANE = D_V_VAL / 32;  // 16 for D_V=512
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
