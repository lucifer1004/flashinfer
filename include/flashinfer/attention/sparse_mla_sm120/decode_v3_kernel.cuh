// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// (license header — full text mirrors the other files in this directory)

#pragma once

#include <cute/swizzle.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

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

// sm_kv smem swizzle, expressed as a cute::Swizzle so the swizzle is a
// property of the layout rather than scattered through call sites.
//
// Swizzle<3, 3, 6> XORs bits [9, 12) of the linear bf16 offset (= low 3
// bits of cand_idx for D_QK=512) onto bits [3, 6) (= bank-affecting bits
// 3..5 of the bf16 offset, equivalently banks 2..4 after the /2 to
// 4-byte units). This gives full 32-bank distribution for Stage 2 QK
// reads (8-way → 0-way) and 2-way for Stage 3 XV reads (4-way → 2-way),
// matching the perf characteristics of the prior hand-rolled `v3_swiz`.
//
// Equivalent hand-rolled form:
//     swiz(dim, cand) = dim ^ ((cand & 7) << 3)   (in bf16 units)
//
// The XOR stays within each cand row (max XOR mask = 56 bf16 < D_QK) and
// preserves the NoPE/RoPE boundary at logical dim D_NOPE = 448 because
// 448's bit-pattern has none of bits 3..5 set.
using SmemKVSwizzle = cute::Swizzle<3, 3, 6>;

// Apply the cute swizzle to a (cand, dim) pair in bf16 units → physical
// linear offset within sm_kv (also in bf16). Templated on D_QK so the
// stride is a compile-time constant.
template <int D_QK_CONST>
__device__ __forceinline__ int v3_swiz_offset(int cand, int dim) {
  const int linear = cand * D_QK_CONST + dim;
  return SmemKVSwizzle::apply(linear);
}

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
  // Cross-warp XV merge buffer — separate from sm_kv because we still
  // need to read V from sm_kv across the outer head loop. Cannot reuse
  // sm_kv; that would corrupt V midway through the loop.
  float* sm_head_buf = sm_warp_sum + V3_N_WARPS * HPB;  // [D_V] = 2 KB

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
    // ILP across consecutive entries: ask nvcc to unroll a few iters so it
    // can issue multiple loads in flight before any dequant completes.
    // Hides the L2 latency that shows up as long_scoreboard (3.87 stalls/
    // issue post-swizzle) — gmem loads can overlap across the 14 LDG.U8s
    // of cand[eo] and cand[eo+1].
#pragma unroll 16
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
        // Note: `__nv_fp8_e4m3 raw{byte}; static_cast<float>(raw)` was
        // dropping the sign bit (positive-only output) — confirmed via
        // printf. Use __half conversion as the canonical decode path.
        __nv_fp8_e4m3 raw;
        raw.__x = static_cast<__nv_fp8_storage_t>(data_base[b]);
        const float v = static_cast<float>(static_cast<__half>(raw));
        // Swizzled store — SmemKVSwizzle (cute::Swizzle<3,3,6>) puts each
        // cand row's payload at a permuted dim offset, so Stage 2/3 reads
        // are bank-conflict-free.
        sm_kv[v3_swiz_offset<D_QK>(entry_idx, b)] =
            __float2bfloat16(is_valid_cand ? (v * scale) : 0.0f);
      }
      // RoPE half: already bf16 in gmem, 128 bytes = 64 bf16, copy verbatim.
      // 32 lanes × 2 bf16 each = 64 bf16. Apply same row swizzle.
      const bf16* rope_src = reinterpret_cast<const bf16*>(data_base + D_NOPE);
      if (lane * 2 < D_ROPE) {
        const int rope_dim0 = D_NOPE + lane * 2;
        const int rope_dim1 = rope_dim0 + 1;
        sm_kv[v3_swiz_offset<D_QK>(entry_idx, rope_dim0)] =
            is_valid_cand ? rope_src[lane * 2] : __float2bfloat16(0.0f);
        sm_kv[v3_swiz_offset<D_QK>(entry_idx, rope_dim1)] =
            is_valid_cand ? rope_src[lane * 2 + 1] : __float2bfloat16(0.0f);
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
    // Per-thread B-operand for bf16 mma_m16n8k16 holds 4 bf16 values arranged
    // as K-rows {tid*2, tid*2+1, tid*2+8, tid*2+9} for the per-thread N-col.
    // Mirrors xv_rope_mma.cuh's manual scalar-load pattern; ldmatrix.x2.trans
    // produces a different layout we don't want.
    //
    // sm_kv is stored with per-row swizzle (see v3_swiz), so each read here
    // resolves the swizzle by the row's *total* entry index in sm_kv (which
    // equals warp_id * V3_ENTRIES_PER_WARP + cand_idx). With the swizzle,
    // the 8 cand-rows in an N-tile hit 8 distinct bank groups, eliminating
    // the prior 8-way load bank conflict.
#pragma unroll
    for (int ks = 0; ks < D_QK / 16; ks++) {
      uint32_t a0, a1, a2, a3;
      ldmatrix_load_A_bf16(a0, a1, a2, a3, sm_q + ks * 16, D_QK, lane);
#pragma unroll
      for (int nt = 0; nt < 2; nt++) {
        // ldmatrix.x2.trans: input is 8 N-rows × 16 K-cols at sm_kv[cand][dim].
        // Output is 16K × 8N register fragment, which is exactly the bf16
        // m16n8k16 B-operand layout. Lanes 0..7 supply addresses for col=0
        // half (cand_row, dim_chunk_start); lanes 8..15 supply col=8 half;
        // lanes 16..31 redundant. The swizzle is applied to the per-thread
        // smem address so the load is bank-conflict-free.
        const int row_in_tile = lane & 7;      // 0..7 cand within this n-tile
        const int col_half = (lane >> 3) & 1;  // 0 or 1
        const int cand_total = warp_id * V3_ENTRIES_PER_WARP + nt * 8 + row_in_tile;
        const int dim_chunk_start = ks * 16 + col_half * 8;
        uint32_t b0, b1;
        ldmatrix_x2_trans(b0, b1, sm_kv + v3_swiz_offset<D_QK>(cand_total, dim_chunk_start));
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
  // CAUTION: sm_warp_max[w*HPB+h] for w=0, h=0 is the SAME slot as
  // sm_warp_max[0]. So the first thread (which writes h=0 slot) needs
  // to read all 4 warp_max values FIRST before overwriting. The loop
  // below reads then writes — sequence is fine in this single-thread
  // body, but we must NOT have other threads writing to slot 0 from a
  // different access while we read. The __syncthreads above guarantees
  // all 4 warps' writes have completed before any thread reads.
  // After the write, slot 0 holds gmax for h=0 (overwriting warp 0's
  // local_max for h=0). The same is true for the next 15 slots.
  if (threadIdx.x < HPB) {
    const int h = threadIdx.x;
    // Read all 4 per-warp values into registers FIRST.
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

  // ── Stage 2.75: globally-normalized softmax weights as BF16 ─────────
  // To avoid an atomic-add cross-warp merge for XV, we partition the
  // OUTPUT DIM across warps instead of partitioning candidates. Each
  // warp will do MMA on ALL 64 candidates × its own D_V/N_WARPS=128-dim
  // slice. For that to work each warp must see ALL candidates' softmax
  // weights, so we stage them into a shared 2 KB buffer sm_p_full of
  // shape [HPB=16 heads × V3_BI=64 candidates] in bf16.
  //
  // We multiply local p (per-warp softmax) by the per-head global
  // rescale exp(local_max - g_max) / g_sum so sm_p_full holds the
  // FULLY normalized softmax weights; XV output is then the final
  // output for this block, no further normalization needed.
  __shared__ bf16 sm_p_full[HPB][V3_BI];  // 16 × 64 × 2 = 2 KB static

  const float gmax0 = sm_warp_max[gid];
  const float gmax1 = sm_warp_max[gid + 8];
  const float gsum0 = sm_warp_sum[gid];
  const float gsum1 = sm_warp_sum[gid + 8];
  const float resc_h0 = (gsum0 > 0.f) ? (exp2f(local_max[0] - gmax0) / gsum0) : 0.f;
  const float resc_h1 = (gsum1 > 0.f) ? (exp2f(local_max[1] - gmax1) / gsum1) : 0.f;

  // Each warp writes its 16 candidates' contribution at column offset
  // [warp_id * V3_ENTRIES_PER_WARP, (warp_id+1) * V3_ENTRIES_PER_WARP).
  // Per thread writes 8 bf16 values into sm_p_full at the right (h, c).
  const int cand_col_base = warp_id * V3_ENTRIES_PER_WARP;
#pragma unroll
  for (int nt = 0; nt < 2; nt++) {
    const int c0 = nt * 8 + tid * 2;
    const int c1 = c0 + 1;
    sm_p_full[gid][cand_col_base + c0] = __float2bfloat16(p[nt][0] * resc_h0);
    sm_p_full[gid][cand_col_base + c1] = __float2bfloat16(p[nt][1] * resc_h0);
    sm_p_full[gid + 8][cand_col_base + c0] = __float2bfloat16(p[nt][2] * resc_h1);
    sm_p_full[gid + 8][cand_col_base + c1] = __float2bfloat16(p[nt][3] * resc_h1);
  }
  __syncthreads();

  // ── Stage 3: XV via bf16 MMA — each warp owns a D_V/N_WARPS slice ──
  // Per warp: 4 K-iter (covering all 64 cands) × N_TILES_PER_WARP N-tiles
  // (covering D_V/N_WARPS=128 dims). No cross-warp merge.
  //
  // K-iter index ks ∈ [0, 4): A-cols are cands [ks*16, ks*16+16).
  // For each ks, do N_TILES_PER_WARP MMAs (= 128/8 = 16) accumulating
  // into per-lane acc[N_TILES_PER_WARP][4] = 64 floats.
  constexpr int DIMS_PER_WARP = D_V / V3_N_WARPS;   // 128
  constexpr int N_TILES_PER_WARP = DIMS_PER_WARP / 8;  // 16
  constexpr int K_ITERS = V3_BI / 16;                // 4

  const int warp_dim_base = warp_id * DIMS_PER_WARP;
  const size_t mid_o_base_ll = ((size_t)t_idx * NUM_HEADS + h_start) * (size_t)num_splits * D_V +
                               (size_t)split_idx * D_V;
  const size_t mid_lse_base_ll =
      (size_t)t_idx * NUM_HEADS * num_splits + (size_t)h_start * num_splits;

  float acc[N_TILES_PER_WARP][4] = {0};

#pragma unroll
  for (int ks = 0; ks < K_ITERS; ks++) {
    // A operand: P[16h, 16k=cands ks*16..ks*16+15]
    uint32_t a0, a1, a2, a3;
    ldmatrix_load_A_bf16(
        a0, a1, a2, a3,
        reinterpret_cast<const bf16*>(&sm_p_full[0][ks * 16]),
        V3_BI, lane);

#pragma unroll
    for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
      const int n_col = warp_dim_base + nt * 8 + gid;  // per-lane N-col

      // B operand: V[16k=cands ks*16+rows, 1n=n_col]. K-rows: ks*16+{tid*2,
      // tid*2+1, tid*2+8, tid*2+9}. K-row idx is the GLOBAL candidate
      // index within this block. Each row read resolves the v3_swiz by
      // the row's own entry idx.
      const int k_base = ks * 16;
      const int ent0 = k_base + tid * 2;
      const int ent1 = ent0 + 1;
      const int ent8 = ent0 + 8;
      const int ent9 = ent0 + 9;
      uint16_t v0 = *reinterpret_cast<const uint16_t*>(sm_kv + v3_swiz_offset<D_QK>(ent0, n_col));
      uint16_t v1 = *reinterpret_cast<const uint16_t*>(sm_kv + v3_swiz_offset<D_QK>(ent1, n_col));
      uint16_t v8 = *reinterpret_cast<const uint16_t*>(sm_kv + v3_swiz_offset<D_QK>(ent8, n_col));
      uint16_t v9 = *reinterpret_cast<const uint16_t*>(sm_kv + v3_swiz_offset<D_QK>(ent9, n_col));
      uint32_t b0 = static_cast<uint32_t>(v0) | (static_cast<uint32_t>(v1) << 16);
      uint32_t b1 = static_cast<uint32_t>(v8) | (static_cast<uint32_t>(v9) << 16);

      MmaBf16Result r = mma_bf16_m16n8k16(
          a0, a1, a2, a3, b0, b1,
          acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]);
      acc[nt][0] = r.d0;
      acc[nt][1] = r.d1;
      acc[nt][2] = r.d2;
      acc[nt][3] = r.d3;
    }
  }

  // Write directly to mid_out (no cross-warp merge needed — each warp
  // owns disjoint D_V slice). Layout per lane:
  //   acc[nt][0,1] = (head gid,   dim warp_dim_base+nt*8+tid*2, +1)
  //   acc[nt][2,3] = (head gid+8, dim warp_dim_base+nt*8+tid*2, +1)
#pragma unroll
  for (int nt = 0; nt < N_TILES_PER_WARP; nt++) {
    const int d0 = warp_dim_base + nt * 8 + tid * 2;
    const int d1 = d0 + 1;
    mid_out[mid_o_base_ll + (size_t)gid * num_splits * D_V + d0] =
        __float2bfloat16(acc[nt][0]);
    mid_out[mid_o_base_ll + (size_t)gid * num_splits * D_V + d1] =
        __float2bfloat16(acc[nt][1]);
    mid_out[mid_o_base_ll + (size_t)(gid + 8) * num_splits * D_V + d0] =
        __float2bfloat16(acc[nt][2]);
    mid_out[mid_o_base_ll + (size_t)(gid + 8) * num_splits * D_V + d1] =
        __float2bfloat16(acc[nt][3]);
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
