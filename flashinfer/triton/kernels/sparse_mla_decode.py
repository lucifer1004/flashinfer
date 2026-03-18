"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Triton kernels for sparse MLA decode attention (NSA Stage 3).

Two variants:
  1. _sparse_mla_decode_fp8_kernel — fused FP8 dequant + attention (production path)
  2. _sparse_mla_decode_kernel — BF16 KV cache (fallback / testing)

FP8 KV cache layout per token (656 bytes, dtype=float8_e4m3fn):
  [0..511]   : CKV as FP8          (512 elements, 4 groups of 128)
  [512..527] : per-group FP32 scales (4 scales × 4 bytes = 16 bytes)
  [528..655] : KPE as BF16          (64 values × 2 bytes = 128 bytes)

Pre-sliced into three views for the kernel:
  KV_NOPE_FP8  [total_tokens, 512]  float8_e4m3fn  stride=(656, 1)
  KV_SCALES    [total_tokens, 4]    float32         stride=(164, 1)
  KV_ROPE_BF16 [total_tokens, 64]   bfloat16        stride=(328, 1)
"""

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused FP8 dequant + sparse MLA decode (production kernel)
# ---------------------------------------------------------------------------

@triton.jit
def _sparse_mla_decode_fp8_kernel(
    Q,                  # [batch * num_heads, head_dim] bf16 (head_dim = dim_nope + dim_rope)
    KV_NOPE_FP8,        # [total_tokens, dim_nope] fp8 (view of fused buffer)
    KV_SCALES,           # [total_tokens, num_tiles] fp32 (view of fused buffer)
    KV_ROPE_BF16,        # [total_tokens, dim_rope] bf16 (view of fused buffer)
    SPARSE_INDICES,      # [batch, top_k] int32
    OUTPUT,              # [batch * num_heads, kv_lora_rank] bf16
    stride_nope,         # KV nope token stride (in fp8 elements)
    stride_scales,       # KV scales token stride (in fp32 elements)
    stride_rope,         # KV rope token stride (in bf16 elements)
    scale,               # float: attention scale
    TOP_K: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    DIM_NOPE: tl.constexpr,      # 512
    DIM_ROPE: tl.constexpr,      # 64
    KV_LORA_RANK: tl.constexpr,  # 512 (= DIM_NOPE for MLA)
    PADDED_V: tl.constexpr,      # next power-of-2 >= KV_LORA_RANK
    GROUP_SIZE: tl.constexpr,     # 128
    NUM_NOPE_TILES: tl.constexpr, # DIM_NOPE // GROUP_SIZE = 4
    BLOCK_K: tl.constexpr,        # 16
):
    """Fused FP8 dequant + sparse MLA decode.  One program per (batch, head)."""
    pid = tl.program_id(0)
    b = pid // NUM_HEADS
    h = pid % NUM_HEADS
    bh = b * NUM_HEADS + h

    HEAD_DIM: tl.constexpr = DIM_NOPE + DIM_ROPE
    q_ptr = Q + bh * HEAD_DIM
    si_ptr = SPARSE_INDICES + b * TOP_K
    o_ptr = OUTPUT + bh * KV_LORA_RANK

    # Online softmax state
    m_prev = float("-inf")
    d_prev = 0.0
    o_acc = tl.zeros([PADDED_V], dtype=tl.float32)

    k_range = tl.arange(0, BLOCK_K)
    v_range = tl.arange(0, PADDED_V)
    v_mask = v_range < KV_LORA_RANK

    for k_start in range(0, TOP_K, BLOCK_K):
        k_idx = k_start + k_range
        k_mask = k_idx < TOP_K
        indices = tl.load(si_ptr + k_idx, mask=k_mask, other=-1)
        valid = k_mask & (indices >= 0)

        # ── Phase 1: Scores = Q_nope · dequant(KV_nope) + Q_rope · KV_rope ──

        scores = tl.zeros([BLOCK_K], dtype=tl.float32)

        # Nope score: tile over 512 dims in GROUP_SIZE=128 chunks
        for st in range(NUM_NOPE_TILES):
            d_offs = st * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
            # Q nope tile
            q_tile = tl.load(q_ptr + d_offs).to(tl.float32)
            # KV nope tile (FP8) + dequant
            nope_ptrs = KV_NOPE_FP8 + indices[:, None] * stride_nope + d_offs[None, :]
            nope_fp8 = tl.load(nope_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
            scale_ptrs = KV_SCALES + indices * stride_scales + st
            kv_scale = tl.load(scale_ptrs, mask=valid, other=0.0)  # [BLOCK_K]
            kv_dequant = nope_fp8 * kv_scale[:, None]
            scores += tl.sum(q_tile[None, :] * kv_dequant, axis=1)

        # Rope score
        rope_offs = tl.arange(0, DIM_ROPE)
        q_rope = tl.load(q_ptr + DIM_NOPE + rope_offs).to(tl.float32)
        rope_ptrs = KV_ROPE_BF16 + indices[:, None] * stride_rope + rope_offs[None, :]
        kv_rope = tl.load(rope_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
        scores += tl.sum(q_rope[None, :] * kv_rope, axis=1)

        scores = scores * scale
        scores = tl.where(valid, scores, float("-inf"))

        # ── Online softmax ──
        m_cur = tl.max(scores)
        m_new = tl.maximum(m_prev, m_cur)
        alpha = tl.where(d_prev > 0.0, tl.exp(m_prev - m_new), 0.0)
        p = tl.exp(scores - m_new)
        p = tl.where(valid, p, 0.0)
        d_new = d_prev * alpha + tl.sum(p)

        # ── Phase 2: V accumulation (V = dequant(KV_nope[:kv_lora_rank])) ──
        # Load full nope FP8 + broadcast scales, dequant in one shot
        nope_v_ptrs = KV_NOPE_FP8 + indices[:, None] * stride_nope + v_range[None, :]
        nope_v_fp8 = tl.load(
            nope_v_ptrs, mask=valid[:, None] & v_mask[None, :], other=0.0
        ).to(tl.float32)
        # Broadcast scales: scale for element d is at tile d // GROUP_SIZE
        tile_idx = v_range // GROUP_SIZE  # compile-time: [0,..,0,1,..,1,2,..,2,3,..,3]
        scale_v_ptrs = KV_SCALES + indices[:, None] * stride_scales + tile_idx[None, :]
        scales_bcast = tl.load(
            scale_v_ptrs, mask=valid[:, None] & v_mask[None, :], other=1.0
        )
        v_dequant = nope_v_fp8 * scales_bcast

        pv = tl.sum(p[:, None] * v_dequant, axis=0)
        o_acc = o_acc * alpha + pv

        m_prev = m_new
        d_prev = d_new

    # Normalize and store
    o_acc = o_acc / tl.maximum(d_prev, 1e-6)
    tl.store(o_ptr + v_range, o_acc.to(tl.bfloat16), mask=v_mask)


# ---------------------------------------------------------------------------
# BF16 KV cache kernel (fallback / testing)
# ---------------------------------------------------------------------------

@triton.jit
def _sparse_mla_decode_kernel(
    Q,              # [batch * num_heads, head_dim] bf16
    KV_CACHE,       # [total_tokens, kv_dim] bf16
    SPARSE_INDICES, # [batch, top_k] int32
    OUTPUT,         # [batch * num_heads, kv_lora_rank] bf16
    stride_kv,      # KV cache token stride (kv_dim for contiguous)
    scale,          # float: attention scale
    TOP_K: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    PADDED_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_TILES: tl.constexpr,
):
    """BF16 sparse MLA decode: one program per (batch, head)."""
    pid = tl.program_id(0)
    b = pid // NUM_HEADS
    h = pid % NUM_HEADS
    bh = b * NUM_HEADS + h

    q_ptr = Q + bh * HEAD_DIM
    si_ptr = SPARSE_INDICES + b * TOP_K
    o_ptr = OUTPUT + bh * KV_LORA_RANK

    m_prev = float("-inf")
    d_prev = 0.0
    o_acc = tl.zeros([PADDED_V], dtype=tl.float32)

    k_range = tl.arange(0, BLOCK_K)
    v_range = tl.arange(0, PADDED_V)
    v_mask = v_range < KV_LORA_RANK

    for k_start in range(0, TOP_K, BLOCK_K):
        k_idx = k_start + k_range
        k_mask = k_idx < TOP_K
        indices = tl.load(si_ptr + k_idx, mask=k_mask, other=-1)
        valid = k_mask & (indices >= 0)

        scores = tl.zeros([BLOCK_K], dtype=tl.float32)
        for dt in range(NUM_D_TILES):
            d_off = dt * BLOCK_D + tl.arange(0, BLOCK_D)
            q_tile = tl.load(q_ptr + d_off).to(tl.float32)
            kv_ptrs = KV_CACHE + indices[:, None] * stride_kv + d_off[None, :]
            kv_tile = tl.load(kv_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
            scores += tl.sum(q_tile[None, :] * kv_tile, axis=1)

        scores = scores * scale
        scores = tl.where(valid, scores, float("-inf"))

        m_cur = tl.max(scores)
        m_new = tl.maximum(m_prev, m_cur)
        alpha = tl.where(d_prev > 0.0, tl.exp(m_prev - m_new), 0.0)
        p = tl.exp(scores - m_new)
        p = tl.where(valid, p, 0.0)
        d_new = d_prev * alpha + tl.sum(p)

        v_ptrs = KV_CACHE + indices[:, None] * stride_kv + v_range[None, :]
        v_tile = tl.load(v_ptrs, mask=valid[:, None] & v_mask[None, :], other=0.0).to(tl.float32)
        pv = tl.sum(p[:, None] * v_tile, axis=0)
        o_acc = o_acc * alpha + pv

        m_prev = m_new
        d_prev = d_new

    o_acc = o_acc / tl.maximum(d_prev, 1e-6)
    tl.store(o_ptr + v_range, o_acc.to(tl.bfloat16), mask=v_mask)
