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

"""Fused FP8 sparse MLA decode: dequant + attention in one kernel.

Reads FP8 KV cache directly, dequants per-tile, computes attention via tl.dot.
Eliminates the intermediate BF16 buffer that the two-pass approach materializes.

Grid: (batch, K_SPLITS)
Each program: all NUM_HEADS heads, processes top_k / K_SPLITS sparse tokens.
Uses tl.dot for tensor cores (requires NUM_HEADS >= 16).

FP8 KV cache layout per token (656 bytes):
  [0:512]   CKV as FP8     (4 groups of 128, each with a FP32 scale)
  [512:528] scales as FP32  (4 × 4 bytes = 16, stored as FP8 bytes)
  [528:656] KPE as BF16     (64 × 2 bytes = 128, stored as FP8 bytes)

Split-K: each program produces partial output [NUM_HEADS, KV_LORA_RANK] + softmax
state (m, d) per head.  A separate merge kernel combines K_SPLITS partials.
"""

import triton
import triton.language as tl


@triton.jit
def _fused_sparse_mla_decode_kernel(
    Q,                  # [batch * NUM_HEADS, HEAD_DIM] bf16
    KV_NOPE_FP8,        # [total_tokens, DIM_NOPE] fp8 (view of fused buf)
    KV_SCALES,           # [total_tokens, NUM_GROUPS] fp32 (view of fused buf)
    KV_ROPE_BF16,        # [total_tokens, DIM_ROPE] bf16 (view of fused buf)
    SPARSE_INDICES,      # [batch, TOP_K] int32
    # Split-K outputs:
    OUT_PARTIAL,         # [batch, K_SPLITS, NUM_HEADS, KV_LORA_RANK] fp32
    OUT_M,               # [batch, K_SPLITS, NUM_HEADS] fp32  (softmax max)
    OUT_D,               # [batch, K_SPLITS, NUM_HEADS] fp32  (softmax denom)
    stride_nope,         # KV nope token stride
    stride_scales,       # KV scales token stride
    stride_rope,         # KV rope token stride
    scale,               # float: attention scale
    TOP_K: tl.constexpr,
    NUM_HEADS: tl.constexpr,      # 16
    DIM_NOPE: tl.constexpr,       # 512
    DIM_ROPE: tl.constexpr,       # 64
    KV_LORA_RANK: tl.constexpr,   # 512
    GROUP_SIZE: tl.constexpr,      # 128
    NUM_GROUPS: tl.constexpr,      # 4
    BLOCK_K: tl.constexpr,         # 16
    K_SPLITS: tl.constexpr,
    TOKENS_PER_SPLIT: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    HEAD_DIM: tl.constexpr = DIM_NOPE + DIM_ROPE
    h_range = tl.arange(0, NUM_HEADS)
    k_range = tl.arange(0, BLOCK_K)

    si_ptr = SPARSE_INDICES + pid_b * TOP_K
    q_base = Q + pid_b * NUM_HEADS * HEAD_DIM

    # Token range for this split
    split_start = pid_s * TOKENS_PER_SPLIT
    split_end = tl.minimum(split_start + TOKENS_PER_SPLIT, TOP_K)

    # Online softmax state: per-head
    m_prev = tl.full([NUM_HEADS], float("-inf"), dtype=tl.float32)
    d_prev = tl.zeros([NUM_HEADS], dtype=tl.float32)

    # V accumulators: 4 groups of [NUM_HEADS, GROUP_SIZE=128]
    o0 = tl.zeros([NUM_HEADS, GROUP_SIZE], dtype=tl.float32)
    o1 = tl.zeros([NUM_HEADS, GROUP_SIZE], dtype=tl.float32)
    o2 = tl.zeros([NUM_HEADS, GROUP_SIZE], dtype=tl.float32)
    o3 = tl.zeros([NUM_HEADS, GROUP_SIZE], dtype=tl.float32)

    for k_start in range(split_start, split_start + TOKENS_PER_SPLIT, BLOCK_K):
        k_idx = k_start + k_range
        k_mask = k_idx < split_end

        indices = tl.load(si_ptr + k_idx, mask=k_mask, other=-1)
        valid = k_mask & (indices >= 0)

        # ── Score: Q × dequant(KV)^T, tiled over nope groups + rope ──
        scores = tl.zeros([NUM_HEADS, BLOCK_K], dtype=tl.float32)

        # Nope groups (4 × 128)
        for g in range(NUM_GROUPS):
            g_off = g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
            # Q tile: [NUM_HEADS, GROUP_SIZE] bf16
            q_tile = tl.load(q_base + h_range[:, None] * HEAD_DIM + g_off[None, :])
            # KV nope tile: [BLOCK_K, GROUP_SIZE] fp8
            nope_ptrs = KV_NOPE_FP8 + indices[:, None] * stride_nope + g_off[None, :]
            nope_fp8 = tl.load(nope_ptrs, mask=valid[:, None], other=0.0)
            # Scale: [BLOCK_K] fp32
            kv_scale = tl.load(KV_SCALES + indices * stride_scales + g, mask=valid, other=0.0)
            # Dequant to bf16: fp8 * scale → bf16
            nope_bf16 = (nope_fp8.to(tl.float32) * kv_scale[:, None]).to(tl.bfloat16)
            # Score partial: [H, GS] × [GS, BK] = [H, BK]
            scores += tl.dot(q_tile, tl.trans(nope_bf16), out_dtype=tl.float32)

        # Rope: [BLOCK_K, DIM_ROPE=64] bf16
        rope_off = tl.arange(0, DIM_ROPE)
        q_rope = tl.load(q_base + h_range[:, None] * HEAD_DIM + DIM_NOPE + rope_off[None, :])
        rope_ptrs = KV_ROPE_BF16 + indices[:, None] * stride_rope + rope_off[None, :]
        kv_rope = tl.load(rope_ptrs, mask=valid[:, None], other=0.0)
        scores += tl.dot(q_rope, tl.trans(kv_rope), out_dtype=tl.float32)

        scores = scores * scale
        scores = tl.where(valid[None, :], scores, float("-inf"))

        # ── Online softmax (per-head) ──
        m_cur = tl.max(scores, axis=1)  # [NUM_HEADS]
        m_new = tl.maximum(m_prev, m_cur)
        alpha = tl.where(d_prev > 0.0, tl.exp(m_prev - m_new), 0.0)  # [NUM_HEADS]
        p = tl.exp(scores - m_new[:, None])  # [NUM_HEADS, BLOCK_K]
        p = tl.where(valid[None, :], p, 0.0)
        d_new = d_prev * alpha + tl.sum(p, axis=1)

        p_bf16 = p.to(tl.bfloat16)

        # ── V accumulation: P × dequant(KV_nope), tiled over groups ──
        # Group 0
        g0_off = tl.arange(0, GROUP_SIZE)
        nope0 = tl.load(KV_NOPE_FP8 + indices[:, None] * stride_nope + g0_off[None, :],
                         mask=valid[:, None], other=0.0)
        s0 = tl.load(KV_SCALES + indices * stride_scales + 0, mask=valid, other=0.0)
        v0 = (nope0.to(tl.float32) * s0[:, None]).to(tl.bfloat16)
        o0 = o0 * alpha[:, None] + tl.dot(p_bf16, v0, out_dtype=tl.float32)

        # Group 1
        g1_off = GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        nope1 = tl.load(KV_NOPE_FP8 + indices[:, None] * stride_nope + g1_off[None, :],
                         mask=valid[:, None], other=0.0)
        s1 = tl.load(KV_SCALES + indices * stride_scales + 1, mask=valid, other=0.0)
        v1 = (nope1.to(tl.float32) * s1[:, None]).to(tl.bfloat16)
        o1 = o1 * alpha[:, None] + tl.dot(p_bf16, v1, out_dtype=tl.float32)

        # Group 2
        g2_off = 2 * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        nope2 = tl.load(KV_NOPE_FP8 + indices[:, None] * stride_nope + g2_off[None, :],
                         mask=valid[:, None], other=0.0)
        s2 = tl.load(KV_SCALES + indices * stride_scales + 2, mask=valid, other=0.0)
        v2 = (nope2.to(tl.float32) * s2[:, None]).to(tl.bfloat16)
        o2 = o2 * alpha[:, None] + tl.dot(p_bf16, v2, out_dtype=tl.float32)

        # Group 3
        g3_off = 3 * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        nope3 = tl.load(KV_NOPE_FP8 + indices[:, None] * stride_nope + g3_off[None, :],
                         mask=valid[:, None], other=0.0)
        s3 = tl.load(KV_SCALES + indices * stride_scales + 3, mask=valid, other=0.0)
        v3 = (nope3.to(tl.float32) * s3[:, None]).to(tl.bfloat16)
        o3 = o3 * alpha[:, None] + tl.dot(p_bf16, v3, out_dtype=tl.float32)

        m_prev = m_new
        d_prev = d_new

    # Store partial results for this split
    out_base = OUT_PARTIAL + (pid_b * K_SPLITS + pid_s) * NUM_HEADS * KV_LORA_RANK
    g_range = tl.arange(0, GROUP_SIZE)
    tl.store(out_base + h_range[:, None] * KV_LORA_RANK + g_range[None, :], o0)
    tl.store(out_base + h_range[:, None] * KV_LORA_RANK + GROUP_SIZE + g_range[None, :], o1)
    tl.store(out_base + h_range[:, None] * KV_LORA_RANK + 2 * GROUP_SIZE + g_range[None, :], o2)
    tl.store(out_base + h_range[:, None] * KV_LORA_RANK + 3 * GROUP_SIZE + g_range[None, :], o3)

    m_base = OUT_M + (pid_b * K_SPLITS + pid_s) * NUM_HEADS
    d_base = OUT_D + (pid_b * K_SPLITS + pid_s) * NUM_HEADS
    tl.store(m_base + h_range, m_prev)
    tl.store(d_base + h_range, d_prev)


@triton.jit
def _merge_splits_kernel(
    OUT_PARTIAL,    # [batch, K_SPLITS, NUM_HEADS, KV_LORA_RANK] fp32
    OUT_M,          # [batch, K_SPLITS, NUM_HEADS] fp32
    OUT_D,          # [batch, K_SPLITS, NUM_HEADS] fp32
    OUTPUT,         # [batch, NUM_HEADS, KV_LORA_RANK] bf16
    K_SPLITS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Merge K_SPLITS partial results via online softmax state combination.
    Grid: (batch * NUM_HEADS, ceil(KV_LORA_RANK / BLOCK_V))
    """
    pid = tl.program_id(0)
    pid_v = tl.program_id(1)
    b = pid // NUM_HEADS
    h = pid % NUM_HEADS

    v_off = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_off < KV_LORA_RANK

    # Merge across splits using online softmax
    m_merged = float("-inf")
    d_merged = 0.0
    o_merged = tl.zeros([BLOCK_V], dtype=tl.float32)

    for s in range(K_SPLITS):
        base = (b * K_SPLITS + s) * NUM_HEADS
        m_s = tl.load(OUT_M + base + h)
        d_s = tl.load(OUT_D + base + h)

        o_base = (b * K_SPLITS + s) * NUM_HEADS * KV_LORA_RANK + h * KV_LORA_RANK
        o_s = tl.load(OUT_PARTIAL + o_base + v_off, mask=v_mask, other=0.0)

        # Online merge
        m_new = tl.maximum(m_merged, m_s)
        alpha_old = tl.where(d_merged > 0.0, tl.exp(m_merged - m_new), 0.0)
        alpha_new = tl.where(d_s > 0.0, tl.exp(m_s - m_new), 0.0)
        d_merged = d_merged * alpha_old + d_s * alpha_new
        o_merged = o_merged * alpha_old + o_s * alpha_new
        m_merged = m_new

    # Normalize and store
    o_merged = o_merged / tl.maximum(d_merged, 1e-6)
    out_ptr = OUTPUT + (b * NUM_HEADS + h) * KV_LORA_RANK
    tl.store(out_ptr + v_off, o_merged.to(tl.bfloat16), mask=v_mask)
