"""Sparse MLA prefill kernel for NSA on SM120.

Grid: (num_q, num_head_tiles)
Each program handles HPG=16 heads for ONE Q token, iterating over top_k KV tokens.
Online softmax avoids materializing [heads, top_k] score matrix.

For num_q=8192, HPG=16: grid = 8192 × 8 = 65536 programs → good GPU utilization.
Compare: decode kernel uses grid = (num_q, k_splits) with ALL 128 heads per program.
"""

import triton
import triton.language as tl


@triton.jit
def _sparse_mla_prefill_kernel(
    Q,                  # [num_q * num_heads, head_dim] bf16
    KV_NOPE_FP8,        # [total_tokens, dim_nope] fp8
    KV_SCALES,           # [total_tokens, num_groups] f32
    KV_ROPE_BF16,        # [total_tokens, dim_rope] bf16
    SPARSE_INDICES,      # [num_q, top_k] int32
    OUTPUT,              # [num_q * num_heads, kv_lora_rank] bf16
    stride_nope,
    stride_scales,
    stride_rope,
    stride_idx,         # top_k
    scale,
    NUM_HEADS: tl.constexpr,
    DIM_NOPE: tl.constexpr,    # 512
    DIM_ROPE: tl.constexpr,    # 64
    KV_LORA_RANK: tl.constexpr, # 512
    GROUP_SIZE: tl.constexpr,   # 128
    HPG: tl.constexpr,          # 16 heads per group
    BLOCK_K: tl.constexpr,      # 16 KV tokens per iter
    TOP_K: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_hg = tl.program_id(1)

    HEAD_DIM: tl.constexpr = DIM_NOPE + DIM_ROPE
    head_start = pid_hg * HPG
    h_range = tl.arange(0, HPG)
    k_range = tl.arange(0, BLOCK_K)

    si_ptr = SPARSE_INDICES + pid_q * stride_idx
    q_base = Q + (pid_q * NUM_HEADS + head_start) * HEAD_DIM

    # Pre-load Q tiles (constant across KV iterations)
    g_range = tl.arange(0, GROUP_SIZE)
    rope_range = tl.arange(0, DIM_ROPE)

    q_g0 = tl.load(q_base + h_range[:, None] * HEAD_DIM + g_range[None, :])
    q_g1 = tl.load(q_base + h_range[:, None] * HEAD_DIM + GROUP_SIZE + g_range[None, :])
    q_g2 = tl.load(q_base + h_range[:, None] * HEAD_DIM + 2 * GROUP_SIZE + g_range[None, :])
    q_g3 = tl.load(q_base + h_range[:, None] * HEAD_DIM + 3 * GROUP_SIZE + g_range[None, :])
    q_rope = tl.load(q_base + h_range[:, None] * HEAD_DIM + DIM_NOPE + rope_range[None, :])

    # Online softmax + V accumulation state
    m_prev = tl.full([HPG], float("-inf"), dtype=tl.float32)
    d_prev = tl.zeros([HPG], dtype=tl.float32)
    o0 = tl.zeros([HPG, GROUP_SIZE], dtype=tl.float32)
    o1 = tl.zeros([HPG, GROUP_SIZE], dtype=tl.float32)
    o2 = tl.zeros([HPG, GROUP_SIZE], dtype=tl.float32)
    o3 = tl.zeros([HPG, GROUP_SIZE], dtype=tl.float32)

    # Iterate over top_k in chunks of BLOCK_K
    for k_start in range(0, TOP_K, BLOCK_K):
        k_idx = k_start + k_range
        k_mask = k_idx < TOP_K

        indices = tl.load(si_ptr + k_idx, mask=k_mask, other=-1)
        valid = k_mask & (indices >= 0)

        # ── Score: 4 nope groups + rope ──────────────────────────────────────
        scores = tl.zeros([HPG, BLOCK_K], dtype=tl.float32)

        # Group 0
        nope0 = tl.load(KV_NOPE_FP8 + indices[:, None] * stride_nope + g_range[None, :],
                         mask=valid[:, None], other=0.0)
        s0 = tl.load(KV_SCALES + indices * stride_scales + 0, mask=valid, other=0.0)
        kv0 = (nope0.to(tl.float32) * s0[:, None]).to(tl.bfloat16)
        scores += tl.dot(q_g0, tl.trans(kv0), out_dtype=tl.float32)

        # Group 1
        nope1 = tl.load(KV_NOPE_FP8 + indices[:, None] * stride_nope + GROUP_SIZE + g_range[None, :],
                         mask=valid[:, None], other=0.0)
        s1 = tl.load(KV_SCALES + indices * stride_scales + 1, mask=valid, other=0.0)
        kv1 = (nope1.to(tl.float32) * s1[:, None]).to(tl.bfloat16)
        scores += tl.dot(q_g1, tl.trans(kv1), out_dtype=tl.float32)

        # Group 2
        nope2 = tl.load(KV_NOPE_FP8 + indices[:, None] * stride_nope + 2 * GROUP_SIZE + g_range[None, :],
                         mask=valid[:, None], other=0.0)
        s2 = tl.load(KV_SCALES + indices * stride_scales + 2, mask=valid, other=0.0)
        kv2 = (nope2.to(tl.float32) * s2[:, None]).to(tl.bfloat16)
        scores += tl.dot(q_g2, tl.trans(kv2), out_dtype=tl.float32)

        # Group 3
        nope3 = tl.load(KV_NOPE_FP8 + indices[:, None] * stride_nope + 3 * GROUP_SIZE + g_range[None, :],
                         mask=valid[:, None], other=0.0)
        s3 = tl.load(KV_SCALES + indices * stride_scales + 3, mask=valid, other=0.0)
        kv3 = (nope3.to(tl.float32) * s3[:, None]).to(tl.bfloat16)
        scores += tl.dot(q_g3, tl.trans(kv3), out_dtype=tl.float32)

        # Rope
        kv_rope = tl.load(KV_ROPE_BF16 + indices[:, None] * stride_rope + rope_range[None, :],
                           mask=valid[:, None], other=0.0)
        scores += tl.dot(q_rope, tl.trans(kv_rope), out_dtype=tl.float32)

        scores = scores * scale
        scores = tl.where(valid[None, :], scores, float("-inf"))

        # ── Online softmax ───────────────────────────────────────────────────
        m_cur = tl.max(scores, axis=1)
        m_new = tl.maximum(m_prev, m_cur)
        alpha = tl.where(d_prev > 0.0, tl.exp(m_prev - m_new), 0.0)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(valid[None, :], p, 0.0)
        d_new = d_prev * alpha + tl.sum(p, axis=1)
        p_bf16 = p.to(tl.bfloat16)

        # ── V accumulation: reuse already-loaded dequanted KV ────────────────
        o0 = o0 * alpha[:, None] + tl.dot(p_bf16, kv0, out_dtype=tl.float32)
        o1 = o1 * alpha[:, None] + tl.dot(p_bf16, kv1, out_dtype=tl.float32)
        o2 = o2 * alpha[:, None] + tl.dot(p_bf16, kv2, out_dtype=tl.float32)
        o3 = o3 * alpha[:, None] + tl.dot(p_bf16, kv3, out_dtype=tl.float32)

        m_prev = m_new
        d_prev = d_new

    # Normalize and store
    inv_d = (1.0 / tl.maximum(d_prev, 1e-6))[:, None]
    out_base = OUTPUT + (pid_q * NUM_HEADS + head_start) * KV_LORA_RANK
    tl.store(out_base + h_range[:, None] * KV_LORA_RANK + g_range[None, :],
             (o0 * inv_d).to(tl.bfloat16))
    tl.store(out_base + h_range[:, None] * KV_LORA_RANK + GROUP_SIZE + g_range[None, :],
             (o1 * inv_d).to(tl.bfloat16))
    tl.store(out_base + h_range[:, None] * KV_LORA_RANK + 2 * GROUP_SIZE + g_range[None, :],
             (o2 * inv_d).to(tl.bfloat16))
    tl.store(out_base + h_range[:, None] * KV_LORA_RANK + 3 * GROUP_SIZE + g_range[None, :],
             (o3 * inv_d).to(tl.bfloat16))
