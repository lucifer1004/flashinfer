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

import triton
import triton.language as tl


@triton.jit
def _fp8_paged_mqa_logits_kernel(
    Q,  # [batch * next_n * num_heads, head_dim] fp8e4m3
    FUSED_KV_FP8,  # fused KV viewed as fp8 (K data loads)
    FUSED_KV_U8,  # same buffer viewed as uint8 (scale byte loads)
    WEIGHTS,  # [batch * next_n, num_heads] float32
    LOGITS,  # [batch * next_n, max_seq_len] float32 (output)
    BLOCK_TABLES,  # [batch, max_blocks_per_seq] int32
    SEQ_LENS,  # [batch] int32
    max_blocks_per_seq,  # grid bound
    fused_stride_block,  # fused_kv.stride(0) in elements
    fused_stride_token: tl.constexpr,  # head_dim_with_sf
    stride_bt_b,  # block_tables stride for batch dim
    stride_logits_b,  # logits stride for batch*next_n dim
    NEXT_N: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    """FP8 paged MQA logits kernel for NSA indexer.

    Each program handles BLOCKS_PER_PROGRAM consecutive KV blocks,
    amortizing Q and weights loads. Uses tl.dot for tensor core matmul.
    Reads K and scales directly from the fused buffer (zero-copy).

    Grid: (batch * next_n, ceil(max_blocks_per_seq / BLOCKS_PER_PROGRAM))
    """
    pid_bn = tl.program_id(0)
    pid_kv_group = tl.program_id(1)

    b = pid_bn // NEXT_N
    seq_len = tl.load(SEQ_LENS + b)

    first_blk = pid_kv_group * BLOCKS_PER_PROGRAM
    if first_blk * BLOCK_KV >= seq_len:
        return

    token_offsets = tl.arange(0, BLOCK_KV)
    dim_offsets = tl.arange(0, HEAD_DIM)
    head_offsets = tl.arange(0, NUM_HEADS)

    # ── Load Q once: [NUM_HEADS, HEAD_DIM] as FP8 ───────────────────────────
    q_base = pid_bn * NUM_HEADS
    q_ptrs = Q + (q_base + head_offsets[:, None]) * HEAD_DIM + dim_offsets[None, :]
    q_fp8 = tl.load(q_ptrs)

    # ── Load weights once: [NUM_HEADS] ──────────────────────────────────────
    weights = tl.load(WEIGHTS + pid_bn * NUM_HEADS + head_offsets)

    # ── Process BLOCKS_PER_PROGRAM consecutive KV blocks ─────────────────────
    for blk_offset in range(BLOCKS_PER_PROGRAM):
        blk_idx = first_blk + blk_offset
        kv_token_start = blk_idx * BLOCK_KV

        if kv_token_start < seq_len and blk_idx < max_blocks_per_seq:
            phys_block = tl.load(BLOCK_TABLES + b * stride_bt_b + blk_idx)
            valid_tokens = tl.minimum(BLOCK_KV, seq_len - kv_token_start)
            fused_base = phys_block * fused_stride_block

            # Load K: [BLOCK_KV, HEAD_DIM] as FP8 from fused buffer
            k_ptrs = (
                FUSED_KV_FP8
                + fused_base
                + token_offsets[:, None] * fused_stride_token
                + dim_offsets[None, :]
            )
            k_mask = token_offsets[:, None] < valid_tokens
            k_fp8 = tl.load(k_ptrs, mask=k_mask, other=0.0)

            # Q @ K^T via tensor cores
            s = tl.dot(q_fp8, tl.trans(k_fp8), out_dtype=tl.float32)

            # Load scales: 4 uint8 bytes per token, reinterpret as float32
            scale_byte_ptrs = (
                FUSED_KV_U8 + fused_base
                + token_offsets * fused_stride_token + HEAD_DIM
            )
            smask = token_offsets < valid_tokens
            b0 = tl.load(scale_byte_ptrs, mask=smask, other=0).to(tl.uint32)
            b1 = tl.load(scale_byte_ptrs + 1, mask=smask, other=0).to(tl.uint32)
            b2 = tl.load(scale_byte_ptrs + 2, mask=smask, other=0).to(tl.uint32)
            b3 = tl.load(scale_byte_ptrs + 3, mask=smask, other=0).to(tl.uint32)
            scales = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)).to(tl.float32, bitcast=True)

            # Scale, ReLU, weight-reduce
            s = s * scales[None, :]
            s = tl.maximum(s, 0.0)
            s = s * weights[:, None]
            logits_out = tl.sum(s, axis=0)

            # Store
            out_ptrs = LOGITS + pid_bn * stride_logits_b + kv_token_start + token_offsets
            tl.store(out_ptrs, logits_out, mask=token_offsets < valid_tokens)
