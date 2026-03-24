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

"""FP8 paged MQA logits for NSA (Native Sparse Attention) indexer.

Drop-in replacement for deep_gemm.fp8_paged_mqa_logits on SM120 (Blackwell).
"""

import torch

from .triton.kernels.fp8_paged_mqa_logits import _fp8_paged_mqa_logits_kernel


def get_paged_mqa_logits_metadata(
    seq_lens: torch.Tensor,
    block_size: int,
    num_sms: int,
) -> torch.Tensor:
    """Compute SM scheduling metadata for load-balanced dispatch.

    The Triton kernel uses a flat grid and does not need sophisticated SM
    scheduling, so this returns a dummy tensor satisfying the API contract.
    """
    return torch.zeros(num_sms + 1, 2, dtype=torch.int32, device=seq_lens.device)


def fp8_paged_mqa_logits(
    q: torch.Tensor,
    fused_kv_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_seq_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    """Compute FP8 paged MQA logits for NSA indexer.

    API-compatible with deep_gemm.fp8_paged_mqa_logits.

    Args:
        q: [batch, next_n, num_heads, head_dim], float8_e4m3fn
        fused_kv_cache: [num_blocks, block_kv, 1, head_dim+4], uint8
        weights: [batch*next_n, num_heads], float32
        seq_lens: [batch], int32
        block_tables: [batch, max_blocks_per_seq], int32
        schedule_metadata: [num_sms+1, 2], int32 (unused by Triton impl)
        max_seq_len: max context length
        clean_logits: zero out positions beyond each sequence's length

    Returns:
        logits: [batch*next_n, max_seq_len], float32
    """
    batch, next_n, num_heads, head_dim = q.shape
    num_blocks, block_kv, _, head_dim_with_sf = fused_kv_cache.shape

    assert q.dtype == torch.float8_e4m3fn
    assert fused_kv_cache.dtype == torch.uint8
    assert weights.dtype == torch.float32
    assert head_dim_with_sf == head_dim + 4
    assert block_kv in (32, 64), f"block_kv must be 32 or 64, got {block_kv}"
    assert next_n in (1, 2)

    q_flat = q.reshape(batch * next_n * num_heads, head_dim).contiguous()

    # Two views of the same fused buffer: fp8 for K data, uint8 for scale bytes
    fused_kv_fp8 = fused_kv_cache.view(torch.float8_e4m3fn)
    fused_kv_u8 = fused_kv_cache
    fused_stride_block = fused_kv_cache.stride(0)
    fused_stride_token = head_dim_with_sf

    max_blocks_per_seq = block_tables.shape[1]
    logits = torch.full(
        (batch * next_n, max_seq_len),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )

    # Heuristic: multi-block when total programs is small
    batch_next_n = batch * next_n
    total_programs = batch_next_n * max_blocks_per_seq
    bpp = min(4, max_blocks_per_seq) if total_programs < 256 else 1
    grid_kv = (max_blocks_per_seq + bpp - 1) // bpp

    _fp8_paged_mqa_logits_kernel[(batch_next_n, grid_kv)](
        q_flat,
        fused_kv_fp8,
        fused_kv_u8,
        weights,
        logits,
        block_tables,
        seq_lens,
        max_blocks_per_seq=max_blocks_per_seq,
        fused_stride_block=fused_stride_block,
        fused_stride_token=fused_stride_token,
        stride_bt_b=block_tables.stride(0),
        stride_logits_b=logits.stride(0),
        NEXT_N=next_n,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_KV=block_kv,
        BLOCKS_PER_PROGRAM=bpp,
        num_warps=4,
        num_stages=1,
    )

    if clean_logits:
        positions = torch.arange(max_seq_len, device=q.device).unsqueeze(0)
        seq_lens_expanded = seq_lens.repeat_interleave(next_n).unsqueeze(1)
        logits.masked_fill_(positions >= seq_lens_expanded, 0.0)

    return logits
