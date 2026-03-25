"""Sparse MLA prefill: head-tiled kernel for large num_q.

For prefill (num_q >> 64), this is much faster than fused_sparse_mla_decode
which processes all 128 heads per program.
"""

import torch
from .triton.kernels.sparse_mla_prefill import _sparse_mla_prefill_kernel


def sparse_mla_prefill(
    q: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    sparse_indices: torch.Tensor,
    scale: float,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    group_size: int = 128,
) -> torch.Tensor:
    """Sparse MLA prefill with head-tiled grid.

    Args:
        q: [num_q, num_heads, head_dim] bf16
        kv_cache_fp8: [total_tokens, dim_total] fp8 (fused layout)
        sparse_indices: [num_q, top_k] int32
        scale: attention scale
    Returns:
        [num_q, num_heads, kv_lora_rank] bf16
    """
    num_q, num_heads, head_dim = q.shape
    top_k = sparse_indices.shape[1]
    dim_nope = kv_lora_rank
    dim_rope = qk_rope_head_dim
    num_groups = dim_nope // group_size

    assert q.dtype == torch.bfloat16
    assert kv_cache_fp8.dtype == torch.float8_e4m3fn

    # Pre-slice fused buffer
    kv_nope_fp8 = kv_cache_fp8[:, :dim_nope]
    kv_scales = kv_cache_fp8[:, dim_nope:dim_nope + num_groups * 4].view(torch.float32)
    kv_rope_bf16 = kv_cache_fp8[:, dim_nope + num_groups * 4:].view(torch.bfloat16)

    q_flat = q.reshape(num_q * num_heads, head_dim).contiguous()
    output = torch.empty(num_q * num_heads, kv_lora_rank, dtype=torch.bfloat16, device=q.device)

    HPG = 16  # heads per group — tile size for head dimension
    assert num_heads % HPG == 0
    num_head_tiles = num_heads // HPG

    BLOCK_K = 16
    grid = (num_q, num_head_tiles)

    _sparse_mla_prefill_kernel[grid](
        q_flat,
        kv_nope_fp8, kv_scales, kv_rope_bf16,
        sparse_indices,
        output,
        stride_nope=kv_nope_fp8.stride(0),
        stride_scales=kv_scales.stride(0),
        stride_rope=kv_rope_bf16.stride(0),
        stride_idx=sparse_indices.stride(0),
        scale=scale,
        NUM_HEADS=num_heads,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
        KV_LORA_RANK=kv_lora_rank,
        GROUP_SIZE=group_size,
        HPG=HPG,
        BLOCK_K=BLOCK_K,
        TOP_K=top_k,
        num_warps=4,
        num_stages=2,
    )

    return output.view(num_q, num_heads, kv_lora_rank)
