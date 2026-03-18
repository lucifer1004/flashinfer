"""Fused FP8 sparse MLA decode: dequant + attention in one kernel.

Reads FP8 KV cache directly — no intermediate BF16 buffer.
Split-K for GPU occupancy, then merge.
"""

import torch

from .triton.kernels.fused_sparse_mla_decode import (
    _fused_sparse_mla_decode_kernel,
    _merge_splits_kernel,
)


def fused_sparse_mla_decode(
    q: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    sparse_indices: torch.Tensor,
    scale: float,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    group_size: int = 128,
    k_splits: int = 4,
) -> torch.Tensor:
    """Fused FP8 dequant + sparse MLA decode.

    Args:
        q: [batch, num_heads, head_dim] bf16
        kv_cache_fp8: [total_tokens, dim_quant=656] float8_e4m3fn
        sparse_indices: [batch, top_k] int32 (-1 = padding)
        scale: attention scale
        kv_lora_rank: V output dim (512)
        qk_rope_head_dim: rope dim (64)
        group_size: FP8 quant group size (128)
        k_splits: split top_k across this many programs per batch

    Returns:
        [batch, num_heads, kv_lora_rank] bf16
    """
    batch, num_heads, head_dim = q.shape
    top_k = sparse_indices.shape[1]
    dim_nope = kv_lora_rank
    dim_rope = qk_rope_head_dim
    num_groups = dim_nope // group_size

    assert q.dtype == torch.bfloat16
    assert kv_cache_fp8.dtype == torch.float8_e4m3fn
    assert num_heads >= 16, f"Need num_heads >= 16 for tensor cores, got {num_heads}"
    assert dim_nope % group_size == 0

    # Pre-slice FP8 fused buffer
    kv_nope_fp8 = kv_cache_fp8[:, :dim_nope]
    kv_scales = kv_cache_fp8[:, dim_nope:dim_nope + num_groups * 4].view(torch.float32)
    kv_rope_bf16 = kv_cache_fp8[:, dim_nope + num_groups * 4:].view(torch.bfloat16)

    q_flat = q.reshape(batch * num_heads, head_dim).contiguous()
    sparse_indices = sparse_indices.contiguous()

    tokens_per_split = (top_k + k_splits - 1) // k_splits
    # Round up to BLOCK_K
    BLOCK_K = 16
    tokens_per_split = ((tokens_per_split + BLOCK_K - 1) // BLOCK_K) * BLOCK_K

    # Allocate split-K partial buffers
    out_partial = torch.empty(batch, k_splits, num_heads, kv_lora_rank,
                              dtype=torch.float32, device=q.device)
    out_m = torch.empty(batch, k_splits, num_heads, dtype=torch.float32, device=q.device)
    out_d = torch.empty(batch, k_splits, num_heads, dtype=torch.float32, device=q.device)

    # Main kernel
    grid_main = (batch, k_splits)
    _fused_sparse_mla_decode_kernel[grid_main](
        q_flat,
        kv_nope_fp8, kv_scales, kv_rope_bf16,
        sparse_indices,
        out_partial, out_m, out_d,
        stride_nope=kv_nope_fp8.stride(0),
        stride_scales=kv_scales.stride(0),
        stride_rope=kv_rope_bf16.stride(0),
        scale=scale,
        TOP_K=top_k,
        NUM_HEADS=num_heads,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
        KV_LORA_RANK=kv_lora_rank,
        GROUP_SIZE=group_size,
        NUM_GROUPS=num_groups,
        BLOCK_K=BLOCK_K,
        K_SPLITS=k_splits,
        TOKENS_PER_SPLIT=tokens_per_split,
        num_warps=4,
        num_stages=1,
    )

    # Merge kernel
    output = torch.empty(batch * num_heads, kv_lora_rank, dtype=torch.bfloat16, device=q.device)
    BLOCK_V = 128
    grid_merge = (batch * num_heads, (kv_lora_rank + BLOCK_V - 1) // BLOCK_V)
    _merge_splits_kernel[grid_merge](
        out_partial, out_m, out_d, output,
        K_SPLITS=k_splits,
        NUM_HEADS=num_heads,
        KV_LORA_RANK=kv_lora_rank,
        BLOCK_V=BLOCK_V,
        num_warps=4,
    )

    return output.view(batch, num_heads, kv_lora_rank)
