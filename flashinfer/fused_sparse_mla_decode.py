"""Fused FP8 sparse MLA decode: dequant + attention in one kernel.

Reads FP8 KV cache directly — no intermediate BF16 buffer.
Adaptive split-K for GPU occupancy, then merge.
"""

import torch

from .triton.kernels.fused_sparse_mla_decode import (
    _fused_sparse_mla_decode_kernel,
    _merge_splits_kernel,
)


def _compute_k_splits(
    batch: int,
    top_k: int,
    block_k: int = 16,
    num_sms: int = 0,
    min_iters_per_split: int = 4,
) -> int:
    """Choose k_splits to fill ~2 waves on the GPU.

    Targets 2 × num_SMs total programs = batch × k_splits.
    Keeps ≥ min_iters_per_split BLOCK_K iterations per split for compute
    amortization.  Result is rounded to a power of 2 to limit Triton JIT
    recompilation variants.
    """
    if num_sms <= 0:
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    target_programs = 2 * num_sms
    min_tokens_per_split = min_iters_per_split * block_k
    max_k_splits = max(1, top_k // min_tokens_per_split)
    k_splits = min(max_k_splits, max(1, target_programs // batch))
    # Clamp to actual useful splits (no empty splits)
    k_splits = min(k_splits, (top_k + block_k - 1) // block_k)
    k_splits = max(1, k_splits)
    # Round up to next power of 2 to limit JIT variants, capped at max_k_splits
    if k_splits > 1:
        k_splits = min(max_k_splits, 1 << (k_splits - 1).bit_length())
    return k_splits


def fused_sparse_mla_decode(
    q: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    sparse_indices: torch.Tensor,
    scale: float,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    group_size: int = 128,
    k_splits: int | None = None,
    num_warps: int | None = None,
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
        k_splits: split top_k across this many programs per batch.
                  None = auto-tune based on batch size and GPU topology.
        num_warps: Triton warps per program. None = auto (4 for small grids,
                   2 for large grids where occupancy matters more than
                   per-program throughput).

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

    if k_splits is None:
        k_splits = _compute_k_splits(batch, top_k)

    grid_size = batch * k_splits
    if num_warps is None:
        # num_warps=2: lower register pressure → more concurrent programs per SM.
        # Better when grid is large enough to saturate (≥2 waves).
        # num_warps=4: higher per-program throughput for tensor cores.
        # Better when grid is small and each program must do more work.
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        num_warps = 2 if grid_size >= 2 * num_sms else 4

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
        num_warps=num_warps,
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
        num_warps=num_warps,
    )

    return output.view(batch, num_heads, kv_lora_rank)
