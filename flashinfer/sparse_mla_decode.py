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

"""Sparse MLA decode attention for NSA (Native Sparse Attention).

Drop-in replacement for the trtllm-gen FMHA sparse MLA path on SM120
(Blackwell RTX).  Pure Triton — no architecture-specific cubins.
"""

import math

import torch

from .triton.kernels.sparse_mla_decode import (
    _sparse_mla_decode_fp8_kernel,
    _sparse_mla_decode_kernel,
)


def _next_power_of_2(n: int) -> int:
    return 1 << math.ceil(math.log2(max(n, 1)))


def sparse_mla_decode_fp8(
    q: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    sparse_indices: torch.Tensor,
    scale: float,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    group_size: int = 128,
) -> torch.Tensor:
    """Fused FP8 dequant + sparse MLA decode attention.

    Reads the FP8 fused KV cache directly — no separate dequant pass.

    Args:
        q: [batch, num_heads, head_dim], bf16.  head_dim = kv_lora_rank + qk_rope_head_dim.
        kv_cache_fp8: [total_tokens, dim_quant], float8_e4m3fn.
            Fused layout: [nope_fp8(512) | scales_f32(16B) | rope_bf16(128B)] = 656 elements.
        sparse_indices: [batch, top_k], int32.  -1 = padding.
        scale: Attention scale.
        kv_lora_rank: CKV dimension (512).
        qk_rope_head_dim: RoPE dimension (64).
        group_size: FP8 quantization group size (128).

    Returns:
        [batch, num_heads, kv_lora_rank], bf16.
    """
    batch, num_heads, head_dim = q.shape
    top_k = sparse_indices.shape[1]
    dim_nope = kv_lora_rank
    dim_rope = qk_rope_head_dim
    num_tiles = dim_nope // group_size

    assert q.dtype == torch.bfloat16
    assert kv_cache_fp8.dtype == torch.float8_e4m3fn
    assert head_dim == dim_nope + dim_rope
    expected_dim = dim_nope + num_tiles * 4 + dim_rope * 2
    assert kv_cache_fp8.shape[1] == expected_dim, (
        f"Expected kv_cache dim {expected_dim}, got {kv_cache_fp8.shape[1]}"
    )

    # Pre-slice the fused buffer into typed views (zero-copy)
    kv_nope_fp8 = kv_cache_fp8[:, :dim_nope]                                      # [N, 512] fp8
    kv_scales = kv_cache_fp8[:, dim_nope:dim_nope + num_tiles * 4].view(torch.float32)  # [N, 4] f32
    kv_rope_bf16 = kv_cache_fp8[:, dim_nope + num_tiles * 4:].view(torch.bfloat16)     # [N, 64] bf16

    q_flat = q.reshape(batch * num_heads, head_dim).contiguous()
    sparse_indices = sparse_indices.contiguous()
    output = torch.empty(batch * num_heads, kv_lora_rank, dtype=torch.bfloat16, device=q.device)

    PADDED_V = _next_power_of_2(kv_lora_rank)
    BLOCK_K = 16

    grid = (batch * num_heads,)
    _sparse_mla_decode_fp8_kernel[grid](
        q_flat,
        kv_nope_fp8,
        kv_scales,
        kv_rope_bf16,
        sparse_indices,
        output,
        stride_nope=kv_nope_fp8.stride(0),
        stride_scales=kv_scales.stride(0),
        stride_rope=kv_rope_bf16.stride(0),
        scale=scale,
        TOP_K=top_k,
        NUM_HEADS=num_heads,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
        KV_LORA_RANK=kv_lora_rank,
        PADDED_V=PADDED_V,
        GROUP_SIZE=group_size,
        NUM_NOPE_TILES=num_tiles,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=1,
    )

    return output.view(batch, num_heads, kv_lora_rank)


def sparse_mla_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    scale: float,
    kv_lora_rank: int,
) -> torch.Tensor:
    """BF16 sparse MLA decode (fallback / testing).

    Args:
        q: [batch, num_heads, head_dim], bf16.
        kv_cache: [total_tokens, kv_dim], bf16.
        sparse_indices: [batch, top_k], int32.
        scale: Attention scale.
        kv_lora_rank: V output dim.

    Returns:
        [batch, num_heads, kv_lora_rank], bf16.
    """
    batch, num_heads, head_dim = q.shape
    top_k = sparse_indices.shape[1]

    assert q.dtype == torch.bfloat16
    assert kv_cache.dtype == torch.bfloat16

    q_flat = q.reshape(batch * num_heads, head_dim).contiguous()
    sparse_indices = sparse_indices.contiguous()
    output = torch.empty(batch * num_heads, kv_lora_rank, dtype=torch.bfloat16, device=q.device)

    BLOCK_K = 16
    BLOCK_D = 64
    assert head_dim % BLOCK_D == 0
    PADDED_V = _next_power_of_2(kv_lora_rank)

    grid = (batch * num_heads,)
    _sparse_mla_decode_kernel[grid](
        q_flat, kv_cache, sparse_indices, output,
        stride_kv=kv_cache.stride(0),
        scale=scale,
        TOP_K=top_k,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        KV_LORA_RANK=kv_lora_rank,
        PADDED_V=PADDED_V,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        NUM_D_TILES=head_dim // BLOCK_D,
        num_warps=4,
        num_stages=1,
    )

    return output.view(batch, num_heads, kv_lora_rank)
