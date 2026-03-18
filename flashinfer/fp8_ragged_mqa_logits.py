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

"""FP8 ragged MQA logits for NSA indexer (prefill path).

Drop-in replacement for deep_gemm.fp8_mqa_logits on SM120 (Blackwell).
"""

import torch

from .triton.kernels.fp8_ragged_mqa_logits import _fp8_ragged_mqa_logits_kernel


def fp8_ragged_mqa_logits(
    q: torch.Tensor,
    kv_fp8: tuple,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    clean_logits: bool = False,
) -> torch.Tensor:
    """Compute FP8 ragged MQA logits for NSA indexer.

    API-compatible with deep_gemm.fp8_mqa_logits.

    Args:
        q: [num_q, next_n, num_heads, head_dim], float8_e4m3fn.
            next_n is always 1 for the current NSA indexer.
        kv_fp8: tuple of (k_fp8, k_scale):
            k_fp8:  [total_kv, head_dim], float8_e4m3fn — contiguous K buffer
            k_scale: [total_kv, num_scale_groups], float32 — per-token scales
        weights: [num_q * next_n, num_heads], float32.
        ks: [num_q * next_n], int32 — KV start index per Q token.
        ke: [num_q * next_n], int32 — KV end index per Q token.
        clean_logits: Zero positions outside each Q token's KV range.

    Returns:
        logits: [num_q * next_n, total_kv], float32.
    """
    # Accept both 3D [num_q, num_heads, head_dim] and 4D [num_q, next_n, num_heads, head_dim]
    if q.ndim == 3:
        q = q.unsqueeze(1)  # [num_q, 1, num_heads, head_dim]
    num_q, next_n, num_heads, head_dim = q.shape
    k_fp8, k_scale = kv_fp8
    total_kv = k_fp8.shape[0]

    assert q.dtype == torch.float8_e4m3fn
    assert k_fp8.dtype == torch.float8_e4m3fn
    assert k_scale.dtype == torch.float32
    assert weights.dtype == torch.float32
    assert next_n in (1, 2)

    num_q_total = num_q * next_n

    # Flatten Q: [num_q * next_n * num_heads, head_dim]
    q_flat = q.reshape(num_q_total * num_heads, head_dim).contiguous()

    logits = torch.full(
        (num_q_total, total_kv),
        float("-inf") if clean_logits else 0.0,
        dtype=torch.float32,
        device=q.device,
    )

    if num_q_total == 0 or total_kv == 0:
        return logits

    # Compute max KV range for grid sizing
    max_kv_len = (ke - ks).max().item()
    BLOCK_KV = 64
    KV_GROUPS = min(4, (max_kv_len + BLOCK_KV - 1) // BLOCK_KV) if max_kv_len > 0 else 1
    grid_kv = (max_kv_len + BLOCK_KV * KV_GROUPS - 1) // (BLOCK_KV * KV_GROUPS)

    grid = (num_q_total, grid_kv)
    _fp8_ragged_mqa_logits_kernel[grid](
        q_flat,
        k_fp8,
        k_scale,
        weights,
        logits,
        ks,
        ke,
        stride_k=k_fp8.stride(0),
        stride_ks=k_scale.stride(0),
        stride_logits=logits.stride(0),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_KV=BLOCK_KV,
        KV_GROUPS=KV_GROUPS,
        num_warps=4,
        num_stages=1,
    )

    if clean_logits:
        positions = torch.arange(total_kv, device=q.device).unsqueeze(0)
        ks_exp = ks.unsqueeze(1)
        ke_exp = ke.unsqueeze(1)
        logits.masked_fill_((positions < ks_exp) | (positions >= ke_exp), 0.0)

    return logits
