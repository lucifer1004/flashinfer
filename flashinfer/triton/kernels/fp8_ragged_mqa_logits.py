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

"""FP8 ragged MQA logits kernel for NSA indexer (prefill path).

Drop-in replacement for deep_gemm.fp8_mqa_logits on SM120 (Blackwell RTX).

Each Q token attends to a contiguous range [ks[i], ke[i]) of the K buffer.
K is pre-split: k_fp8 [total_kv, head_dim] and k_scale [total_kv, num_groups].

Computation per Q token i, KV token j in [ks[i], ke[i]):
  raw_score[h, j] = Q[i, h, :] · (K_fp8[j, :] * scale[j])   (FP8 dequant dot)
  logit[i, j] = sum_h(weight[i, h] * relu(raw_score[h, j]))

Grid: (num_q_tokens, ceil(max_kv_len / (BLOCK_KV * KV_GROUPS)))
"""

import triton
import triton.language as tl


@triton.jit
def _fp8_ragged_mqa_logits_kernel(
    Q,              # [num_q * num_heads, head_dim] fp8
    K_FP8,          # [total_kv, head_dim] fp8
    K_SCALE,        # [total_kv, num_scale_groups] fp32
    WEIGHTS,        # [num_q, num_heads] fp32
    LOGITS,         # [num_q, total_kv] fp32 (output)
    KS,             # [num_q] int32 — KV range start per Q token
    KE,             # [num_q] int32 — KV range end per Q token
    stride_k,       # K_FP8 token stride
    stride_ks,      # K_SCALE token stride
    stride_logits,  # LOGITS row stride (= total_kv)
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    KV_GROUPS: tl.constexpr,
):
    """FP8 ragged MQA logits: one program per (Q token, KV group)."""
    pid_q = tl.program_id(0)
    pid_kv_group = tl.program_id(1)

    kv_start = tl.load(KS + pid_q)
    kv_end = tl.load(KE + pid_q)
    kv_len = kv_end - kv_start

    first_kv = pid_kv_group * KV_GROUPS * BLOCK_KV
    if first_kv >= kv_len:
        return

    token_offsets = tl.arange(0, BLOCK_KV)
    dim_offsets = tl.arange(0, HEAD_DIM)
    head_offsets = tl.arange(0, NUM_HEADS)

    # Load Q once: [NUM_HEADS, HEAD_DIM] as FP8
    q_ptrs = Q + (pid_q * NUM_HEADS + head_offsets[:, None]) * HEAD_DIM + dim_offsets[None, :]
    q_fp8 = tl.load(q_ptrs)

    # Load weights once: [NUM_HEADS]
    weights = tl.load(WEIGHTS + pid_q * NUM_HEADS + head_offsets)

    # Process KV_GROUPS consecutive KV blocks
    for grp in range(KV_GROUPS):
        kv_offset = first_kv + grp * BLOCK_KV

        if kv_offset < kv_len:
            abs_kv = kv_start + kv_offset  # absolute position in K buffer
            valid_tokens = tl.minimum(BLOCK_KV, kv_len - kv_offset)

            # Load K: [BLOCK_KV, HEAD_DIM] FP8
            k_ptrs = K_FP8 + (abs_kv + token_offsets[:, None]) * stride_k + dim_offsets[None, :]
            k_mask = token_offsets[:, None] < valid_tokens
            k_fp8 = tl.load(k_ptrs, mask=k_mask, other=0.0)

            # Q @ K^T → [NUM_HEADS, BLOCK_KV] via tensor cores
            s = tl.dot(q_fp8, tl.trans(k_fp8), out_dtype=tl.float32)

            # Load per-token scale: [BLOCK_KV] (single scale per token for indexer K)
            scales = tl.load(
                K_SCALE + (abs_kv + token_offsets) * stride_ks,
                mask=token_offsets < valid_tokens, other=0.0,
            )

            # Scale, ReLU, weight-reduce → [BLOCK_KV]
            s = s * scales[None, :]
            s = tl.maximum(s, 0.0)
            s = s * weights[:, None]
            logits_out = tl.sum(s, axis=0)  # [BLOCK_KV]

            # Store to output at absolute KV position
            out_ptrs = LOGITS + pid_q * stride_logits + abs_kv + token_offsets
            tl.store(out_ptrs, logits_out, mask=token_offsets < valid_tokens)
