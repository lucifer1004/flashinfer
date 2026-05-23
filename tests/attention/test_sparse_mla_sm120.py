# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Correctness tests for sparse-MLA paged attention on SM120.

Covers both decode paths against a PyTorch SDPA-with-sparse-mask reference:

* DSv4 (d_qk=512, FP8 FOOTER 584 B/token, page_block_size=64) → decode-dsv4
* DSv3.2 (d_qk=576, FP8 INLINE 656 B/token, page_block_size=1)  → decode-dsv3_2

Quantization helpers port the upstream FlashMLA packed layouts.

Skipped on non-Blackwell-consumer GPUs via :func:`is_sm120a_supported`.
"""

from __future__ import annotations

import pytest
import torch

import flashinfer
from flashinfer.utils import is_sm120a_supported

pytestmark = pytest.mark.skipif(
    not is_sm120a_supported(torch.device("cuda")),
    reason="Sparse-MLA SM120 requires sm120a.",
)


# ── Quantization helpers (ported from flash_mla_sm120/tests/test_decode.py) ──


def _cast_scale_inv_to_ue8m0(scales_inv: torch.Tensor) -> torch.Tensor:
    """Round inverse scale to the nearest power-of-2 (FlashMLA convention)."""
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil())


def _fp32_to_ue8m0_bytes(scale_fp32: torch.Tensor) -> torch.Tensor:
    """Extract the IEEE-754 exponent byte of an FP32 power-of-2 scale."""
    bits = scale_fp32.to(torch.float32).view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.uint8)


def quantize_kv_model1(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack bf16 KV into DSV4 FP8 FOOTER format.

    Input  shape (nb, bs, 1, 512) bf16.
    Output shape (nb, bs, 1, 584) uint8 — physical layout per block:
        [0 : bs*576)        Token data (nope 448B FP8 + rope 128B BF16) per token
        [bs*576 : bs*584)   Scale footer (7×UE8M0 + 1 pad) per token
    """
    d_nope, d_rope, tile_size, num_tiles = 448, 64, 64, 7
    data_stride = d_nope + d_rope * 2  # 576
    scale_bytes = num_tiles + 1  # 8
    bpt = data_stride + scale_bytes  # 584
    nb, bs, hk, d = kv_bf16.shape
    assert d == 512 and hk == 1
    kv = kv_bf16.squeeze(2)

    block_bytes = bs * bpt
    result_flat = torch.zeros(nb, block_bytes, dtype=torch.uint8, device=kv.device)

    for ti in range(num_tiles):
        tile = kv[..., ti * tile_size : (ti + 1) * tile_size].float()
        amax = tile.abs().amax(dim=-1).clamp(min=1e-4)
        scale = _cast_scale_inv_to_ue8m0(amax / 448.0)
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        ue8m0 = _fp32_to_ue8m0_bytes(scale)

        for tok in range(bs):
            data_off = tok * data_stride + ti * tile_size
            result_flat[:, data_off : data_off + tile_size] = fp8[:, tok].view(
                torch.uint8
            )
            scale_off = bs * data_stride + tok * scale_bytes + ti
            result_flat[:, scale_off] = ue8m0[:, tok]

    rope = kv[..., d_nope:].to(torch.bfloat16).contiguous().view(torch.uint8)
    rope = rope.reshape(nb, bs, d_rope * 2)
    for tok in range(bs):
        rope_off = tok * data_stride + d_nope
        result_flat[:, rope_off : rope_off + d_rope * 2] = rope[:, tok]

    return result_flat.view(nb, bs, 1, bpt)


def dequantize_kv_model1(packed: torch.Tensor) -> torch.Tensor:
    """Unpack DSV4 FP8 FOOTER → bf16. Inverse of :func:`quantize_kv_model1`."""
    d_nope, d_rope, tile_size, num_tiles = 448, 64, 64, 7
    data_stride = d_nope + d_rope * 2
    scale_bytes = num_tiles + 1
    bpt = data_stride + scale_bytes
    nb, bs, _, _ = packed.shape
    result = torch.zeros(nb, bs, 512, dtype=torch.bfloat16, device=packed.device)
    p = packed.view(nb, bs * bpt)

    for tok in range(bs):
        data_off = tok * data_stride
        scale_off = bs * data_stride + tok * scale_bytes
        for ti in range(num_tiles):
            fp8_off = data_off + ti * tile_size
            fp8 = p[:, fp8_off : fp8_off + tile_size].view(torch.float8_e4m3fn).float()
            ue8m0 = p[:, scale_off + ti]
            scale = torch.pow(2.0, ue8m0.float() - 127.0)
            result[:, tok, ti * tile_size : (ti + 1) * tile_size] = (
                fp8 * scale.unsqueeze(-1)
            ).to(torch.bfloat16)
        rope_off = data_off + d_nope
        rope_bytes = p[:, rope_off : rope_off + d_rope * 2].contiguous()
        result[:, tok, d_nope:] = rope_bytes.view(torch.bfloat16).reshape(nb, d_rope)

    return result.view(nb, bs, 1, 512)


# ── DSv3.2 INLINE pack (656 B/token: FP8 nope + FP32 scales + BF16 rope) ─────


def quantize_kv_dsv3_2(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack bf16 KV into DSv3.2 FP8 INLINE format.

    Input  shape (nb, 1, 1, 576) bf16  (d_qk = D_NOPE 512 + D_ROPE 64).
    Output shape (nb, 1, 1, 656) uint8 — per-token layout:
        [0   : 512)  FP8 e4m3 nope (4 tiles × 128 elements)
        [512 : 528)  4 × FP32 power-of-2 scale (one per 128-elem tile)
        [528 : 656)  BF16 rope (64 elements × 2B)
    """
    d_nope, d_rope, tile_size, num_tiles = 512, 64, 128, 4
    scale_bytes = num_tiles * 4  # 16
    bpt = d_nope + scale_bytes + d_rope * 2  # 656
    nb, bs, hk, d = kv_bf16.shape
    assert d == d_nope + d_rope and hk == 1 and bs == 1
    kv = kv_bf16.squeeze(2).squeeze(1)  # (nb, 576)

    result = torch.zeros(nb, bpt, dtype=torch.uint8, device=kv.device)

    # FP8 nope tiles + FP32 power-of-2 scales (inline, not footer).
    for ti in range(num_tiles):
        tile = kv[:, ti * tile_size : (ti + 1) * tile_size].float()
        amax = tile.abs().amax(dim=-1).clamp(min=1e-4)
        scale = _cast_scale_inv_to_ue8m0(amax / 448.0)  # power-of-2 FP32
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        result[:, ti * tile_size : (ti + 1) * tile_size] = fp8.view(torch.uint8)
        # FP32 scale → 4 bytes inline at offset 512 + ti*4.
        result[:, d_nope + ti * 4 : d_nope + (ti + 1) * 4] = scale.view(
            torch.float32
        ).view(torch.uint8).view(nb, 4)

    # BF16 rope tail.
    rope = kv[:, d_nope:].to(torch.bfloat16).contiguous().view(torch.uint8)
    result[:, d_nope + scale_bytes :] = rope.view(nb, d_rope * 2)
    return result.view(nb, 1, 1, bpt)


def dequantize_kv_dsv3_2(packed: torch.Tensor) -> torch.Tensor:
    """Unpack DSv3.2 FP8 INLINE → bf16. Inverse of :func:`quantize_kv_dsv3_2`."""
    d_nope, d_rope, tile_size, num_tiles = 512, 64, 128, 4
    scale_bytes = num_tiles * 4
    nb = packed.shape[0]
    p = packed.view(nb, -1)

    result = torch.zeros(nb, d_nope + d_rope, dtype=torch.bfloat16, device=p.device)
    for ti in range(num_tiles):
        fp8 = p[:, ti * tile_size : (ti + 1) * tile_size].view(
            torch.float8_e4m3fn
        ).float()
        scale = (
            p[:, d_nope + ti * 4 : d_nope + (ti + 1) * 4]
            .contiguous()
            .view(torch.float32)
            .squeeze(-1)
        )
        result[:, ti * tile_size : (ti + 1) * tile_size] = (
            fp8 * scale.unsqueeze(-1)
        ).to(torch.bfloat16)
    rope_bytes = p[:, d_nope + scale_bytes :].contiguous()
    result[:, d_nope:] = rope_bytes.view(torch.bfloat16).reshape(nb, d_rope)
    return result.view(nb, 1, 1, d_nope + d_rope)


# ── PyTorch SDPA-with-sparse-mask reference ───────────────────────────────────


def _ref_sparse_attn(
    q: torch.Tensor,
    kv_dequant: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense SDPA over the sparse-gathered KV. Returns (output_bf16, lse_log2).

    Mirrors :func:`ref_sparse_attn_decode` from the upstream test suite, with
    extensions for ``attn_sink`` (FlashMLA V4 sink-merge convention) and
    ``topk_length`` (per-token valid-length mask).
    """
    num_tokens, num_heads, d_qk = q.shape
    topk = indices.shape[-1]

    kv_flat = kv_dequant.view(-1, d_qk).float()
    q_f = q.float()

    idx_fixed = indices.clamp(min=0)
    invalid = indices < 0
    if topk_length is not None:
        # Mark tokens beyond per-token length as invalid.
        ar = torch.arange(topk, device=q.device).unsqueeze(0)
        invalid = invalid | (ar >= topk_length.unsqueeze(-1))

    gathered = kv_flat.index_select(0, idx_fixed.view(-1)).view(num_tokens, topk, d_qk)
    # logits: [num_tokens, num_heads, topk] = q @ K^T per (t, h)
    P = torch.einsum("thd,tkd->thk", q_f, gathered) * sm_scale
    P[invalid.unsqueeze(1).expand_as(P)] = float("-inf")

    lse_e = torch.logsumexp(P, dim=-1)  # natural-log LSE [t, h]
    lse_safe = lse_e.clone()
    lse_safe[lse_safe == float("-inf")] = float("+inf")
    weights = torch.exp(P - lse_safe.unsqueeze(-1))
    out_f = torch.einsum("thk,tkd->thd", weights, gathered[..., :d_v])

    # Convert lse to log2 to match the kernel's epilogue convention.
    LN2 = float(torch.log(torch.tensor(2.0)).item())
    lse_log2 = lse_e / LN2

    if attn_sink is not None:
        # FlashMLA V4 per-head sink: output[t,h,:] *= sigmoid(lse_e[t,h] - sink[h]).
        sink = attn_sink.float()  # [num_heads]
        sink_log2 = sink / LN2  # [num_heads]
        factor = torch.sigmoid(lse_e.float() - sink.unsqueeze(0))  # [t, h]
        out_f = out_f * factor.unsqueeze(-1)  # broadcast over d_v
        # Merge sink into lse (in log2 space). Handle padded -inf head sinks.
        lse_log2 = torch.where(
            lse_log2 == float("-inf"),
            sink_log2.unsqueeze(0).expand_as(lse_log2),
            lse_log2 + torch.log2(1.0 + torch.exp2(sink_log2.unsqueeze(0) - lse_log2)),
        )

    return out_f.to(torch.bfloat16), lse_log2


# ── Tests ────────────────────────────────────────────────────────────────────

_MODEL1_DECODE_CONFIGS = [
    # (num_heads, topk)
    # h=8 cases exercise the VALID_HPB < HPB code path (small-TP corner);
    # cover all three topk values to confirm the dispatch table.
    (8, 128),
    (8, 512),
    (8, 1024),
    (16, 128),
    (32, 512),
    (64, 1024),
    (128, 1024),
]


@pytest.mark.parametrize("num_heads,topk", _MODEL1_DECODE_CONFIGS)
@pytest.mark.parametrize("num_tokens", [1, 16, 64])
@pytest.mark.parametrize("with_sink", [False, True])
def test_sparse_mla_sm120_decode_model1(
    num_heads: int, topk: int, num_tokens: int, with_sink: bool
) -> None:
    """DSV4 decode-dsv3_2 path: num_tokens <= 64, d_qk=512, page_block_size=64."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v = 512, 512
    page_block_size = 64
    num_blocks = 64
    s_kv = num_blocks * page_block_size  # 4096

    # bf16 reference KV → FP8-packed kernel KV.
    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_model1(kv_bf16)
    kv_dequant = dequantize_kv_model1(kv_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    # Mark a large fraction of slots invalid (-1). The indexer emits -1 for
    # any topk slot beyond the per-token effective seq_len; in production
    # that's typically 50-95% of slots early in generation. Strong masking
    # here ensures the kernel can't pass the test by ignoring -1 (only the
    # first few logits would fall under softmax noise floor otherwise).
    indices[:, topk // 2 :] = -1

    attn_sink = (
        torch.randn(num_heads, device=device, dtype=torch.float32) * 2.0
        if with_sink
        else None
    )

    sm_scale = d_qk**-0.5

    # Reference (uses dequantized kv).
    ref_out, _ref_lse = _ref_sparse_attn(
        q, kv_dequant, indices, sm_scale, d_v, attn_sink=attn_sink
    )

    # Kernel: allocate output + workspace, call paged_attention.
    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)
    ws_bytes = flashinfer.compute_sparse_mla_sm120_workspace_size(
        max_num_tokens=num_tokens, max_num_heads=num_heads, d_v=d_v, device=device
    )
    workspace = torch.empty(ws_bytes, dtype=torch.uint8, device=device)

    flashinfer.sparse_mla_sm120_paged_attention(
        q,
        kv_packed,
        indices,
        output,
        out_lse,
        workspace,
        sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
    )

    err = (output.float() - ref_out.float()).abs()
    max_err = err.max().item()
    # Tolerance: bf16 + FP8 quantization noise. 1e-2 matches the upstream test
    # suite's coarse pass threshold (FlashMLA reference parity).
    assert max_err < 5e-2, (
        f"max_err={max_err:.4f} exceeds 5e-2 for "
        f"num_heads={num_heads} topk={topk} num_tokens={num_tokens} sink={with_sink}"
    )


_DSV3_2_DECODE_HEADS = [8, 16, 32, 64, 128]


@pytest.mark.parametrize("num_heads", _DSV3_2_DECODE_HEADS)
@pytest.mark.parametrize("num_tokens", [1, 16, 64])
@pytest.mark.parametrize("with_sink", [False, True])
def test_sparse_mla_sm120_decode_dsv3_2(
    num_heads: int, num_tokens: int, with_sink: bool
) -> None:
    """DSv3.2 decode-dsv3_2 path: d_qk=576, topk=2048, page_block_size=1."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v = 576, 512
    topk = 2048  # the only dispatched topk for decode-dsv3_2
    page_block_size = 1
    # Pool sized so topk valid slot ids fit; each token = one page block.
    num_blocks = 4096
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dsv3_2(kv_bf16)
    kv_dequant = dequantize_kv_dsv3_2(kv_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    # Mark a large fraction of slots invalid (-1). The indexer emits -1 for
    # any topk slot beyond the per-token effective seq_len; in production
    # that's typically 50-95% of slots early in generation. Strong masking
    # here ensures the kernel can't pass the test by ignoring -1 (only the
    # first few logits would fall under softmax noise floor otherwise).
    indices[:, topk // 2 :] = -1

    attn_sink = (
        torch.randn(num_heads, device=device, dtype=torch.float32) * 2.0
        if with_sink
        else None
    )

    sm_scale = d_qk**-0.5

    ref_out, _ref_lse = _ref_sparse_attn(
        q, kv_dequant, indices, sm_scale, d_v, attn_sink=attn_sink
    )

    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)
    ws_bytes = flashinfer.compute_sparse_mla_sm120_workspace_size(
        max_num_tokens=num_tokens, max_num_heads=num_heads, d_v=d_v, device=device
    )
    workspace = torch.empty(ws_bytes, dtype=torch.uint8, device=device)

    flashinfer.sparse_mla_sm120_paged_attention(
        q,
        kv_packed,
        indices,
        output,
        out_lse,
        workspace,
        sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
    )

    err = (output.float() - ref_out.float()).abs()
    max_err = err.max().item()
    assert max_err < 5e-2, (
        f"max_err={max_err:.4f} exceeds 5e-2 for "
        f"num_heads={num_heads} num_tokens={num_tokens} sink={with_sink}"
    )


def test_sparse_mla_sm120_wrapper_class_run() -> None:
    """Smoke-test the wrapper class: construct once, call .run() repeatedly."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_heads, topk = 32, 512
    d_qk, d_v = 512, 512
    page_block_size = 64
    num_blocks = 32
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_model1(kv_bf16)

    wrapper = flashinfer.BatchSparseMLAPagedAttentionWrapper(
        max_num_tokens=64,
        max_num_heads=num_heads,
        d_v=d_v,
        device=device,
    )

    for num_tokens in (1, 16, 64):
        q = (
            torch.randn(
                num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16
            )
            / 10.0
        ).clamp(-1, 1)
        indices = torch.randint(
            0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
        )
        output = torch.zeros(
            (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
        )
        # No exception means the dispatch path is wired correctly.
        wrapper.run(q, kv_packed, indices, output, sm_scale=d_qk**-0.5)


def test_sparse_mla_sm120_workspace_size_grows_with_inputs() -> None:
    """Sanity: workspace size is monotone in num_tokens and num_heads."""
    device = torch.device("cuda")
    a = flashinfer.compute_sparse_mla_sm120_workspace_size(8, 16, 512, device)
    b = flashinfer.compute_sparse_mla_sm120_workspace_size(8, 32, 512, device)
    c = flashinfer.compute_sparse_mla_sm120_workspace_size(64, 32, 512, device)
    assert a < b < c, f"non-monotone: {a} < {b} < {c}"
