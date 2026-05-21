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

"""Sparse-MLA paged attention for SM120 family (RTX PRO 6000 Blackwell).

Auto-dispatches decode-v2 (num_tokens <= 64) vs prefill (larger) internally.

Two public surfaces, mirroring the b12x_fused_moe + B12xMoEWrapper convention:

1. Functional API (:func:`sparse_mla_sm120_paged_attention`) — caller passes a
   pre-allocated ``workspace_buffer``; cudagraph-friendly when reusing the
   same buffer across calls.

2. Wrapper API (:class:`BatchSparseMLAPagedAttentionWrapper`) — class with
   pre-allocated workspace + output-LSE buffer for ``use_cuda_graph=True``
   workflows.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Optional

import torch

from .api_logging import flashinfer_api
from .jit.sparse_mla_sm120 import gen_sparse_mla_sm120_module
from .trace.templates.attention import sparse_mla_sm120_paged_trace
from .utils import (
    register_custom_op,
    register_fake_op,
    supported_compute_capability,
)

# Kernel-side constants. Mirrored from
# include/flashinfer/attention/sparse_mla_sm120/{arch,model}/*.cuh.
_HPB = 16  # heads per HPB tile (HEADS_PER_BLOCK)
_D_V = 512  # value head dim (universal across V32 and MODEL1)
_FIXED_OVERHEAD = 64  # scheduler tile-overhead constant

# Decode-v2 cutoff: num_tokens > _DECODE_MAX_TOKENS routes to the prefill
# kernel, which writes output / out_lse directly and does NOT use o_accum /
# lse_accum / sched_meta / num_splits. The wrapper's workspace_buffer is
# therefore sized for decode only — independent of prefill's token bound,
# which would otherwise blow up workspace to GB scale for vLLM-style
# max_num_batched_tokens.
_DECODE_MAX_TOKENS = 64


def _compute_num_sm_parts(num_heads: int, device: torch.device) -> int:
    """FlashMLA partitioning heuristic: ``num_SMs // replicate_h`` (s_q == 1)."""
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    replicate_h = (num_heads + _HPB - 1) // _HPB
    return max(num_sms // replicate_h, 1)


# Workspace section alignment. The combine kernel issues float4 loads against
# o_accum, so every section start must be at least 16-byte aligned; 32 bytes
# satisfies that and matches DecodingSchedMeta's __align__(32).
_WORKSPACE_ALIGN = 32


def _align_up(n: int, a: int = _WORKSPACE_ALIGN) -> int:
    return (n + a - 1) // a * a


def compute_sparse_mla_sm120_workspace_size(
    max_num_tokens: int,
    max_num_heads: int,
    d_v: int = _D_V,
    device: Optional[torch.device] = None,
) -> int:
    """Bytes required for ``workspace_buffer`` given max input bounds.

    Layout (uint8 offsets, in order, each section start aligned to 32 bytes):
        sched_meta:  num_sm_parts * 32 bytes
        num_splits:  (num_tokens + 1) * 4 bytes
        o_accum:     total_splits * num_heads * d_v * 4 bytes
        lse_accum:   total_splits * num_heads * 4 bytes

    where ``total_splits = num_tokens + num_sm_parts`` (FlashMLA's upper bound
    on per-batch splits) and ``num_sm_parts`` depends on the device's SM count.

    Returns the exact byte count needed when ``num_tokens == max_num_tokens``
    and ``num_heads == max_num_heads``. Callers planning for a range should
    pass the worst-case bounds.
    """
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    num_sm_parts = _compute_num_sm_parts(max_num_heads, device)
    total_splits = max_num_tokens + num_sm_parts

    return (
        _align_up(num_sm_parts * 32)  # DecodingSchedMeta (8x int32)
        + _align_up((max_num_tokens + 1) * 4)  # num_splits int32 prefix sum
        + _align_up(total_splits * max_num_heads * d_v * 4)  # o_accum float32
        + _align_up(total_splits * max_num_heads * 4)  # lse_accum float32
    )


def _partition_workspace(
    workspace_buffer: torch.Tensor,
    num_tokens: int,
    num_heads: int,
    d_v: int,
    num_sm_parts: int,
):
    """Carve workspace_buffer into (sched_meta, num_splits, o_accum, lse_accum)
    typed views. Each section start is 32-byte aligned (float4 + DecodingSchedMeta
    requirement). Raises if the buffer is too small for the request.
    """
    if workspace_buffer.dtype != torch.uint8 or workspace_buffer.ndim != 1:
        raise ValueError(
            "workspace_buffer must be a 1-D uint8 tensor; got "
            f"dtype={workspace_buffer.dtype}, ndim={workspace_buffer.ndim}"
        )
    total_splits = num_tokens + num_sm_parts
    sched_bytes = _align_up(num_sm_parts * 32)
    ns_bytes = _align_up((num_tokens + 1) * 4)
    oa_bytes = _align_up(total_splits * num_heads * d_v * 4)
    la_bytes = _align_up(total_splits * num_heads * 4)
    need = sched_bytes + ns_bytes + oa_bytes + la_bytes
    if workspace_buffer.numel() < need:
        raise ValueError(
            f"workspace_buffer too small: have {workspace_buffer.numel()} B, "
            f"need {need} B for num_tokens={num_tokens}, num_heads={num_heads}, "
            f"d_v={d_v}, num_sm_parts={num_sm_parts}"
        )

    # The base pointer of workspace_buffer is at least 256-byte aligned
    # (cudaMalloc default), so all section starts inherit that alignment.
    off = 0
    sched_meta = (
        workspace_buffer[off : off + num_sm_parts * 32]
        .view(torch.int32)
        .view(num_sm_parts * 8)
    )
    off += sched_bytes
    num_splits = (
        workspace_buffer[off : off + (num_tokens + 1) * 4]
        .view(torch.int32)
        .view(num_tokens + 1)
    )
    off += ns_bytes
    o_accum = (
        workspace_buffer[off : off + total_splits * num_heads * d_v * 4]
        .view(torch.float32)
        .view(total_splits, 1, num_heads, d_v)
    )
    off += oa_bytes
    lse_accum = (
        workspace_buffer[off : off + total_splits * num_heads * 4]
        .view(torch.float32)
        .view(total_splits, 1, num_heads)
    )
    return sched_meta, num_splits, o_accum, lse_accum


@functools.cache
def get_sparse_mla_sm120_module():
    """Build and cache the sparse-MLA SM120 module + bound custom op."""
    module = gen_sparse_mla_sm120_module().build_and_load()

    @register_custom_op(
        "flashinfer::sparse_mla_sm120_paged_attention",
        mutates_args=("output", "out_lse", "workspace_buffer"),
    )
    def _paged_attention(
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        output: torch.Tensor,
        out_lse: torch.Tensor,
        workspace_buffer: torch.Tensor,
        sm_scale: float,
        d_v: int,
        topk_length: Optional[torch.Tensor],
        attn_sink: Optional[torch.Tensor],
        extra_kv_cache: Optional[torch.Tensor],
        extra_indices: Optional[torch.Tensor],
        extra_topk_length: Optional[torch.Tensor],
    ) -> None:
        num_tokens, num_heads, _ = q.shape
        num_sm_parts = _compute_num_sm_parts(num_heads, q.device)
        # Decode-v2 uses the workspace partitions; prefill writes output /
        # out_lse directly and ignores them. Cap the partition request at
        # the decode cutoff so the caller's workspace_buffer only needs to
        # hold the decode-side scratch (independent of prefill's token bound).
        partition_tokens = min(num_tokens, _DECODE_MAX_TOKENS)
        sched_meta, num_splits, o_accum, lse_accum = _partition_workspace(
            workspace_buffer,
            partition_tokens,
            num_heads,
            d_v,
            num_sm_parts,
        )
        module.sparse_mla_sm120_paged_attention(
            q,
            kv_cache,
            indices,
            output,
            out_lse,
            o_accum,
            lse_accum,
            sched_meta,
            num_splits,
            sm_scale,
            num_sm_parts,
            topk_length,
            attn_sink,
            extra_kv_cache,
            extra_indices,
            extra_topk_length,
        )

    @register_fake_op("flashinfer::sparse_mla_sm120_paged_attention")
    def _fake_paged_attention(*_args, **_kwargs) -> None:
        return None

    return SimpleNamespace(paged_attention=_paged_attention)


@supported_compute_capability([120, 121])
@flashinfer_api(trace=sparse_mla_sm120_paged_trace)
def sparse_mla_sm120_paged_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    out_lse: torch.Tensor,
    workspace_buffer: torch.Tensor,
    sm_scale: float,
    *,
    d_v: int = _D_V,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> None:
    r"""Sparse-MLA paged attention on SM120 (RTX PRO 6000 Blackwell).

    Auto-dispatches decode-v2 (``num_tokens <= 64``) vs prefill (larger).
    Mutates ``output``, ``out_lse``, and ``workspace_buffer`` in place.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor, shape ``[num_tokens, num_heads, d_qk]``, dtype bf16.
        ``d_qk`` selects the model: ``576`` → V32 (DSv3.2 / GLM5.1),
        ``512`` → MODEL1 (DSv4 family).
    kv_cache : torch.Tensor
        Paged main KV cache, shape ``[num_blocks, page_block_size, 1, bytes]``
        with byte-packed FP8 inner dim.
    indices : torch.Tensor
        Paged slot IDs per query token, shape ``[num_tokens, topk]``, dtype
        int32. ``-1`` marks invalid / out-of-window slots (kernel skips).
    output : torch.Tensor
        In-place output, shape ``[num_tokens, num_heads, d_v]``, dtype bf16.
    out_lse : torch.Tensor
        In-place log-sum-exp, shape ``[num_tokens, num_heads]``, dtype float32.
    workspace_buffer : torch.Tensor
        1-D uint8 scratch buffer. Size required:
        :func:`compute_sparse_mla_sm120_workspace_size`.
    sm_scale : float
        Softmax scale (typically ``1 / sqrt(d_qk)``).
    d_v : int
        Value head dim. ``512`` for both V32 and MODEL1 today.
    topk_length : Optional[torch.Tensor]
        Effective top-k length per query token, shape ``[num_tokens]``, dtype
        int32. Required for sliding-window MLA near sequence start; ``None``
        for uniform top-k.
    attn_sink : Optional[torch.Tensor]
        Per-head learnable bias added pre-softmax, shape ``[num_heads]``,
        dtype float32. FlashMLA V4 convention: ``output *= sigmoid(lse -
        sink)`` and ``lse' = log(exp(lse) + exp(sink))``.
    extra_kv_cache : Optional[torch.Tensor]
        Optional secondary KV cache (DSv4 C4A / C128A layers). When provided,
        ``extra_indices`` must also be passed. MODEL1-only.
    extra_indices : Optional[torch.Tensor]
        Paged slot IDs for the secondary cache, shape
        ``[num_tokens, extra_topk]``, dtype int32.
    extra_topk_length : Optional[torch.Tensor]
        Effective top-k length per query token for the secondary cache,
        shape ``[num_tokens]``, dtype int32.

    Notes
    -----
    Requires SM120a / SM121a (block-scaled MXFP8 MMA + cp.async.bulk TMA).
    """
    impl = get_sparse_mla_sm120_module()
    impl.paged_attention(
        q,
        kv_cache,
        indices,
        output,
        out_lse,
        workspace_buffer,
        sm_scale,
        d_v,
        topk_length,
        attn_sink,
        extra_kv_cache,
        extra_indices,
        extra_topk_length,
    )


class BatchSparseMLAPagedAttentionWrapper:
    """Sparse-MLA paged attention wrapper for SM120 with cudagraph support.

    Pre-allocates workspace + LSE buffer at construction so :meth:`run` is
    allocation-free and safe to capture inside a CUDA graph.

    Mirrors the :class:`B12xMoEWrapper` precedent (functional API + wrapper
    class for cudagraph workflows).

    Parameters
    ----------
    max_num_tokens : int
        Worst-case ``num_tokens`` the wrapper will accept. Used to size the
        ``out_lse`` buffer. The internal ``workspace_buffer`` is sized at the
        decode cutoff (``num_tokens <= 64``) regardless — prefill writes
        ``output``/``out_lse`` directly and does not touch the workspace.
    max_num_heads : int
        Worst-case ``num_heads``. Together with the decode-cutoff token bound
        determines workspace size.
    d_v : int
        Value head dim. ``512`` for V32 / MODEL1.
    device : Optional[torch.device]
        Allocation target. Defaults to the current CUDA device.

    Example
    -------
    >>> wrapper = BatchSparseMLAPagedAttentionWrapper(
    ...     max_num_tokens=4096, max_num_heads=128
    ... )
    >>> wrapper.run(q, kv_cache, indices, output, sm_scale=...)
    """

    @supported_compute_capability([120, 121])
    @flashinfer_api
    def __init__(
        self,
        max_num_tokens: int,
        max_num_heads: int,
        *,
        d_v: int = _D_V,
        device: Optional[torch.device] = None,
    ) -> None:
        if max_num_tokens <= 0:
            raise ValueError(f"max_num_tokens must be > 0, got {max_num_tokens}")
        if max_num_heads <= 0 or max_num_heads > 128:
            raise ValueError(f"max_num_heads must be in (0, 128], got {max_num_heads}")

        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self._device = torch.device(device)
        self._max_num_tokens = max_num_tokens
        self._max_num_heads = max_num_heads
        self._d_v = d_v

        # Workspace is decode-only — sized at the decode cutoff (64 tokens)
        # regardless of max_num_tokens. Prefill (num_tokens > 64) doesn't
        # touch sched_meta / num_splits / o_accum / lse_accum.
        ws_bytes = compute_sparse_mla_sm120_workspace_size(
            max_num_tokens=_DECODE_MAX_TOKENS,
            max_num_heads=max_num_heads,
            d_v=d_v,
            device=self._device,
        )
        self._workspace_buffer = torch.empty(
            ws_bytes, dtype=torch.uint8, device=self._device
        )
        # Pre-allocated LSE buffer; sliced to actual shape on run(). Sized
        # for prefill worst case since prefill writes here too.
        self._out_lse = torch.empty(
            (max_num_tokens, max_num_heads),
            dtype=torch.float32,
            device=self._device,
        )

    @flashinfer_api(trace=sparse_mla_sm120_paged_trace)
    def run(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        output: torch.Tensor,
        sm_scale: float,
        *,
        topk_length: Optional[torch.Tensor] = None,
        attn_sink: Optional[torch.Tensor] = None,
        extra_kv_cache: Optional[torch.Tensor] = None,
        extra_indices: Optional[torch.Tensor] = None,
        extra_topk_length: Optional[torch.Tensor] = None,
        return_lse: bool = False,
    ) -> Optional[torch.Tensor]:
        """Run sparse-MLA paged attention.

        Mutates ``output`` (and the wrapper's internal LSE buffer) in place.
        When ``return_lse=True``, returns a view into the LSE buffer sized to
        the actual ``num_tokens``; otherwise returns ``None``.

        Accepts ``q``/``output`` either as 3-D ``[num_tokens, num_heads, head_dim]``
        or as 4-D ``[num_tokens, 1, num_heads, head_dim]`` (with the inner s_q=1
        dim used by some callers e.g. vLLM); the 4-D form is squeezed in place.
        """
        if q.dim() == 4:
            if q.size(1) != 1:
                raise ValueError(
                    f"4-D q is only supported with s_q=1, got q.shape={tuple(q.shape)}"
                )
            q = q.squeeze(1)
            if output.dim() == 4:
                if output.size(1) != 1:
                    raise ValueError(
                        f"4-D output is only supported with s_q=1, got "
                        f"output.shape={tuple(output.shape)}"
                    )
                output = output.squeeze(1)
        num_tokens, num_heads, _ = q.shape
        if num_tokens > self._max_num_tokens:
            raise ValueError(
                f"num_tokens ({num_tokens}) exceeds max_num_tokens "
                f"({self._max_num_tokens})"
            )
        if num_heads > self._max_num_heads:
            raise ValueError(
                f"num_heads ({num_heads}) exceeds max_num_heads ({self._max_num_heads})"
            )

        out_lse_view = self._out_lse[:num_tokens, :num_heads]
        sparse_mla_sm120_paged_attention(
            q,
            kv_cache,
            indices,
            output,
            out_lse_view,
            self._workspace_buffer,
            sm_scale,
            d_v=self._d_v,
            topk_length=topk_length,
            attn_sink=attn_sink,
            extra_kv_cache=extra_kv_cache,
            extra_indices=extra_indices,
            extra_topk_length=extra_topk_length,
        )
        return out_lse_view if return_lse else None
