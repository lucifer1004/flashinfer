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

"""Sparse-MLA paged attention for SM120.

Auto-dispatches between decode (num_tokens <= 64) and prefill (larger). For
decode, DSv4 routes to the warp-specialized decode-dsv4 kernel and DSv3.2
routes to the scheduler-driven decode-dsv3_2 kernel.

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
import os
from types import SimpleNamespace
from typing import List, Optional

import torch

from .api_logging import flashinfer_api
from .autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
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
_D_V = 512  # value head dim (universal across DSV3_2 and DSV4)
_BI = 64  # KV partition tile size in candidates (BLOCK_SIZE_N)
_MAX_OCCUPANCY = 2  # max additional waves of split-K parallelism beyond baseline
_FIXED_OVERHEAD = 64  # scheduler tile-overhead constant

# Decode/prefill cutoff: num_tokens > _DECODE_MAX_TOKENS routes to the
# prefill kernel, which writes output / out_lse directly and does NOT use
# the o_accum / lse_accum / sched_meta / num_splits workspace sections.
# The wrapper's workspace_buffer is therefore sized for decode only,
# independent of the (much larger) prefill token bound.
_DECODE_MAX_TOKENS = 64

# decode-dsv4 supports a fixed (num_heads, topk) dispatch table. Outside of
# this set the orchestrator falls back to decode-dsv3_2 / prefill. DSV4 only.
#
# num_heads=8 is the small-TP corner case (e.g. TP=16 on 128 KV heads): the
# kernel internally pads the head tile to HPB=16 with zero-Q rows and guards
# mid_out / mid_lse writes to NUM_HEADS, so only the 8 valid heads land in
# the output.
_DECODE_DSV4_DISPATCH = frozenset({
    (8, 128), (8, 512), (8, 1024),
    (16, 128), (16, 512), (16, 1024),
    (32, 128), (32, 512), (32, 1024),
    (64, 128), (64, 512), (64, 1024),
    (128, 128), (128, 512), (128, 1024),
})
_DECODE_DSV4_PAGE_BLOCK_SIZE = 64

# decode-dsv3_2-v2 (V32 family, V4-style warp-spec). Dispatch envelope spans
# the production GLM-5.1 / Kimi K2.5 / DSv3.2 shapes (num_heads ∈
# {8,16,32,64,128} covers TP={16,8,4,2,1}; topk ∈ {128, 512, 1024, 2048}).
# v2 is the default V32 decode kernel; v1 (scheduler-driven) remains
# available as a legacy fallback via FLASHINFER_DSV3_2_KERNEL=v1 — kept for
# regression bisection and pbs=1 inputs that v2 doesn't support.
_DECODE_DSV3_2_V2_DISPATCH = frozenset({
    (8, 128), (8, 512), (8, 1024), (8, 2048),
    (16, 128), (16, 512), (16, 1024), (16, 2048),
    (32, 128), (32, 512), (32, 1024), (32, 2048),
    (64, 128), (64, 512), (64, 1024), (64, 2048),
    (128, 128), (128, 512), (128, 1024), (128, 2048),
})
# Production page_block_size for V32 indexer caches. vLLM CUDA forces 64;
# ROCm-style pbs=1 is not supported by v2 (fall through to v1).
_DECODE_DSV3_2_V2_PAGE_BLOCK_SIZE = 64


def _decode_dsv3_2_v2_disabled() -> bool:
    """True iff FLASHINFER_DSV3_2_KERNEL=v1 forces the legacy path."""
    return os.environ.get("FLASHINFER_DSV3_2_KERNEL", "v2").lower() == "v1"


def _decode_dsv3_2_v2_dispatchable(
    num_tokens: int, num_heads: int, topk: int, d_qk: int, page_block_size: int
) -> bool:
    """True iff decode-dsv3_2-v2 supports this shape configuration."""
    return (
        num_tokens <= _DECODE_MAX_TOKENS
        and d_qk == 576
        and page_block_size == _DECODE_DSV3_2_V2_PAGE_BLOCK_SIZE
        and (num_heads, topk) in _DECODE_DSV3_2_V2_DISPATCH
    )


def _decode_dsv4_dispatchable(num_tokens: int, num_heads: int, topk: int, d_qk: int,
                            page_block_size: int) -> bool:
    """Return True iff decode-dsv4 supports this shape configuration.

    decode-dsv4 is DSv4-only (d_qk=512) with a fixed (num_heads, topk)
    instantiation set and PAGE_BLOCK_SIZE=64. Outside this envelope the
    orchestrator routes to decode-dsv3_2 (V32-only) or prefill.
    """
    return (
        num_tokens <= _DECODE_MAX_TOKENS
        and d_qk == 512
        and page_block_size == _DECODE_DSV4_PAGE_BLOCK_SIZE
        and (num_heads, topk) in _DECODE_DSV4_DISPATCH
    )


def _compute_num_sm_parts(
    num_heads: int,
    device: torch.device,
    num_tokens: Optional[int] = None,
    topk: Optional[int] = None,
) -> int:
    """Choose split-K partition count for decode-dsv3_2.

    Without shape (``num_tokens``/``topk`` = None): returns the FlashMLA
    baseline so the wrapper can size its workspace for any later runtime call.

    With shape: pick the partition count that maximises throughput by
    weighing two effects in the kernel:

    1. The decode-dsv3_2 kernel has a bf16-direct-write fast path
       (``is_no_split=true``, skips f32 o_accum staging and the combine
       kernel) whenever a partition fully covers each of its batches with
       no neighbour-partition split. For ``s_q == 1`` callers this fires
       when ``num_sm_parts <= num_tokens`` and per-batch work fits in
       each partition.
    2. Grid size = ``replicate_h × num_sm_parts``; falling below ~half the
       SM count underfills the GPU and dominates any bf16-path savings.

    The chosen rule: prefer ``min(num_tokens, baseline)`` (no-split path)
    when it still fills at least half the SMs across the head-tile dim,
    else fall back to the FlashMLA baseline. ``num_tokens == 1`` is a
    common decode path; we take the no-split path unconditionally there
    since the per-token launch overhead dominates even when underfilled.
    """
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    replicate_h = (num_heads + _HPB - 1) // _HPB
    baseline = max(num_sms // replicate_h, 1)
    if num_tokens is None or topk is None:
        return baseline
    if num_tokens == 1:
        return 1
    no_split = min(num_tokens, baseline)
    if no_split * replicate_h >= num_sms // 2:
        return no_split
    return baseline


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
        num_tokens, num_heads, d_qk = q.shape
        topk = indices.shape[-1]

        # decode-dsv4 fast path. Only dispatch when the (num_heads, topk) pair
        # is in DSv4's instantiation set and the model matches DSv4's layout;
        # otherwise fall through to decode-dsv3_2 / prefill.
        # kv_cache layout: [num_blocks, page_block_size, 1, bytes_per_token].
        kv_pbs = int(kv_cache.size(-3)) if kv_cache.ndim >= 3 else 0
        if (kv_pbs == _DECODE_DSV4_PAGE_BLOCK_SIZE
                and _decode_dsv4_dispatchable(num_tokens, num_heads, topk, d_qk, kv_pbs)):
            # mid_out / mid_lse scratch is small enough to allocate per call
            # (could be carved from workspace_buffer later for cudagraph
            # alloc-free reuse if it ever shows up in profiling).
            num_splits_main = (topk + _BI - 1) // _BI
            extra_topk = (
                int(extra_indices.size(-1)) if extra_indices is not None else 0
            )
            num_splits_extra = (extra_topk + _BI - 1) // _BI
            num_splits = num_splits_main + num_splits_extra
            mid_out = torch.empty(
                (num_tokens, num_heads, num_splits, d_v),
                dtype=torch.bfloat16,
                device=q.device,
            )
            mid_lse = torch.empty(
                (num_tokens, num_heads, num_splits),
                dtype=torch.float32,
                device=q.device,
            )
            # Pass kv_cache as-is — both 4D paged layouts (with possibly
            # padded block stride) and 2D microbench layouts are supported;
            # the FFI binding extracts the true block stride from .stride(0).
            sparse_mla_sm120_decode_dsv4(
                q,
                kv_cache,
                indices,
                mid_out,
                mid_lse,
                output,
                out_lse,
                sm_scale,
                topk_length=topk_length,
                attn_sink=attn_sink,
                extra_kv_cache=extra_kv_cache,
                extra_indices=extra_indices,
                extra_topk_length=extra_topk_length,
            )
            return

        # decode-dsv3_2-v2 fast path. V32 family (d_qk=576), V4-style warp-spec
        # standalone kernel — the default V32 decode path. Falls through to
        # the legacy v1 orchestrator below for unsupported shapes or when
        # FLASHINFER_DSV3_2_KERNEL=v1 forces it.
        if (
            not _decode_dsv3_2_v2_disabled()
            and _decode_dsv3_2_v2_dispatchable(num_tokens, num_heads, topk, d_qk, kv_pbs)
        ):
            num_splits = (topk + _BI - 1) // _BI
            mid_out = torch.empty(
                (num_tokens, num_heads, num_splits, d_v),
                dtype=torch.bfloat16,
                device=q.device,
            )
            mid_lse = torch.empty(
                (num_tokens, num_heads, num_splits),
                dtype=torch.float32,
                device=q.device,
            )
            module.sparse_mla_sm120_decode_dsv3_2_v2(
                q,
                kv_cache,
                indices,
                mid_out,
                mid_lse,
                output,
                out_lse,
                num_splits,
                sm_scale,
                topk_length,
                attn_sink,
                -1,  # chunks_per_block_override = -1 → C++ heuristic
            )
            return

        # decode-dsv3_2 / prefill fallback path.
        num_sm_parts = _compute_num_sm_parts(num_heads, q.device, num_tokens, topk)
        # The decode-dsv3_2 path uses the workspace partitions; prefill writes
        # output / out_lse directly and ignores them. Cap the partition
        # request at the decode cutoff so the caller's workspace_buffer only
        # needs to hold the decode-side scratch.
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
    r"""Sparse-MLA paged attention on SM120.

    Auto-dispatches decode (``num_tokens <= 64``) vs prefill (larger).
    Mutates ``output``, ``out_lse``, and ``workspace_buffer`` in place.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor, shape ``[num_tokens, num_heads, d_qk]``, dtype bf16.
        ``d_qk`` selects the model: ``576`` → DSV3_2 (DSv3.2 / GLM5.1),
        ``512`` → DSV4 (DSv4 family).
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
        Value head dim. ``512`` for both DSV3_2 and DSV4 today.
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
        ``extra_indices`` must also be passed. DSV4-only.
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
        Value head dim. ``512`` for DSV3_2 / DSV4.
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
        or as 4-D ``[num_tokens, 1, num_heads, head_dim]`` (some callers carry
        a singleton s_q dim); the 4-D form is squeezed in place.
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


# ─────────────────────────────────────────────────────────────────────
# Decode-DSv4: AutoTuner-driven chunks_per_block tuning
# ─────────────────────────────────────────────────────────────────────
#
# decode-dsv4 is the split-K decode kernel where each block handles
# `chunks_per_block` chunks of `_BI`=64 candidates each. The wall-time-optimal
# cpb is non-monotonic in (num_tokens, num_heads, topk): per-shape sweep shows
# 20-28% gains over the C++ closed-form heuristic on contested shapes (e.g.,
# 128/512/T=16, 128/1024/T=16, 64/1024/T=16, 64/512/T=16) — the gain is
# finicky (cpb=13 vs cpb=14 differ ~12% on one shape) so a closed-form is
# unlikely to capture it without ML-style fitting.
#
# Solution: expose `chunks_per_block` as a TunableRunner tactic and let
# AutoTuner cache the best value per (T_bucket, num_heads, topk).


@functools.cache
def _get_sparse_mla_decode_dsv4_module():
    """Build + cache the decode-dsv4 module and its TunableRunner class."""
    module = gen_sparse_mla_sm120_module().build_and_load()

    class SparseMlaDecodeV3Runner(TunableRunner):
        """One runner per (kernel module). Tactic = chunks_per_block ∈
        [1, num_splits]. tactic=-1 (or 0) falls back to the C++ heuristic."""

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            indices = inputs[2]
            topk = indices.shape[1]
            num_splits = (topk + _BI - 1) // _BI
            # tactic encodes chunks_per_block (1..num_splits). We include
            # 0 as a synonym for "use heuristic" so the autotuner can fall
            # back if all real tactics are slower than heuristic.
            return list(range(1, num_splits + 1))

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            q, kv_cache, indices, mid_out, mid_lse, output, out_lse = inputs
            sm_scale = kwargs["sm_scale"]
            topk_length = kwargs.get("topk_length")
            attn_sink = kwargs.get("attn_sink")
            extra_kv_cache = kwargs.get("extra_kv_cache")
            extra_indices = kwargs.get("extra_indices")
            extra_topk_length = kwargs.get("extra_topk_length")
            topk = indices.shape[-1]  # 2D [T, topk] or 3D [T, 1, topk]
            # Total num_splits = main chunks + extra chunks. The kernel grid
            # spans all of them; mid_out is sized for this combined total by
            # the caller. Computing only main here would silently undersize
            # the kernel launch and skip the extra section.
            extra_topk = extra_indices.shape[-1] if extra_indices is not None else 0
            num_splits = (topk + _BI - 1) // _BI + (extra_topk + _BI - 1) // _BI
            # tactic ∈ [1, num_splits] → pass through; tactic == -1 (autotuner
            # fallback) → pass -1 so the C++ heuristic picks cpb.
            cpb_override = tactic if tactic > 0 else -1
            module.sparse_mla_sm120_decode_dsv4(
                q,
                kv_cache,
                indices,
                mid_out,
                mid_lse,
                output,
                out_lse,
                num_splits,
                sm_scale,
                topk_length,
                attn_sink,
                extra_kv_cache,
                extra_indices,
                extra_topk_length,
                cpb_override,
            )
            return output

    return SimpleNamespace(module=module, runner_cls=SparseMlaDecodeV3Runner)


def _decode_dsv4_num_token_buckets(*_args, **_kwargs):
    """Power-of-2-ish T buckets matching the contested decode shapes."""
    return (1, 4, 8, 16, 32, 64)


def _decode_dsv4_map_to_token_bucket(x):
    """Round T up to the next bucket boundary used by tuning."""
    buckets = (1, 4, 8, 16, 32, 64)
    for b in buckets:
        if x <= b:
            return b
    return buckets[-1]


def _decode_dsv4_init_q(shapes, dtype, device):
    """bf16 q ~ N(0, 0.1) clamped to [-1, 1] — matches the unit test distribution."""
    return (torch.randn(shapes, device=device, dtype=torch.float32) / 10.0).clamp(-1, 1).to(dtype)


def _decode_dsv4_init_indices(shapes, dtype, device):
    """int32 indices in a small safe range; assumes kv_cache has >=256 blocks.

    AutoTuner only profiles wall time, not correctness — random valid indices
    are sufficient. The cache built for the real call uses the ACTUAL indices.
    """
    return torch.randint(0, 256, shapes, dtype=dtype, device=device)


@functools.cache
def _decode_dsv4_tuning_config() -> TuningConfig:
    """Build + cache the static TuningConfig once per process.

    Avoids dataclass instantiation + Spec construction on every call into
    sparse_mla_sm120_decode_dsv4 — the config is shape-independent (depends
    only on tactic semantics + bucket scheme).
    """
    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0, 2),
                dim_idx=(0, 0),
                gen_tuning_buckets=_decode_dsv4_num_token_buckets,
                map_to_tuning_buckets=_decode_dsv4_map_to_token_bucket,
                tensor_initializers=[_decode_dsv4_init_q, _decode_dsv4_init_indices],
            ),
        ),
        # Constrain T (dim 0) of all output/scratch tensors to q's T so the
        # autotuner's synthesised q propagates to mid_out (3), mid_lse (4),
        # output (5), out_lse (6). Without these constraints, the kernel
        # writes past the real tensors' T dim → IMA.
        constraint_specs=(
            ConstraintSpec(3, 0, lambda shapes: shapes[0][0]),  # mid_out
            ConstraintSpec(4, 0, lambda shapes: shapes[0][0]),  # mid_lse
            ConstraintSpec(5, 0, lambda shapes: shapes[0][0]),  # output
            ConstraintSpec(6, 0, lambda shapes: shapes[0][0]),  # out_lse
        ),
    )


@functools.cache
def _decode_dsv4_runner_singleton():
    """Cache one runner instance per process — avoids re-instantiating on
    every call. The runner is stateless modulo `_module`."""
    return _get_sparse_mla_decode_dsv4_module().runner_cls()


def _decode_dsv4_default_cache_path():
    """Default disk path for the decode-dsv4 AutoTuner cache.

    Pattern mirrors flashinfer's JIT cache: ``$FLASHINFER_WORKSPACE_DIR/
    autotune/sparse_mla_sm120_decode_dsv4.json``. Per-version + per-arch
    (different GPUs / SM counts pick different optimal cpb), invalidated
    when ``flashinfer_version`` changes.

    Override via ``FLASHINFER_AUTOTUNE_DIR`` env var or call
    :func:`sparse_mla_sm120_decode_dsv4_autotune` with an explicit path.
    """
    import pathlib

    override = os.getenv("FLASHINFER_AUTOTUNE_DIR")
    if override:
        base = pathlib.Path(override)
    else:
        from .jit.env import FLASHINFER_WORKSPACE_DIR

        base = FLASHINFER_WORKSPACE_DIR / "autotune"
    return base / "sparse_mla_sm120_decode_dsv4.json"


_decode_dsv4_cache_mtime: float = -1.0

# Per-process hot cache mapping shape signature → cpb tactic. Bypasses
# AutoTuner.choose_one on the steady-state path: dict lookup is ~1 µs vs
# ~3 µs through AutoTuner (lock + tensor-shape extract + cache-key build +
# hash + metadata check). Populated lazily on cold misses + invalidated
# implicitly when a `with autotune(True):` session re-tunes (since the
# tuning-mode branch routes through choose_one and refreshes the entry).
_decode_dsv4_hot_cache: dict = {}


def _decode_dsv4_maybe_load_cache() -> None:
    """Mtime-gated lazy load of the default disk cache.

    Loads the cache if the file is newer than what we've loaded before (or
    never loaded). This is multi-rank safe: a rank that started serving
    BEFORE another rank finished tuning will pick up the updated cache the
    first time it calls into a still-untuned shape (which triggers
    choose_one's cold path that wraps this function). One stat() per cold
    call is cheap; the steady-state hot-cache path skips this entirely.

    Silent on missing file (fresh install) and on load failure (version
    mismatch, corrupt JSON): falls back to C++ heuristic via AutoTuner's
    normal fallback path. Surfacing these as errors would break serving
    for a cache problem.
    """
    global _decode_dsv4_cache_mtime
    path = _decode_dsv4_default_cache_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        # File doesn't exist yet — nothing to load. Keep mtime = -1 so we
        # re-stat on the next cold call (cheap) until the file appears.
        return
    if mtime <= _decode_dsv4_cache_mtime:
        return
    try:
        AutoTuner.get().load_configs(str(path))
        _decode_dsv4_cache_mtime = mtime
    except Exception:
        # Load failed; don't update mtime so we try again next cold call.
        # (If the file is truly broken, we keep retrying — harmless since
        # cold calls are rare in steady state.)
        pass


def sparse_mla_sm120_decode_dsv4_autotune(cache_path: Optional[str] = None):
    """Context manager that opens an autotuning session for decode-dsv4 with
    persistent disk caching.

    Usage::

        with sparse_mla_sm120_decode_dsv4_autotune():
            # First call on each (T_bucket, h, k) shape profiles all cpb
            # tactics and caches the best. Subsequent calls (in this session
            # OR a future fresh process) hit the cache.
            for T in (1, 4, 16, 32):
                sparse_mla_sm120_decode_dsv4(q[:T], ...)

    On exit, the cache is saved to disk at
    :func:`_decode_dsv4_default_cache_path` (or ``cache_path`` if supplied).

    For one-off tuning without a session (e.g. when warming up at server
    startup), simply call this once and let the context manager handle
    load/save.

    Parameters
    ----------
    cache_path : Optional[str]
        Override the default disk cache location. If ``None``, uses
        ``$FLASHINFER_WORKSPACE_DIR/autotune/sparse_mla_sm120_decode_dsv4.json``.
    """
    import pathlib

    from .autotuner import autotune

    if cache_path is None:
        path = _decode_dsv4_default_cache_path()
    else:
        path = pathlib.Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return autotune(True, cache=str(path))


@supported_compute_capability([120, 121])
@flashinfer_api
def sparse_mla_sm120_decode_dsv4(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    output: torch.Tensor,
    out_lse: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    chunks_per_block: Optional[int] = None,
) -> torch.Tensor:
    r"""Sparse-MLA paged decode (DSv4 standalone kernel) on SM120.

    The decode-dsv4 path is the split-K decode variant where each block handles
    ``chunks_per_block`` chunks of 64 candidates each. The wall-time-optimal
    ``chunks_per_block`` is shape-dependent and not well captured by a closed-
    form heuristic. This wrapper integrates flashinfer's :mod:`AutoTuner` to
    pick the per-shape best.

    Behaviour:

    - ``chunks_per_block`` explicitly given → use that value directly (no
      autotuning).
    - Otherwise, if a ``with autotune(...)`` context is active or a previous
      tuning run cached this shape → use the AutoTuner's choice.
    - Otherwise → fall back to the C++ closed-form heuristic.

    Parameters
    ----------
    q : torch.Tensor
        ``[T, num_heads, d_qk]`` bf16. ``d_qk == 512`` (DSV4 only).
    kv_cache : torch.Tensor
        Paged FP8 cache, shape ``[num_blocks, page_bytes]`` uint8.
    indices : torch.Tensor
        ``[T, topk]`` int32. ``topk`` must be one of {128, 512, 1024}; ``-1``
        marks invalid slots.
    mid_out : torch.Tensor
        Scratch, ``[T, num_heads, num_splits, d_v]`` bf16. ``num_splits =
        ceil(topk / 64)``.
    mid_lse : torch.Tensor
        Scratch, ``[T, num_heads, num_splits]`` float32.
    output : torch.Tensor
        In-place output, ``[T, num_heads, d_v]`` bf16.
    out_lse : torch.Tensor
        In-place log-sum-exp, ``[T, num_heads]`` float32.
    sm_scale : float
        Softmax scale.
    topk_length : Optional[torch.Tensor]
        Per-token effective top-k length, ``[T]`` int32.
    chunks_per_block : Optional[int]
        Explicit override. If ``None`` and no AutoTuner active, uses heuristic.

    Returns
    -------
    output : torch.Tensor
        The mutated output tensor (for chaining).
    """
    runner = _decode_dsv4_runner_singleton()
    inputs = [q, kv_cache, indices, mid_out, mid_lse, output, out_lse]

    forward_kwargs = {
        "sm_scale": sm_scale,
        "topk_length": topk_length,
        "attn_sink": attn_sink,
        "extra_kv_cache": extra_kv_cache,
        "extra_indices": extra_indices,
        "extra_topk_length": extra_topk_length,
    }

    if chunks_per_block is not None:
        # Explicit user override — skip AutoTuner entirely.
        runner(
            inputs=inputs,
            tactic=int(chunks_per_block),
            **forward_kwargs,
        )
        return output

    # Hot-cache fast path: tactic is fully determined by (T_bucket, num_heads,
    # topk); skip AutoTuner.choose_one entirely once we've resolved a shape.
    # Only fires outside an active tuning session — inside `with autotune(True):`
    # we route through choose_one so the autotuner gets the data it needs.
    tuner = AutoTuner.get()
    if not tuner.is_tuning_mode:
        T_bucket = _decode_dsv4_map_to_token_bucket(q.shape[0])
        hot_key = (T_bucket, q.shape[1], indices.shape[-1])
        cached_tactic = _decode_dsv4_hot_cache.get(hot_key)
        if cached_tactic is not None:
            runner(
                inputs=inputs,
                tactic=cached_tactic,
                **forward_kwargs,
            )
            return output

    # Cold path (first call on this shape OR active tuning session). Lazy-load
    # the persistent disk cache once per process, then resolve via AutoTuner.
    _decode_dsv4_maybe_load_cache()
    chosen, tactic = tuner.choose_one(
        "sparse_mla_sm120_decode_dsv4",
        [runner],
        _decode_dsv4_tuning_config(),
        inputs,
        **forward_kwargs,
    )
    # Only cache POSITIVE tactics (real tuning results). The autotuner uses
    # tactic=-1 as the "no tuning data, fall back to C++ heuristic" sentinel.
    # Caching -1 would pin every subsequent call to the heuristic, even if
    # a later tuning session (on this rank OR another rank that wrote the
    # shared disk cache) provides a real result — the disk reload would be
    # silently shadowed by the stale -1 in this process's hot dict.
    #
    # By skipping the cache on tactic=-1, untuned shapes pay the ~3 µs
    # choose_one overhead on every call (acceptable since the kernel itself
    # is also running the slower heuristic path), and stay agile to later
    # tuning data arriving via disk reload or in-process autotune().
    if int(tactic) > 0:
        T_bucket = _decode_dsv4_map_to_token_bucket(q.shape[0])
        hot_key = (T_bucket, q.shape[1], indices.shape[-1])
        _decode_dsv4_hot_cache[hot_key] = int(tactic)
    chosen(inputs=inputs, tactic=tactic, **forward_kwargs)
    return output
