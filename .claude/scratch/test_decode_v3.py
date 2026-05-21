"""Correctness test: decode-v3 vs decode-v2 on a single contested shape.

Drives the bare TVM-FFI export of the decode-v3 kernel and compares its
output to the production decode-v2 path.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

import torch

import flashinfer
from flashinfer.jit.sparse_mla_sm120 import gen_sparse_mla_sm120_module


def quantize_kv_model1(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Mirrors benchmarks/bench_sparse_mla_sm120.py."""
    nb, bs, _, d_qk = kv_bf16.shape
    d_nope, d_rope, tile_size, num_tiles = 448, 64, 64, 7
    nope = kv_bf16[..., :d_nope]
    rope = kv_bf16[..., d_nope:]
    fp8 = torch.empty(nb, bs, 1, d_nope, dtype=torch.float8_e4m3fn, device=kv_bf16.device)
    ue8m0 = torch.empty(nb, bs, 1, num_tiles, dtype=torch.uint8, device=kv_bf16.device)
    for t in range(num_tiles):
        slab = nope[..., t * tile_size : (t + 1) * tile_size]
        amax = slab.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-6)
        scale = amax / 448.0
        log2_scale = torch.ceil(torch.log2(scale))
        ue8m0_byte = (log2_scale.to(torch.int32) + 127).clamp(0, 255).to(torch.uint8)
        ue8m0[..., t] = ue8m0_byte.squeeze(-1)
        eff_scale = torch.pow(2.0, log2_scale)
        fp8_slab = (slab / eff_scale).to(torch.float8_e4m3fn)
        fp8[..., t * tile_size : (t + 1) * tile_size] = fp8_slab
    data_stride = d_nope + d_rope * 2
    scale_bytes = num_tiles + 1
    bpt = data_stride + scale_bytes
    result_flat = torch.zeros(nb, bs * bpt, dtype=torch.uint8, device=kv_bf16.device)
    fp8_u8 = fp8.view(torch.uint8)
    rope_u8 = rope.view(torch.uint8)
    for ti in range(num_tiles):
        for tok in range(bs):
            data_off = tok * data_stride + ti * tile_size
            result_flat[:, data_off : data_off + tile_size] = fp8_u8[:, tok, 0, ti * tile_size : (ti + 1) * tile_size]
    for tok in range(bs):
        rope_off = tok * data_stride + d_nope
        result_flat[:, rope_off : rope_off + d_rope * 2] = rope_u8[:, tok, 0]
    for ti in range(num_tiles):
        for tok in range(bs):
            scale_off = bs * data_stride + tok * scale_bytes + ti
            result_flat[:, scale_off] = ue8m0[:, tok, 0, ti]
    return result_flat


def run_decode_v2(q, kv, indices, sm_scale):
    num_tokens, num_heads, d_qk = q.shape
    d_v = 512
    topk = indices.shape[-1]
    output = torch.empty(num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=q.device)
    out_lse = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=q.device)
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    flashinfer.sparse_mla_sm120_paged_attention(
        q=q,
        kv_cache=kv,
        indices=indices,
        output=output,
        out_lse=out_lse,
        workspace_buffer=workspace_buffer,
        sm_scale=sm_scale,
        d_v=d_v,
    )
    return output, out_lse


def run_decode_v3(q, kv, indices, sm_scale, num_splits):
    module = gen_sparse_mla_sm120_module().build_and_load()
    num_tokens, num_heads, d_qk = q.shape
    d_v = 512
    output = torch.empty(num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=q.device)
    out_lse = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=q.device)
    mid = torch.empty(num_tokens, num_heads, num_splits, d_v, dtype=torch.bfloat16, device=q.device)
    mid_lse = torch.empty(num_tokens, num_heads, num_splits, dtype=torch.float32, device=q.device)
    module.sparse_mla_sm120_decode_v3(q, kv, indices, mid, mid_lse, output, out_lse, num_splits, sm_scale, None)
    return output, out_lse


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens = 16
    num_heads = 128
    topk = 512
    d_qk = 512
    page_block_size = 64
    num_blocks = 1024
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    kv_flat = quantize_kv_model1(kv_bf16)  # [nb, bs*bpt]
    bpt = kv_flat.size(1) // page_block_size  # 584 for MODEL1
    kv_4d = kv_flat.view(num_blocks, page_block_size, 1, bpt)  # for v2 path
    q = (torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
    indices = torch.randint(0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32)
    sm_scale = d_qk**-0.5

    print(f"shape: T={num_tokens} H={num_heads} topk={topk} d_qk={d_qk} bpt={bpt}", flush=True)

    print("running v2...", flush=True)
    out_v2, lse_v2 = run_decode_v2(q, kv_4d, indices, sm_scale)
    torch.cuda.synchronize()
    print(f"  v2 output  abs.max = {out_v2.abs().max().item():.4f}")
    print(f"  v2 lse     abs.max = {lse_v2.abs().max().item():.4f}")

    print("running v3...", flush=True)
    num_splits = 8  # topk / V3_CAND_WINDOW = 512/64
    out_v3, lse_v3 = run_decode_v3(q, kv_flat, indices, sm_scale, num_splits)
    torch.cuda.synchronize()
    print(f"  v3 output  abs.max = {out_v3.abs().max().item():.4f}")
    print(f"  v3 lse     abs.max = {lse_v3.abs().max().item():.4f}")

    # Perf bench v3
    import time
    module = gen_sparse_mla_sm120_module().build_and_load()
    d_v = 512
    mid = torch.empty(num_tokens, num_heads, 8, d_v, dtype=torch.bfloat16, device=device)
    mid_lse = torch.empty(num_tokens, num_heads, 8, dtype=torch.float32, device=device)
    output_v3 = torch.empty(num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=device)
    out_lse_v3 = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=device)
    for _ in range(5):
        module.sparse_mla_sm120_decode_v3(q, kv_flat, indices, mid, mid_lse, output_v3, out_lse_v3, 8, sm_scale, None)
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(50):
        module.sparse_mla_sm120_decode_v3(q, kv_flat, indices, mid, mid_lse, output_v3, out_lse_v3, 8, sm_scale, None)
    end_evt.record()
    torch.cuda.synchronize()
    v3_us = start_evt.elapsed_time(end_evt) / 50 * 1000
    print(f"\n  v3 perf: {v3_us:.1f} us/iter at h={num_heads} topk={topk} T={num_tokens}")

    # Compare
    out_f32_v2 = out_v2.float()
    out_f32_v3 = out_v3.float()
    abs_err = (out_f32_v2 - out_f32_v3).abs()
    rel_err = abs_err / (out_f32_v2.abs() + 1e-6)
    print(f"\n  output max abs err: {abs_err.max().item():.4e}")
    print(f"  output max rel err: {rel_err.max().item():.4e}")
    print(f"  lse    max abs err: {(lse_v2 - lse_v3).abs().max().item():.4e}")

    # Diagnostic: count zero outputs in v3
    zero_mask = out_f32_v3.abs() < 1e-6
    print(f"\n  v3 zero-output fraction: {zero_mask.float().mean().item():.3f}")
    print(f"  v3 per-head zero count (head 0): {zero_mask[0, 0, :].sum().item()}/512")
    print(f"  v3 per-head zero count (head 5): {zero_mask[0, 5, :].sum().item()}/512")
    print(f"  v3 per-head zero count (head 8): {zero_mask[0, 8, :].sum().item()}/512")
    print(f"  v2 head 0 first 8 dims: {out_v2[0, 0, :8].tolist()}")
    print(f"  v3 head 0 first 8 dims: {out_v3[0, 0, :8].tolist()}")
    print(f"  v2 head 0 dims 64..72: {out_v2[0, 0, 64:72].tolist()}")
    print(f"  v3 head 0 dims 64..72: {out_v3[0, 0, 64:72].tolist()}")


if __name__ == "__main__":
    main()
