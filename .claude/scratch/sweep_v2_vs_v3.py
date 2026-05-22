"""Sweep v2 vs v3 across the PR_BODY bench grid (h × topk × T)."""

from __future__ import annotations

import os

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

import sys

import torch
import flashinfer
from flashinfer.jit.sparse_mla_sm120 import gen_sparse_mla_sm120_module

sys.path.insert(0, os.path.dirname(__file__))
from test_decode_v3 import quantize_kv_model1


def time_fn(fn, n_warmup=5, n_iter=50):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter * 1000  # microseconds


def bench_one(num_heads: int, topk: int, num_tokens: int) -> dict:
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk = 512
    d_v = 512
    page_block_size = 64
    num_blocks = max(1024, (topk * 2 + page_block_size - 1) // page_block_size)
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16) / 10.0
    ).clamp(-1, 1)
    kv_flat = quantize_kv_model1(kv_bf16)
    bpt = kv_flat.size(1) // page_block_size
    kv_4d = kv_flat.view(num_blocks, page_block_size, 1, bpt)
    q = (torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
    indices = torch.randint(0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32)
    sm_scale = d_qk ** -0.5

    # v2 setup
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    output_v2 = torch.empty(num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=device)
    out_lse_v2 = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=device)

    def call_v2():
        flashinfer.sparse_mla_sm120_paged_attention(
            q=q, kv_cache=kv_4d, indices=indices, output=output_v2, out_lse=out_lse_v2,
            workspace_buffer=workspace, sm_scale=sm_scale, d_v=d_v,
        )

    # v3 setup
    module = gen_sparse_mla_sm120_module().build_and_load()
    cand_window = 64
    num_splits = (topk + cand_window - 1) // cand_window
    mid = torch.empty(num_tokens, num_heads, num_splits, d_v, dtype=torch.bfloat16, device=device)
    mid_lse = torch.empty(num_tokens, num_heads, num_splits, dtype=torch.float32, device=device)
    output_v3 = torch.empty(num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=device)
    out_lse_v3 = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=device)

    def call_v3():
        module.sparse_mla_sm120_decode_v3(
            q, kv_flat, indices, mid, mid_lse, output_v3, out_lse_v3, num_splits, sm_scale, None,
        )

    # correctness check
    call_v2()
    call_v3()
    torch.cuda.synchronize()
    abs_err = (output_v2.float() - output_v3.float()).abs().max().item()

    v2_us = time_fn(call_v2)
    v3_us = time_fn(call_v3)
    return {"h": num_heads, "topk": topk, "T": num_tokens, "v2_us": v2_us, "v3_us": v3_us,
            "speedup": v2_us / v3_us, "abs_err": abs_err}


def main():
    shapes = [
        (16, 128, 1), (16, 128, 16), (16, 512, 16),
        (32, 128, 1), (32, 128, 16), (32, 512, 16),
        (64, 128, 1), (64, 128, 16), (64, 512, 16), (64, 1024, 16),
        (128, 128, 1), (128, 128, 16), (128, 512, 16), (128, 1024, 16),
        (128, 128, 32), (128, 512, 32),
    ]
    print(f"{'h':>4} {'topk':>5} {'T':>4}  {'v2 (us)':>9} {'v3 (us)':>9} {'speedup':>8}  {'abs_err':>9}")
    print("-" * 64)
    rows = []
    for h, k, t in shapes:
        try:
            r = bench_one(h, k, t)
            rows.append(r)
            mark = "✓" if r["speedup"] > 1.0 else "✗"
            print(f"{r['h']:>4} {r['topk']:>5} {r['T']:>4}  "
                  f"{r['v2_us']:>9.1f} {r['v3_us']:>9.1f} {r['speedup']:>7.2f}x  "
                  f"{r['abs_err']:>9.2e} {mark}")
        except Exception as e:
            print(f"{h:>4} {k:>5} {t:>4}  FAILED: {e}")

    print()
    wins = sum(1 for r in rows if r["speedup"] > 1.0)
    print(f"v3 wins {wins}/{len(rows)} shapes")


if __name__ == "__main__":
    main()
