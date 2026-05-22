"""Sweep live v3 vs frozen v3_backup snapshot + jasl + PyTorch reference.

v2 has been beaten on every PR_BODY shape, so it's removed as a baseline.
v3_backup is the frozen v3 at commit 41ac1687 (the previous "best v3"), and
any live v3 change should be compared against it for both perf and correctness.

PyTorch (FP8-aware, dequant + softmax) is the correctness reference.

Run with: FLASHINFER_DISABLE_VERSION_CHECK=1 .venv/bin/python sweep_v3_vs_backup.py
"""

from __future__ import annotations

import os

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

import sys
import torch
from flashinfer.jit.sparse_mla_sm120 import gen_sparse_mla_sm120_module

sys.path.insert(0, os.path.dirname(__file__))
from test_decode_v3 import quantize_kv_model1
from sweep_v3_vs_ref import dequant_kv_fast, reference_attention, time_fn


def bench_one(num_heads, topk, num_tokens):
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk = d_v = 512
    page_block_size = 64
    num_blocks = max(1024, (topk * 2 + page_block_size - 1) // page_block_size)
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    kv_flat = quantize_kv_model1(kv_bf16)
    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16) / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32)
    sm_scale = d_qk ** -0.5

    # PyTorch reference (FP8-aware dequant + softmax)
    kv_deq = dequant_kv_fast(kv_flat, num_blocks=num_blocks, bs=page_block_size)
    out_ref = reference_attention(q, kv_deq, indices, sm_scale)

    module = gen_sparse_mla_sm120_module().build_and_load()
    # A1.2 dropped CW back to 64 to fit double-buffered KV smem; backup is also 64.
    V3_CAND_WINDOW_LIVE = 64
    V3_CAND_WINDOW_BACKUP = 64
    num_splits_v3 = (topk + V3_CAND_WINDOW_LIVE - 1) // V3_CAND_WINDOW_LIVE
    num_splits_bk = (topk + V3_CAND_WINDOW_BACKUP - 1) // V3_CAND_WINDOW_BACKUP

    # Live v3
    mid_v3 = torch.empty(num_tokens, num_heads, num_splits_v3, d_v, dtype=torch.bfloat16, device=device)
    mid_lse_v3 = torch.empty(num_tokens, num_heads, num_splits_v3, dtype=torch.float32, device=device)
    out_v3 = torch.empty(num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=device)
    lse_v3 = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=device)

    def call_v3():
        module.sparse_mla_sm120_decode_v3(
            q, kv_flat, indices, mid_v3, mid_lse_v3, out_v3, lse_v3, num_splits_v3, sm_scale,
            None,  # topk_length
            None,  # attn_sink
            None,  # extra_kv_cache
            None,  # extra_indices
            None,  # extra_topk_length
            -1,    # chunks_per_block override (-1 = heuristic)
        )

    # Frozen v3_backup
    mid_bk = torch.empty(num_tokens, num_heads, num_splits_bk, d_v, dtype=torch.bfloat16, device=device)
    mid_lse_bk = torch.empty(num_tokens, num_heads, num_splits_bk, dtype=torch.float32, device=device)
    out_bk = torch.empty(num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=device)
    lse_bk = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=device)

    def call_backup():
        module.sparse_mla_sm120_decode_v3_backup(
            q, kv_flat, indices, mid_bk, mid_lse_bk, out_bk, lse_bk, num_splits_bk, sm_scale, None,
        )

    call_v3()
    call_backup()
    torch.cuda.synchronize()
    err_v3 = (out_v3.float() - out_ref).abs().max().item()
    err_bk = (out_bk.float() - out_ref).abs().max().item()
    # v3 vs backup output divergence (kernel-to-kernel drift)
    drift = (out_v3.float() - out_bk.float()).abs().max().item()

    v3_us = time_fn(call_v3)
    bk_us = time_fn(call_backup)
    return {
        "h": num_heads, "topk": topk, "T": num_tokens,
        "v3_us": v3_us, "bk_us": bk_us, "speedup": bk_us / v3_us,
        "err_v3": err_v3, "err_bk": err_bk, "drift": drift,
    }


def main():
    shapes = [
        (16, 128, 1), (16, 128, 16), (16, 512, 16),
        (32, 128, 1), (32, 128, 16), (32, 512, 16),
        (64, 128, 1), (64, 128, 16), (64, 512, 16), (64, 1024, 16),
        (128, 128, 1), (128, 128, 16), (128, 512, 16), (128, 1024, 16),
        (128, 128, 32), (128, 512, 32),
    ]
    print(f"{'h':>4} {'topk':>5} {'T':>4}  "
          f"{'v3 (us)':>9} {'bkup (us)':>10} {'speedup':>8}  "
          f"{'err_v3':>9} {'err_bk':>9} {'drift':>9}")
    print("-" * 90)
    rows = []
    for h, k, t in shapes:
        try:
            r = bench_one(h, k, t)
            rows.append(r)
            mark = "v3✓" if r["speedup"] > 1.0 else ("==" if abs(r["speedup"] - 1.0) < 0.01 else "bk✓")
            v3_ok = " " if r["err_v3"] < 0.05 else "!"
            bk_ok = " " if r["err_bk"] < 0.05 else "!"
            print(f"{r['h']:>4} {r['topk']:>5} {r['T']:>4}  "
                  f"{r['v3_us']:>9.1f} {r['bk_us']:>10.1f} {r['speedup']:>7.2f}x  "
                  f"{r['err_v3']:>9.2e}{v3_ok} {r['err_bk']:>9.2e}{bk_ok} "
                  f"{r['drift']:>9.2e}  {mark}")
        except Exception as e:
            print(f"{h:>4} {k:>5} {t:>4}  FAILED: {e}")

    print()
    wins = sum(1 for r in rows if r["speedup"] > 1.0)
    losses = sum(1 for r in rows if r["speedup"] < 0.99)
    print(f"v3 vs v3_backup: {wins}/{len(rows)} wins, {losses}/{len(rows)} regressions")
    max_drift = max((r["drift"] for r in rows), default=0)
    print(f"Max v3↔backup output drift: {max_drift:.3e}")


if __name__ == "__main__":
    main()
