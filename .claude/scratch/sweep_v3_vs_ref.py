"""Sweep v2 vs v3 vs FP8-aware reference across the PR_BODY bench grid."""

from __future__ import annotations

import os

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

import sys
import torch
import flashinfer
from flashinfer.jit.sparse_mla_sm120 import gen_sparse_mla_sm120_module

sys.path.insert(0, os.path.dirname(__file__))
from test_decode_v3 import quantize_kv_model1


def dequant_kv_fast(kv_flat_u8, num_blocks=1024, bs=64):
    """Vectorized dequant of the FP8 footer-packed cache to bf16 [s_kv, D_QK]."""
    D_NOPE, D_ROPE, QUANT_TILE = 448, 64, 64
    NUM_SCALES = 7
    IO_STRIDE = D_NOPE + D_ROPE * 2  # 576
    SCALE_BYTES = NUM_SCALES + 1     # 8
    D_QK = D_NOPE + D_ROPE
    bpt = IO_STRIDE + SCALE_BYTES    # 584
    device = kv_flat_u8.device

    # kv_flat_u8: [num_blocks, bs * bpt]
    flat = kv_flat_u8
    # FP8 NoPE: [num_blocks, bs, D_NOPE]
    fp8 = torch.empty(num_blocks, bs, D_NOPE, dtype=torch.uint8, device=device)
    for tok in range(bs):
        fp8[:, tok, :] = flat[:, tok * IO_STRIDE : tok * IO_STRIDE + D_NOPE]

    # RoPE bytes: [num_blocks, bs, D_ROPE * 2 bytes]
    rope_u8 = torch.empty(num_blocks, bs, D_ROPE * 2, dtype=torch.uint8, device=device)
    for tok in range(bs):
        rope_u8[:, tok, :] = flat[:, tok * IO_STRIDE + D_NOPE : (tok + 1) * IO_STRIDE]
    rope_bf16 = rope_u8.view(torch.bfloat16).reshape(num_blocks, bs, D_ROPE)

    # Scale bytes: [num_blocks, bs, NUM_SCALES]
    scales = torch.empty(num_blocks, bs, NUM_SCALES, dtype=torch.uint8, device=device)
    for tok in range(bs):
        scale_off = bs * IO_STRIDE + tok * SCALE_BYTES
        scales[:, tok, :] = flat[:, scale_off : scale_off + NUM_SCALES]

    # Decode UE8M0 -> float scale
    scale_f = (scales.to(torch.float32) - 127).exp2()  # [num_blocks, bs, NUM_SCALES]

    # Dequant FP8 NoPE: reshape to [num_blocks, bs, NUM_SCALES, QUANT_TILE]
    fp8_tiles = fp8.view(num_blocks, bs, NUM_SCALES, QUANT_TILE)
    fp8_floats = fp8_tiles.view(torch.float8_e4m3fn).float()
    nope_bf16 = (fp8_floats * scale_f.unsqueeze(-1)).to(torch.bfloat16).view(num_blocks, bs, D_NOPE)

    result = torch.cat([nope_bf16, rope_bf16], dim=-1)  # [num_blocks, bs, D_QK]
    return result.view(num_blocks * bs, D_QK)


def reference_attention(q, kv_deq, indices, sm_scale):
    qf = q.float()
    kv_gathered = kv_deq[indices.long()].float()
    scores = torch.einsum("thd,tkd->thk", qf, kv_gathered) * sm_scale
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("thk,tkd->thd", weights, kv_gathered)
    return out


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
    return s.elapsed_time(e) / n_iter * 1000


def bench_one(num_heads, topk, num_tokens):
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk = d_v = 512
    page_block_size = 64
    num_blocks = max(1024, (topk * 2 + page_block_size - 1) // page_block_size)
    s_kv = num_blocks * page_block_size

    kv_bf16 = (torch.randn(num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
    kv_flat = quantize_kv_model1(kv_bf16)
    kv_4d = kv_flat.view(num_blocks, page_block_size, 1, kv_flat.size(1) // page_block_size)
    q = (torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
    indices = torch.randint(0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32)
    sm_scale = d_qk ** -0.5

    # Reference (FP8-aware)
    kv_deq = dequant_kv_fast(kv_flat, num_blocks=num_blocks, bs=page_block_size)
    out_ref = reference_attention(q, kv_deq, indices, sm_scale)

    # v2
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    output_v2 = torch.empty(num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=device)
    out_lse_v2 = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=device)

    def call_v2():
        flashinfer.sparse_mla_sm120_paged_attention(
            q=q, kv_cache=kv_4d, indices=indices, output=output_v2, out_lse=out_lse_v2,
            workspace_buffer=workspace, sm_scale=sm_scale, d_v=d_v,
        )

    # v3
    module = gen_sparse_mla_sm120_module().build_and_load()
    V3_CAND_WINDOW = 64  # must match constexpr in decode_v3_kernel.cuh
    num_splits = (topk + V3_CAND_WINDOW - 1) // V3_CAND_WINDOW
    mid = torch.empty(num_tokens, num_heads, num_splits, d_v, dtype=torch.bfloat16, device=device)
    mid_lse = torch.empty(num_tokens, num_heads, num_splits, dtype=torch.float32, device=device)
    output_v3 = torch.empty(num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=device)
    out_lse_v3 = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=device)

    def call_v3():
        module.sparse_mla_sm120_decode_v3(
            q, kv_flat, indices, mid, mid_lse, output_v3, out_lse_v3, num_splits, sm_scale, None,
        )

    call_v2()
    call_v3()
    torch.cuda.synchronize()
    err_v2 = (output_v2.float() - out_ref).abs().max().item()
    err_v3 = (output_v3.float() - out_ref).abs().max().item()

    v2_us = time_fn(call_v2)
    v3_us = time_fn(call_v3)
    return {"h": num_heads, "topk": topk, "T": num_tokens, "v2_us": v2_us, "v3_us": v3_us,
            "speedup": v2_us / v3_us, "err_v2": err_v2, "err_v3": err_v3}


def main():
    shapes = [
        (16, 128, 1), (16, 128, 16), (16, 512, 16),
        (32, 128, 1), (32, 128, 16), (32, 512, 16),
        (64, 128, 1), (64, 128, 16), (64, 512, 16), (64, 1024, 16),
        (128, 128, 1), (128, 128, 16), (128, 512, 16), (128, 1024, 16),
        (128, 128, 32), (128, 512, 32),
    ]
    print(f"{'h':>4} {'topk':>5} {'T':>4}  {'v2 (us)':>9} {'v3 (us)':>9} {'speedup':>8}  "
          f"{'err_v2':>9} {'err_v3':>9}")
    print("-" * 75)
    rows = []
    for h, k, t in shapes:
        try:
            r = bench_one(h, k, t)
            rows.append(r)
            mark = "v3✓" if r["speedup"] > 1.0 else "v2✓"
            v2_ok = " " if r["err_v2"] < 0.05 else "!"
            v3_ok = " " if r["err_v3"] < 0.05 else "!"
            print(f"{r['h']:>4} {r['topk']:>5} {r['T']:>4}  "
                  f"{r['v2_us']:>9.1f} {r['v3_us']:>9.1f} {r['speedup']:>7.2f}x  "
                  f"{r['err_v2']:>9.2e}{v2_ok}  {r['err_v3']:>9.2e}{v3_ok}  {mark}")
        except Exception as e:
            print(f"{h:>4} {k:>5} {t:>4}  FAILED: {e}")

    print()
    wins = sum(1 for r in rows if r["speedup"] > 1.0)
    print(f"v3 wins {wins}/{len(rows)} shapes (vs v2 wall-clock)")


if __name__ == "__main__":
    main()
