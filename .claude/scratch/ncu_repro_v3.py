"""Single-shot v3 decode driver for ncu capture.

Builds the kernel, warms up 5×, then runs cudaProfilerApi-bracketed range
with ONE invocation so ncu's `--capture-range=cudaProfilerApi` picks exactly
that launch (+ its merge kernel).

Shape: h=128 / topk=512 / T=16 (one of the 4 jasl-losing shapes).
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

import torch
import torch.cuda.profiler as cuda_profiler
from flashinfer.jit.sparse_mla_sm120 import gen_sparse_mla_sm120_module

sys.path.insert(0, os.path.dirname(__file__))
from test_decode_v3 import quantize_kv_model1


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens, num_heads, topk = 16, 128, 512
    d_qk, d_v = 512, 512
    pbs = 64
    num_blocks = 1024
    s_kv = num_blocks * pbs

    kv_bf16 = (torch.randn(num_blocks, pbs, 1, d_qk, device=device, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
    kv_flat = quantize_kv_model1(kv_bf16)
    q = (torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
    indices = torch.randint(0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32)
    sm_scale = d_qk ** -0.5

    module = gen_sparse_mla_sm120_module().build_and_load()

    V3_CW = 64  # A1.2: dropped to 64 for double-buffered KV smem
    num_splits = (topk + V3_CW - 1) // V3_CW  # 4 for topk=512
    mid = torch.empty(num_tokens, num_heads, num_splits, d_v, dtype=torch.bfloat16, device=device)
    mid_lse = torch.empty(num_tokens, num_heads, num_splits, dtype=torch.float32, device=device)
    out = torch.empty(num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=device)
    out_lse = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=device)

    def call_v3() -> None:
        module.sparse_mla_sm120_decode_v3(q, kv_flat, indices, mid, mid_lse, out, out_lse, num_splits, sm_scale, None)

    # warmup outside profiler range
    for _ in range(5):
        call_v3()
    torch.cuda.synchronize()

    # one bracketed invocation for ncu --capture-range=cudaProfilerApi
    cuda_profiler.start()
    call_v3()
    cuda_profiler.stop()
    torch.cuda.synchronize()
    print(f"v3 driver: h={num_heads} topk={topk} T={num_tokens} num_splits={num_splits} done")


if __name__ == "__main__":
    main()
