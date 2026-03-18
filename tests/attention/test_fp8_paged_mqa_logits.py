"""Tests for FP8 paged MQA logits kernel (NSA indexer)."""

import math

import pytest
import torch

from flashinfer.fp8_paged_mqa_logits import (
    fp8_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
)


def _quantize_fp8(x: torch.Tensor):
    """Quantize float32 tensor to FP8 e4m3 with per-token scaling."""
    amax = x.abs().float().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = amax / 448.0  # FP8 e4m3 max value
    x_fp8 = (x / scale).to(torch.float8_e4m3fn)
    return x_fp8, scale.squeeze(-1)


def _reference_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,  # [batch, next_n, num_heads, head_dim] fp8
    kv_fp8_pages: list,  # list of (k_fp8[block_kv, head_dim], scale[block_kv]) per block
    weights: torch.Tensor,  # [batch*next_n, num_heads] float32
    seq_lens: torch.Tensor,  # [batch] int32
    block_tables: torch.Tensor,  # [batch, max_blocks] int32
    max_seq_len: int,
    block_kv: int = 64,
) -> torch.Tensor:
    """Reference implementation in PyTorch."""
    batch, next_n, num_heads, head_dim = q_fp8.shape
    logits = torch.full(
        (batch * next_n, max_seq_len), float("-inf"), dtype=torch.float32, device=q_fp8.device
    )

    q_f32 = q_fp8.to(torch.float32)

    for b in range(batch):
        sl = seq_lens[b].item()
        num_blocks = math.ceil(sl / block_kv)
        for n in range(next_n):
            bn = b * next_n + n
            for blk_idx in range(num_blocks):
                phys_block = block_tables[b, blk_idx].item()
                k_fp8, k_scale = kv_fp8_pages[phys_block]
                k_f32 = k_fp8.to(torch.float32)

                token_start = blk_idx * block_kv
                valid = min(block_kv, sl - token_start)

                for t in range(valid):
                    logit_acc = 0.0
                    for h in range(num_heads):
                        dot = (q_f32[b, n, h, :] * k_f32[t, :]).sum().item()
                        s = dot * k_scale[t].item()
                        s = max(s, 0.0)  # ReLU
                        logit_acc += s * weights[bn, h].item()
                    logits[bn, token_start + t] = logit_acc

    return logits


def _build_fused_kv_cache(kv_fp8_pages, head_dim, block_kv, device):
    """Build fused KV cache [num_blocks, block_kv, 1, head_dim+4] as uint8."""
    num_blocks = len(kv_fp8_pages)
    head_dim_with_sf = head_dim + 4
    fused = torch.zeros(num_blocks, block_kv, 1, head_dim_with_sf, dtype=torch.uint8, device=device)

    for bi, (k_fp8, k_scale) in enumerate(kv_fp8_pages):
        # FP8 data: first head_dim bytes
        k_uint8 = k_fp8.view(torch.uint8).reshape(block_kv, head_dim)
        fused[bi, :, 0, :head_dim] = k_uint8
        # FP32 scale: last 4 bytes
        scale_uint8 = k_scale.view(torch.uint8).reshape(block_kv, 4)
        fused[bi, :, 0, head_dim : head_dim + 4] = scale_uint8

    return fused


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("avg_seq_len", [64, 256])
@pytest.mark.parametrize("next_n", [1])
def test_fp8_paged_mqa_logits_correctness(batch_size, avg_seq_len, next_n):
    """Test Triton kernel against reference implementation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    num_heads = 32
    head_dim = 128
    block_kv = 64

    # Generate random sequence lengths
    torch.manual_seed(42)
    seq_lens = torch.randint(
        max(block_kv, avg_seq_len // 2),
        avg_seq_len + avg_seq_len // 2,
        (batch_size,),
        dtype=torch.int32,
        device=device,
    )
    max_seq_len = seq_lens.max().item()
    max_blocks = math.ceil(max_seq_len / block_kv)

    # Total physical blocks needed
    total_blocks = sum(math.ceil(s.item() / block_kv) for s in seq_lens)
    # Add some extra blocks for realistic page table
    num_phys_blocks = total_blocks + batch_size * 2

    # Generate random K data and quantize to FP8
    kv_fp8_pages = []
    for _ in range(num_phys_blocks):
        k_f32 = torch.randn(block_kv, head_dim, device=device)
        k_fp8, k_scale = _quantize_fp8(k_f32)
        kv_fp8_pages.append((k_fp8, k_scale))

    # Build page tables (simple sequential assignment)
    block_tables = torch.zeros(batch_size, max_blocks, dtype=torch.int32, device=device)
    block_idx = 0
    for b in range(batch_size):
        n_blocks = math.ceil(seq_lens[b].item() / block_kv)
        for i in range(n_blocks):
            block_tables[b, i] = block_idx
            block_idx += 1

    # Generate random Q and weights
    q_f32 = torch.randn(batch_size, next_n, num_heads, head_dim, device=device)
    q_fp8, _ = _quantize_fp8(q_f32.reshape(-1, head_dim))
    q_fp8 = q_fp8.reshape(batch_size, next_n, num_heads, head_dim)

    weights = torch.randn(batch_size * next_n, num_heads, device=device, dtype=torch.float32)

    # Build fused KV cache
    fused_kv = _build_fused_kv_cache(kv_fp8_pages, head_dim, block_kv, device)

    # Schedule metadata (dummy for Triton)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    schedule_meta = get_paged_mqa_logits_metadata(seq_lens, block_kv, num_sms)

    # Run Triton kernel
    logits_triton = fp8_paged_mqa_logits(
        q_fp8, fused_kv, weights, seq_lens, block_tables,
        schedule_meta, max_seq_len, clean_logits=False,
    )

    # Run reference
    logits_ref = _reference_fp8_paged_mqa_logits(
        q_fp8, kv_fp8_pages, weights, seq_lens, block_tables,
        max_seq_len, block_kv,
    )

    # Compare only valid positions
    for b in range(batch_size):
        sl = seq_lens[b].item()
        for n in range(next_n):
            bn = b * next_n + n
            triton_valid = logits_triton[bn, :sl]
            ref_valid = logits_ref[bn, :sl]

            # Allow tolerance for FP8 quantization noise
            max_abs_err = (triton_valid - ref_valid).abs().max().item()
            assert max_abs_err < 0.1, (
                f"batch={b}, next_n={n}: max abs error {max_abs_err:.4f} > 0.1"
            )


def test_clean_logits():
    """Test that clean_logits zeros out positions beyond sequence length."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    batch_size = 2
    next_n = 1
    num_heads = 32
    head_dim = 128
    block_kv = 64

    seq_lens = torch.tensor([64, 128], dtype=torch.int32, device=device)
    max_seq_len = 128
    max_blocks = 2

    # Build minimal KV cache
    kv_fp8_pages = []
    for _ in range(4):
        k_f32 = torch.randn(block_kv, head_dim, device=device)
        k_fp8, k_scale = _quantize_fp8(k_f32)
        kv_fp8_pages.append((k_fp8, k_scale))

    fused_kv = _build_fused_kv_cache(kv_fp8_pages, head_dim, block_kv, device)
    block_tables = torch.tensor([[0, -1], [1, 2]], dtype=torch.int32, device=device)

    q_f32 = torch.randn(batch_size, next_n, num_heads, head_dim, device=device)
    q_fp8, _ = _quantize_fp8(q_f32.reshape(-1, head_dim))
    q_fp8 = q_fp8.reshape(batch_size, next_n, num_heads, head_dim)
    weights = torch.randn(batch_size * next_n, num_heads, device=device, dtype=torch.float32)

    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    schedule_meta = get_paged_mqa_logits_metadata(seq_lens, block_kv, num_sms)

    logits = fp8_paged_mqa_logits(
        q_fp8, fused_kv, weights, seq_lens, block_tables,
        schedule_meta, max_seq_len, clean_logits=True,
    )

    # Batch 0: seq_len=64, positions 64-127 should be 0
    assert (logits[0, 64:] == 0.0).all(), "clean_logits failed for batch 0"
    # Batch 1: seq_len=128, all positions valid (no cleaning needed)


if __name__ == "__main__":
    test_fp8_paged_mqa_logits_correctness(batch_size=1, avg_seq_len=64, next_n=1)
    print("PASSED: batch=1, seq=64")
    test_fp8_paged_mqa_logits_correctness(batch_size=4, avg_seq_len=256, next_n=1)
    print("PASSED: batch=4, seq=256")
    test_clean_logits()
    print("PASSED: clean_logits")
    print("All tests passed!")
