"""Tests for the Triton sparse MLA decode kernel.

Validates against a PyTorch reference that computes exact sparse attention
over the NSA-selected top-k tokens.
"""

import pytest
import torch


def _reference_sparse_mla_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    scale: float,
    kv_lora_rank: int,
) -> torch.Tensor:
    """PyTorch reference: brute-force sparse MLA decode.

    Args:
        q: [batch, num_heads, head_dim] float32
        kv_cache: [total_tokens, kv_dim] float32
        sparse_indices: [batch, top_k] int32 (-1 = padding)
        scale: attention scale
        kv_lora_rank: V output dim (first kv_lora_rank dims of KV)

    Returns:
        output: [batch, num_heads, kv_lora_rank] float32
    """
    batch, num_heads, head_dim = q.shape
    top_k = sparse_indices.shape[1]

    output = torch.zeros(batch, num_heads, kv_lora_rank, dtype=torch.float32, device=q.device)

    for b in range(batch):
        for h in range(num_heads):
            q_vec = q[b, h, :]  # [head_dim]

            # Gather valid KV tokens
            indices = sparse_indices[b]  # [top_k]
            valid_mask = indices >= 0
            valid_indices = indices[valid_mask]

            if valid_indices.numel() == 0:
                continue

            kv_rows = kv_cache[valid_indices.long(), :]  # [n_valid, kv_dim]

            # Scores: Q dot KV
            scores = (kv_rows @ q_vec) * scale  # [n_valid]

            # Softmax
            scores_max = scores.max()
            exp_scores = torch.exp(scores - scores_max)
            attn = exp_scores / exp_scores.sum()

            # V = first kv_lora_rank dims
            v = kv_rows[:, :kv_lora_rank]  # [n_valid, kv_lora_rank]
            output[b, h, :] = attn @ v

    return output


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


class TestSparseMlaDecode:
    """Test suite for sparse_mla_decode kernel."""

    def _run_and_compare(
        self,
        device,
        batch,
        num_heads,
        head_dim,
        kv_lora_rank,
        top_k,
        total_tokens,
        atol=2e-2,
        rtol=1e-2,
    ):
        from flashinfer.sparse_mla_decode import sparse_mla_decode

        torch.manual_seed(42)

        # Random Q and KV cache
        q = torch.randn(batch, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        kv_cache = torch.randn(total_tokens, head_dim, dtype=torch.bfloat16, device=device)

        # Random sparse indices (no duplicates, within [0, total_tokens))
        sparse_indices = torch.stack([
            torch.randperm(total_tokens, device=device)[:top_k].to(torch.int32)
            for _ in range(batch)
        ])

        scale = 1.0 / (head_dim ** 0.5)

        # Triton kernel
        out_triton = sparse_mla_decode(q, kv_cache, sparse_indices, scale, kv_lora_rank)

        # Reference (fp32 for precision)
        out_ref = _reference_sparse_mla_decode(
            q.float(), kv_cache.float(), sparse_indices, scale, kv_lora_rank,
        )

        torch.testing.assert_close(
            out_triton.float(), out_ref, atol=atol, rtol=rtol,
        )

    def test_dsv3_dimensions(self, device):
        """DeepSeek-V3.2 typical decode dimensions."""
        self._run_and_compare(
            device,
            batch=4,
            num_heads=16,  # 128 / TP=8
            head_dim=576,
            kv_lora_rank=512,
            top_k=2048,
            total_tokens=8192,
        )

    def test_small(self, device):
        """Small problem for fast validation."""
        self._run_and_compare(
            device,
            batch=2,
            num_heads=4,
            head_dim=128,
            kv_lora_rank=96,
            top_k=64,
            total_tokens=256,
            atol=1e-2,
        )

    def test_single_batch(self, device):
        """Batch size 1."""
        self._run_and_compare(
            device,
            batch=1,
            num_heads=16,
            head_dim=576,
            kv_lora_rank=512,
            top_k=2048,
            total_tokens=4096,
        )

    def test_large_batch(self, device):
        """Batch size 48 (typical decode batch)."""
        self._run_and_compare(
            device,
            batch=48,
            num_heads=16,
            head_dim=576,
            kv_lora_rank=512,
            top_k=2048,
            total_tokens=65536,
        )

    def test_with_padding(self, device):
        """Sparse indices with -1 padding entries."""
        from flashinfer.sparse_mla_decode import sparse_mla_decode

        torch.manual_seed(123)
        batch, num_heads, head_dim = 2, 8, 576
        kv_lora_rank, top_k, total_tokens = 512, 2048, 4096

        q = torch.randn(batch, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        kv_cache = torch.randn(total_tokens, head_dim, dtype=torch.bfloat16, device=device)

        # Last 256 entries are -1 (padding)
        valid = torch.randperm(total_tokens, device=device)[:top_k - 256].to(torch.int32)
        padding = torch.full((256,), -1, dtype=torch.int32, device=device)
        row = torch.cat([valid, padding])
        sparse_indices = row.unsqueeze(0).expand(batch, -1).contiguous()

        scale = 1.0 / (head_dim ** 0.5)

        out_triton = sparse_mla_decode(q, kv_cache, sparse_indices, scale, kv_lora_rank)
        out_ref = _reference_sparse_mla_decode(
            q.float(), kv_cache.float(), sparse_indices, scale, kv_lora_rank,
        )

        torch.testing.assert_close(out_triton.float(), out_ref, atol=2e-2, rtol=1e-2)

    def test_top_k_not_divisible_by_block(self, device):
        """top_k not evenly divisible by BLOCK_K=16."""
        self._run_and_compare(
            device,
            batch=2,
            num_heads=4,
            head_dim=192,
            kv_lora_rank=128,
            top_k=100,  # 100 % 16 = 4
            total_tokens=512,
            atol=1e-2,
        )

    def test_head_dim_256(self, device):
        """Different head_dim (256 = 192 + 64)."""
        self._run_and_compare(
            device,
            batch=4,
            num_heads=8,
            head_dim=256,
            kv_lora_rank=192,
            top_k=512,
            total_tokens=2048,
            atol=1e-2,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
