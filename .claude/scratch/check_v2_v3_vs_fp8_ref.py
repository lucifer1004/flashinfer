"""Compare v2 + v3 to a reference that uses the SAME FP8-quantized cache (so quant noise is matched)."""
import os
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
import sys
sys.path.insert(0, os.path.dirname(__file__))
import torch
import flashinfer
from flashinfer.jit.sparse_mla_sm120 import gen_sparse_mla_sm120_module
from test_decode_v3 import quantize_kv_model1


def dequant_kv(kv_flat_u8, num_blocks=1024, bs=64):
    """Dequantize the FP8 footer-packed cache back to bf16 [s_kv, D_QK]."""
    D_NOPE, D_ROPE, QUANT_TILE = 448, 64, 64
    NUM_SCALES = 7
    IO_STRIDE = D_NOPE + D_ROPE * 2  # 576
    SCALE_BYTES = NUM_SCALES + 1     # 8
    D_QK = D_NOPE + D_ROPE
    bpt = IO_STRIDE + SCALE_BYTES    # 584
    device = kv_flat_u8.device

    s_kv = num_blocks * bs
    result = torch.zeros(s_kv, D_QK, dtype=torch.bfloat16, device=device)
    flat = kv_flat_u8.view(num_blocks, bs * bpt)  # already in this shape
    for b in range(num_blocks):
        for tok in range(bs):
            # FP8 NoPE
            fp8_bytes = flat[b, tok * IO_STRIDE : tok * IO_STRIDE + D_NOPE]
            # RoPE bf16
            rope_bytes = flat[b, tok * IO_STRIDE + D_NOPE : tok * IO_STRIDE + IO_STRIDE]
            rope_bf16 = rope_bytes.view(torch.bfloat16)
            # Scale bytes (UE8M0)
            scale_off = bs * IO_STRIDE + tok * SCALE_BYTES
            scale_bytes = flat[b, scale_off : scale_off + SCALE_BYTES]
            # Dequant FP8 per tile
            slot_idx = b * bs + tok
            for ti in range(NUM_SCALES):
                fp8_tile = fp8_bytes[ti * QUANT_TILE : (ti + 1) * QUANT_TILE]
                scale = 2.0 ** (int(scale_bytes[ti].item()) - 127)
                # Decode FP8 e4m3 -> float
                fp8_view = fp8_tile.view(torch.float8_e4m3fn)
                bf16_tile = (fp8_view.float() * scale).to(torch.bfloat16)
                result[slot_idx, ti * QUANT_TILE : (ti + 1) * QUANT_TILE] = bf16_tile
            result[slot_idx, D_NOPE:] = rope_bf16
    return result


def reference_attention_fp8_aware(q, kv_deq_bf16, indices, sm_scale):
    T, H, D = q.shape
    kv_gathered = kv_deq_bf16[indices.long()].float()  # [T, topk, D]
    qf = q.float()
    scores = torch.einsum("thd,tkd->thk", qf, kv_gathered) * sm_scale
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("thk,tkd->thd", weights, kv_gathered)
    return out


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")
    T = 24
    num_heads = 128
    topk = 512
    d_qk = d_v = 512
    num_blocks = 1024
    s_kv = num_blocks * 64

    kv_bf16 = (torch.randn(num_blocks, 64, 1, d_qk, device=device, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
    kv_flat = quantize_kv_model1(kv_bf16)
    kv_4d = kv_flat.view(num_blocks, 64, 1, kv_flat.size(1) // 64)
    q = (torch.randn(T, num_heads, d_qk, device=device, dtype=torch.bfloat16) / 10.0).clamp(-1, 1)
    indices = torch.randint(0, s_kv, (T, topk), device=device, dtype=torch.int32)
    sm_scale = d_qk ** -0.5

    # Dequant the FP8 cache -> bf16 reference
    kv_deq = dequant_kv(kv_flat, num_blocks=num_blocks, bs=64)
    out_ref = reference_attention_fp8_aware(q, kv_deq, indices, sm_scale)

    # v2
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    output_v2 = torch.empty(T, num_heads, d_v, dtype=torch.bfloat16, device=device)
    out_lse_v2 = torch.empty(T, num_heads, dtype=torch.float32, device=device)
    flashinfer.sparse_mla_sm120_paged_attention(q=q, kv_cache=kv_4d, indices=indices, output=output_v2, out_lse=out_lse_v2, workspace_buffer=workspace, sm_scale=sm_scale, d_v=d_v)

    # v3
    module = gen_sparse_mla_sm120_module().build_and_load()
    num_splits = 8
    mid = torch.empty(T, num_heads, num_splits, d_v, dtype=torch.bfloat16, device=device)
    mid_lse = torch.empty(T, num_heads, num_splits, dtype=torch.float32, device=device)
    output_v3 = torch.empty(T, num_heads, d_v, dtype=torch.bfloat16, device=device)
    out_lse_v3 = torch.empty(T, num_heads, dtype=torch.float32, device=device)
    module.sparse_mla_sm120_decode_v3(q, kv_flat, indices, mid, mid_lse, output_v3, out_lse_v3, num_splits, sm_scale, None)
    torch.cuda.synchronize()

    diff_v2 = (output_v2.float() - out_ref).abs()
    diff_v3 = (output_v3.float() - out_ref).abs()
    print("=== T=24, h=128, topk=512 ===")
    print(f"v2 vs FP8-aware ref: max={diff_v2.max().item():.4e}, mean={diff_v2.mean().item():.4e}")
    print(f"v3 vs FP8-aware ref: max={diff_v3.max().item():.4e}, mean={diff_v3.mean().item():.4e}")
    per_tok_v2 = diff_v2.view(T, -1).max(dim=-1).values
    per_tok_v3 = diff_v3.view(T, -1).max(dim=-1).values
    print(f"v2 per-token max: min={per_tok_v2.min().item():.4e} max={per_tok_v2.max().item():.4e}")
    print(f"v3 per-token max: min={per_tok_v3.min().item():.4e} max={per_tok_v3.max().item():.4e}")
    # Broken tokens (err > 1e-2)
    broken_v2 = (per_tok_v2 > 1e-2).nonzero().flatten().tolist()
    broken_v3 = (per_tok_v3 > 1e-2).nonzero().flatten().tolist()
    print(f"v2 broken tokens (>1e-2 err vs ref): {broken_v2}")
    print(f"v3 broken tokens (>1e-2 err vs ref): {broken_v3}")


main()
