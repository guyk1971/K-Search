import torch


class Gemm4096x4096Fp8E4m3Acc32(torch.nn.Module):
    def __init__(self, *, out_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.out_dtype = out_dtype
        # Use unit scaling for an "unscaled" FP8 GEMM reference.
        # Register as buffers so they move with `.to(device)` and don't reallocate each call.
        self.register_buffer("_scale_a", torch.tensor(1.0, dtype=torch.float32), persistent=False)
        self.register_buffer("_scale_b", torch.tensor(1.0, dtype=torch.float32), persistent=False)

    # a: (4096, 4096) fp8 e4m3fn
    # b: (4096, 4096) fp8 e4m3fn
    # out: (4096, 4096) out_dtype
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Prefer cuBLAS FP8 path via torch._scaled_mm when available.
        if hasattr(torch, "_scaled_mm"):
            a_contig = a.contiguous()
            # torch._scaled_mm expects B provided as a transposed (N,K) tensor; it computes A @ B.T
            b_t = b.contiguous().transpose(0, 1).contiguous()
            return torch._scaled_mm(a_contig, b_t, scale_a=self._scale_a, scale_b=self._scale_b, out_dtype=self.out_dtype)

        # Fallback: whatever matmul path this build supports for float8.
        return torch.matmul(a, b).to(self.out_dtype)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required for this reference problem"
    assert hasattr(torch, "float8_e4m3fn"), "This PyTorch build does not expose float8_e4m3fn"
    device = "cuda"
    a = torch.randn(4096, 4096, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
    b = torch.randn(4096, 4096, device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
    model = Gemm4096x4096Fp8E4m3Acc32(out_dtype=torch.float16).to(device)
    with torch.no_grad():
        out = model(a, b)
    print(out.shape, out.dtype, out.device)

