import torch


class Gemm4096x4096Fp16Acc32(torch.nn.Module):
    def __init__(self, *, out_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.out_dtype = out_dtype

    # a: (4096, 4096) fp16
    # b: (4096, 4096) fp16
    # out: (4096, 4096) out_dtype (FP32 accumulation is handled internally by cuBLAS)
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # cuBLAS uses Tensor Cores for fp16 GEMM and accumulates in FP32 by default.
        c = torch.matmul(a, b)
        return c.to(self.out_dtype)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required for this reference problem"
    device = "cuda"
    a = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    model = Gemm4096x4096Fp16Acc32(out_dtype=torch.float16).to(device)
    with torch.no_grad():
        out = model(a, b)
    print(out.shape, out.dtype, out.device)

