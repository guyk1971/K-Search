import torch


class Gemm8192x1024x4096Fp16Acc32(torch.nn.Module):
    def __init__(self, *, out_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.out_dtype = out_dtype

    # a: (8192, 4096) fp16
    # b: (4096, 1024) fp16
    # out: (8192, 1024) out_dtype (FP32 accumulation is handled internally by cuBLAS)
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        c = torch.matmul(a, b)
        return c.to(self.out_dtype)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required for this reference problem"
    device = "cuda"
    a = torch.randn(8192, 4096, device=device, dtype=torch.float16)
    b = torch.randn(4096, 1024, device=device, dtype=torch.float16)
    model = Gemm8192x1024x4096Fp16Acc32(out_dtype=torch.float16).to(device)
    with torch.no_grad():
        out = model(a, b)
    print(out.shape, out.dtype, out.device)

