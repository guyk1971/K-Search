import torch


class Gemm4096x4096Bf16Acc32(torch.nn.Module):
    def __init__(self, *, out_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.out_dtype = out_dtype

    # a: (4096, 4096) bf16
    # b: (4096, 4096) bf16
    # out: (4096, 4096) out_dtype (FP32 accumulation is handled internally by cuBLAS)
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        c = torch.matmul(a, b)
        return c.to(self.out_dtype)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required for this reference problem"
    device = "cuda"
    a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    model = Gemm4096x4096Bf16Acc32(out_dtype=torch.bfloat16).to(device)
    with torch.no_grad():
        out = model(a, b)
    print(out.shape, out.dtype, out.device)

