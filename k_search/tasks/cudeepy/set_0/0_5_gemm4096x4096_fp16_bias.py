import torch


class Gemm4096x4096Fp16Bias(torch.nn.Module):
    def __init__(self, *, out_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.out_dtype = out_dtype

    # a: (4096, 4096) fp16
    # b: (4096, 4096) fp16
    # bias: (4096,) fp16 (added to columns)
    # out: (4096, 4096) out_dtype
    def forward(self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        # Avoid FP32 upcast and unnecessary reshapes; 1D bias broadcasts across rows.
        c = torch.matmul(a, b)
        c = c + bias
        return c.to(self.out_dtype)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required for this reference problem"
    device = "cuda"
    a = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    bias = torch.randn(4096, device=device, dtype=torch.float16)
    model = Gemm4096x4096Fp16Bias(out_dtype=torch.float16).to(device)
    with torch.no_grad():
        out = model(a, b, bias)
    print(out.shape, out.dtype, out.device)

