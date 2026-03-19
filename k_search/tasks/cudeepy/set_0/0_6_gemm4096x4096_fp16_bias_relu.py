import torch
import torch.nn.functional as F


class Gemm4096x4096Fp16BiasRelu(torch.nn.Module):
    def __init__(self, *, out_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.out_dtype = out_dtype

    # a: (4096, 4096) fp16
    # b: (4096, 4096) fp16
    # bias: (4096,) fp16 (added to columns)
    # out: (4096, 4096) out_dtype
    def forward(self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        c = torch.matmul(a, b)
        c = c + bias
        c = F.relu(c)
        return c.to(self.out_dtype)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required for this reference problem"
    device = "cuda"
    a = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    bias = torch.randn(4096, device=device, dtype=torch.float16)
    model = Gemm4096x4096Fp16BiasRelu(out_dtype=torch.float16).to(device)
    with torch.no_grad():
        out = model(a, b, bias)
    print(out.shape, out.dtype, out.device)

