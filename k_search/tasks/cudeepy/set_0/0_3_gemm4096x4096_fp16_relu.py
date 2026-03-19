import torch
import torch.nn.functional as F


class Gemm4096x4096Fp16Relu(torch.nn.Module):
    def __init__(self, *, out_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.out_dtype = out_dtype

    # a: (4096, 4096) fp16
    # b: (4096, 4096) fp16
    # out: (4096, 4096) out_dtype
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Keep everything in fp16 for maximum throughput; cuBLAS accumulates internally in FP32.
        c = F.relu(torch.matmul(a, b))
        return c.to(self.out_dtype)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required for this reference problem"
    device = "cuda"
    a = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    model = Gemm4096x4096Fp16Relu(out_dtype=torch.float16).to(device)
    with torch.no_grad():
        out = model(a, b)
    print(out.shape, out.dtype, out.device)

