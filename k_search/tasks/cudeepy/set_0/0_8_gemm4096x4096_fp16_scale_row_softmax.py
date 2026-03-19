import torch


class Gemm4096x4096Fp16ScaleRowSoftmax(torch.nn.Module):
    def __init__(self, *, scale: float = 1.0, out_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.scale = float(scale)
        self.out_dtype = out_dtype

    # a: (4096, 4096) fp16
    # b: (4096, 4096) fp16
    # out: (4096, 4096) out_dtype
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(a, b) * self.scale
        x = torch.softmax(x, dim=1)
        return x.to(self.out_dtype)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required for this reference problem"
    device = "cuda"
    a = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    # knob: softmax scale (useful for attention-like score matrices)
    scale = 1.0 / (128.0**0.5)
    model = Gemm4096x4096Fp16ScaleRowSoftmax(scale=scale, out_dtype=torch.float16).to(device)
    with torch.no_grad():
        out = model(a, b)
    print(out.shape, out.dtype, out.device, "row_sum_mean", out.float().sum(dim=1).mean().item())

