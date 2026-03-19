import torch


class Gemm4096x4096Fp16RowSoftmax(torch.nn.Module):
    def __init__(self, *, softmax_dim: int = 1, out_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.softmax_dim = softmax_dim
        self.out_dtype = out_dtype

    # a: (4096, 4096) fp16
    # b: (4096, 4096) fp16
    # out: (4096, 4096) out_dtype (row-wise softmax by default)
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Use PyTorch's fused softmax kernel (faster than manual exp/sum).
        x = torch.matmul(a, b)
        x = torch.softmax(x, dim=self.softmax_dim)
        return x.to(self.out_dtype)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required for this reference problem"
    device = "cuda"
    a = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    model = Gemm4096x4096Fp16RowSoftmax(softmax_dim=1, out_dtype=torch.float16).to(device)
    with torch.no_grad():
        out = model(a, b)
    print(out.shape, out.dtype, out.device, "row_sum_mean", out.float().sum(dim=1).mean().item())

