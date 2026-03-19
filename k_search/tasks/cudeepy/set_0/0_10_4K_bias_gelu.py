import torch
import torch.nn.functional as F

def gemm_gelu_reference(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    GEMM with fused GELU activation and bias addition.
    
    Args:
        A: Input matrix of shape (M, K), dtype=float16
        B: Weight matrix of shape (N, K), dtype=float16 (will be transposed)
        bias: Bias vector of shape (N,), dtype=float16
    
    Returns:
        Output matrix of shape (M, N), dtype=float16
        
    Formula: C = GELU(A @ B.T + bias)
    """
    # Use FP16 cuBLAS matmul for optimal GEMM performance
    # torch.matmul with FP16 inputs uses cuBLAS TensorCore ops
    result = torch.matmul(A, B.T)  # (M, N) - FP16 cuBLAS
    # Bias and GELU in FP32 for numerical stability
    result = result.float() + bias.float()  # Broadcast bias in FP32
    result = F.gelu(result)                  # GELU activation
    return result.to(torch.float16)


# Problem dimensions
M, K, N = 4096, 4096, 4096
A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(N, K, device="cuda", dtype=torch.float16)  
bias = torch.randn(N, device="cuda", dtype=torch.float16)

# Reference output
C_ref = gemm_gelu_reference(A, B, bias)

