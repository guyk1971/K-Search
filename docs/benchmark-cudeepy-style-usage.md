# cudeepy-Style Benchmark: Usage Guide

This guide explains how to use the cross-system benchmark framework to evaluate K-Search generated kernels using cudeepy's evaluation methodology — enabling fair performance comparison between both systems.

## Quick Start

```bash
# Benchmark a K-Search kernel on a cudeepy GEMM problem
python3 scripts/benchmark_cudeepy_style.py \
    --problem /path/to/cudeepy/problems/set_0/0_0_gemm4096x4096_fp16_acc32.py \
    --kernel my_submission.py
```

---

## Prerequisites

- GPU access (local or via Docker container)
- PyTorch with CUDA support
- K-Search repository checked out
- (For cudeepy problems) Access to the cudeepy problems directory

---

## Concepts

### What This Framework Does

| Aspect | K-Search default eval | This framework |
|--------|----------------------|----------------|
| Timing | Per-iteration CUDA events (sync between each call) | Batch CUDA events (cudeepy-style, no sync overhead) |
| Reference | Not timed | Timed via `torch.compile(mode="max-autotune")` |
| Reports | Absolute latency + 1/latency score | Latency + speedup vs reference + TFLOPS |
| TFLOPS | Not reported | Auto-detected for GEMM problems |

### Kernel Interface

For cudeepy problems, your K-Search kernel must define `custom_kernel` with the **same positional arguments** as the cudeepy reference `forward()`:

```python
# For a GEMM problem where forward(self, a, b) -> Tensor:
def custom_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Your optimized implementation
    ...

# For a GEMM+bias problem where forward(self, a, b, bias) -> Tensor:
def custom_kernel(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    ...
```

Check the cudeepy problem file's `forward()` signature to see what arguments are expected.

---

## Use Case 1: Single cudeepy Problem

Benchmark your kernel against one specific problem:

```bash
python3 scripts/benchmark_cudeepy_style.py \
    --problem ~/code/gitlab/clones/cudeepy/problems/set_0/0_0_gemm4096x4096_fp16_acc32.py \
    --kernel my_gemm_kernel.py
```

**Sample output:**
```
Kernel: my_gemm_kernel
Timing: warmup=10, iters=50
Problems: 1
  [PASS] 0_0_gemm4096x4096_fp16_acc32 — 0.241ms (1.137x)  571.2 TFLOPS

==========================================================================================
  Benchmark: cudeepy-style evaluation
  Kernel: my_gemm_kernel
==========================================================================================

  CORRECTNESS
  ──────────────────────────────────────────────────────────────────────────────────────────
  [PASS] 0_0_gemm4096x4096_fp16_acc32          max_err=0.031250  mean_err=0.004120

  PERFORMANCE (batch CUDA event timing, warmup=10, iters=50)
  ──────────────────────────────────────────────────────────────────────────────────────────
  Problem                                  Ref (ms)  Kernel (ms)  Speedup  Ref TFLOPS  Knl TFLOPS
  0_0_gemm4096x4096_fp16_acc32                0.274        0.241   1.137x       501.1       570.2
==========================================================================================
```

---

## Use Case 2: All Problems in a Set

Benchmark your kernel across an entire problem set. Useful for evaluating how well a kernel generalizes:

```bash
python3 scripts/benchmark_cudeepy_style.py \
    --problem-dir ~/code/gitlab/clones/cudeepy/problems/set_0/ \
    --kernel my_gemm_kernel.py
```

Problems that fail to load (e.g., incompatible interface) are skipped with a `[SKIP]` message. The report shows per-problem results plus geometric and arithmetic mean aggregates.

---

## Use Case 3: Multiple Specific Problems

Cherry-pick problems from different sets:

```bash
python3 scripts/benchmark_cudeepy_style.py \
    --problem \
        ~/code/gitlab/clones/cudeepy/problems/set_0/0_0_gemm4096x4096_fp16_acc32.py \
        ~/code/gitlab/clones/cudeepy/problems/set_0/0_3_gemm4096x4096_fp16_relu.py \
        ~/code/gitlab/clones/cudeepy/problems/set_1/1_0_gemm_256x256x32768_fp16.py \
    --kernel my_gemm_kernel.py
```

---

## Use Case 4: From a K-Search Solution JSON

If you have a saved K-Search solution (generated during optimization runs):

```bash
python3 scripts/benchmark_cudeepy_style.py \
    --problem ~/code/gitlab/clones/cudeepy/problems/set_0/0_0_gemm4096x4096_fp16_acc32.py \
    --solution-json .ksearch-output/gpumode_trimul/solutions/gpumode_trimul/my_solution.json
```

This works for both Python/Triton and CUDA solutions — CUDA solutions are automatically compiled via `torch.utils.cpp_extension.load`.

---

## Use Case 5: TriMul (K-Search Native Task)

Backward-compatible mode for evaluating TriMul kernels with cudeepy-style timing:

```bash
# Benchmark the baseline reference implementation
python3 scripts/benchmark_cudeepy_style.py --task trimul

# Benchmark a specific submission
python3 scripts/benchmark_cudeepy_style.py \
    --task trimul \
    --kernel path/to/my_trimul_submission.py

# From a solution JSON
python3 scripts/benchmark_cudeepy_style.py \
    --task trimul \
    --solution-json path/to/solution.json
```

TriMul mode runs all 7 benchmark specs from `task_orig.yml` (seqlen 256–1024, dim 128–384, normal+cauchy distributions) and reports geometric mean across all specs.

---

## Use Case 6: Sanity Check (Reference vs Itself)

Omit `--kernel` to benchmark the reference against itself. Useful to verify the framework works and see baseline TFLOPS:

```bash
python3 scripts/benchmark_cudeepy_style.py \
    --problem ~/code/gitlab/clones/cudeepy/problems/set_0/0_0_gemm4096x4096_fp16_acc32.py
```

Expected: speedup ~1.000x (reference timed against itself).

---

## Use Case 7: Compare Timing Methodologies

Use `--per-iter` to see the difference between cudeepy's batch timing and K-Search's per-iteration timing:

```bash
python3 scripts/benchmark_cudeepy_style.py \
    --problem ~/code/gitlab/clones/cudeepy/problems/set_0/0_0_gemm4096x4096_fp16_acc32.py \
    --kernel my_kernel.py \
    --per-iter
```

**Additional output:**
```
Per-iteration timing (K-Search style):
  0_0_gemm4096x4096_fp16_acc32  mean=0.253ms  std=0.008ms  best=0.241ms  worst=0.284ms  (batch=0.241ms, delta=+0.012ms)
```

The `delta` shows how much per-iteration sync overhead adds to the measurement. For fast kernels (<1ms), expect 5–15% overhead.

---

## Options Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--problem FILE [FILE ...]` | — | cudeepy problem file(s) to benchmark against |
| `--problem-dir DIR` | — | Directory of cudeepy problem files (runs all) |
| `--task {trimul}` | — | Built-in K-Search task |
| `--kernel FILE` | — | Path to kernel `submission.py` with `custom_kernel()` |
| `--solution-json FILE` | — | Path to K-Search solution JSON |
| `--warmup N` | 10 | Warmup iterations (not timed) |
| `--iters N` | 50 | Timed iterations |
| `--atol F` | auto | Override absolute tolerance (auto: 0.1 for FP16, 0.5 for FP8) |
| `--rtol F` | auto | Override relative tolerance |
| `--flops N` | auto | Override FLOP count for TFLOPS (auto-detected for GEMMs) |
| `--no-compile-ref` | false | Use eager PyTorch reference instead of `torch.compile` |
| `--seed N` | 42 | Random seed for reproducibility |
| `--per-iter` | false | Also show per-iteration timing stats |

---

## Writing a K-Search Kernel for cudeepy Problems

### Step 1: Inspect the Problem

Open the cudeepy problem file and check:
- **Input shapes and dtypes** — in the `__main__` block
- **`forward()` signature** — your `custom_kernel` must match this
- **Operation** — what computation is being done

Example (`0_0_gemm4096x4096_fp16_acc32.py`):
```python
class Gemm4096x4096Fp16Acc32(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        c = torch.matmul(a, b)
        return c.to(self.out_dtype)

if __name__ == "__main__":
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
```

### Step 2: Write Your Kernel

Create a `submission.py` with matching interface:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(...):
    # Your Triton GEMM implementation
    ...

def custom_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=torch.float16)
    # Launch your Triton kernel
    _matmul_kernel[grid](a, b, c, M, N, K, ...)
    return c
```

### Step 3: Benchmark

```bash
python3 scripts/benchmark_cudeepy_style.py \
    --problem ~/code/gitlab/clones/cudeepy/problems/set_0/0_0_gemm4096x4096_fp16_acc32.py \
    --kernel submission.py
```

### Step 4: Test Across Shapes

Try different problem shapes to verify generalization:

```bash
python3 scripts/benchmark_cudeepy_style.py \
    --problem \
        ~/code/gitlab/clones/cudeepy/problems/set_0/0_0_gemm4096x4096_fp16_acc32.py \
        ~/code/gitlab/clones/cudeepy/problems/set_0/0_2_gemm8192x1024x4096_fp16_acc32.py \
        ~/code/gitlab/clones/cudeepy/problems/set_1/1_0_gemm_256x256x32768_fp16.py \
    --kernel submission.py
```

---

## Comparing K-Search vs cudeepy Kernels

To compare kernels from both systems on the same problem:

```bash
# 1. Benchmark the K-Search kernel
python3 scripts/benchmark_cudeepy_style.py \
    --problem ~/code/gitlab/clones/cudeepy/problems/set_0/0_0_gemm4096x4096_fp16_acc32.py \
    --kernel ksearch_gemm_submission.py

# 2. Benchmark the cudeepy kernel (if it exports custom_kernel)
#    Or note cudeepy's TFLOPS/speedup from its own test_harness.py output

# 3. Compare: both are measured against the same torch.compile reference
#    with the same timing methodology, so speedup numbers are directly comparable.
```

The key insight: both measurements use `torch.compile(mode="max-autotune")` as the reference, batch CUDA event timing, and the same input data. The speedup numbers are directly comparable.

---

## Running K-Search Optimization on cudeepy Problems

Beyond benchmarking existing kernels, you can use K-Search's full LLM optimization loop to **generate** optimized kernels for cudeepy problems. This uses `--task-source cudeepy` in the main entry point.

### Basic Usage

```bash
python3 generate_kernels_and_eval.py \
    --task-source cudeepy \
    --cudeepy-problem ~/code/gitlab/clones/cudeepy/problems/set_0/0_0_gemm4096x4096_fp16_acc32.py \
    --model-name gpt-5 \
    --language triton \
    --target-gpu H100 \
    --max-opt-rounds 10 \
    --save-solutions
```

This will:
1. Load the cudeepy problem (extract reference, input shapes, etc.)
2. Build an LLM prompt describing the operation and interface
3. Ask the LLM to generate an optimized Triton kernel
4. Evaluate it (correctness + cudeepy-style batch timing + speedup vs torch.compile)
5. Feed performance results back to the LLM for the next optimization round
6. Repeat for `--max-opt-rounds` iterations, tracking the best solution

### With World Model

```bash
python3 generate_kernels_and_eval.py \
    --task-source cudeepy \
    --cudeepy-problem ~/code/gitlab/clones/cudeepy/problems/set_0/0_0_gemm4096x4096_fp16_acc32.py \
    --model-name gpt-5 \
    --language triton \
    --target-gpu H100 \
    --world-model \
    --max-opt-rounds 20 \
    --save-solutions \
    --artifacts-dir .ksearch-cudeepy
```

### cudeepy-Specific Options

| Flag | Default | Description |
|------|---------|-------------|
| `--cudeepy-problem PATH` | required | Path to the cudeepy problem .py file |
| `--cudeepy-warmup N` | 10 | Warmup iterations for benchmarking |
| `--cudeepy-iters N` | 50 | Timed iterations for benchmarking |
| `--cudeepy-no-compile-ref` | false | Use eager reference instead of torch.compile |
| `--cudeepy-atol F` | auto | Override correctness absolute tolerance |
| `--cudeepy-rtol F` | auto | Override correctness relative tolerance |

### What the LLM Sees

The prompt sent to the LLM includes:
- The problem name and class
- The reference `forward()` source code
- Input tensor shapes and dtypes
- Output shape and dtype
- FLOPS count (auto-detected for GEMMs)
- The required kernel interface: `def custom_kernel(a, b) -> torch.Tensor`
- Correctness tolerances

### Scoring

The kernel is scored by **speedup vs torch.compile reference** (higher = better). This is stored in `EvalResult.metrics["score"]` and used by the world model for action selection.

### Post-Hoc Comparison

After K-Search generates a kernel, compare it against cudeepy's kernel using the benchmark script:

```bash
# Find the best solution saved by K-Search
ls .ksearch-cudeepy/*/solutions/*/

# Benchmark it
python3 scripts/benchmark_cudeepy_style.py \
    --problem ~/code/gitlab/clones/cudeepy/problems/set_0/0_0_gemm4096x4096_fp16_acc32.py \
    --solution-json .ksearch-cudeepy/.../best_solution.json
```

---

## Troubleshooting

### "No nn.Module subclass found"

The problem file doesn't define a class inheriting from `torch.nn.Module`. Check the file — it might use a different pattern.

### "No CUDA input tensors found"

The `__main__` block didn't create any CUDA tensors. Ensure your GPU is available and the problem file creates tensors on `device="cuda"`.

### "No `custom_kernel` function found"

Your kernel file must define a function named exactly `custom_kernel`.

### Correctness failures with tight tolerances

Override with `--atol` and `--rtol`:
```bash
python3 scripts/benchmark_cudeepy_style.py \
    --problem problem.py --kernel kernel.py \
    --atol 0.5 --rtol 1e-2
```

### torch.compile hangs or is slow

Use `--no-compile-ref` to fall back to eager PyTorch (numbers won't match cudeepy's compiled reference, but useful for debugging):
```bash
python3 scripts/benchmark_cudeepy_style.py \
    --problem problem.py --kernel kernel.py \
    --no-compile-ref
```
