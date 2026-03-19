# Task-Agnostic Benchmark Framework (cudeepy-Style Evaluation)

**Date:** 2026-03-19
**Status:** Approved

## Motivation

K-Search and cudeepy both generate GPU kernels but use different evaluation methodologies:

- **K-Search**: Per-iteration CUDA events with `torch.cuda.synchronize()` between each call. Reports absolute latency only (no reference timing, no TFLOPS).
- **cudeepy**: Batch CUDA events (single start/end around N iterations). Times both kernel and reference (`torch.compile`). Reports speedup and TFLOPS.

The numbers are not directly comparable. This framework enables fair cross-system comparison by:
1. Providing a generic benchmark library using cudeepy's timing methodology
2. Loading cudeepy problem definitions directly (nn.Module + input specs)
3. Benchmarking K-Search generated kernels against the same reference

## Design

### File 1: `k_search/eval/benchmark.py` — Generic Benchmark Library

Task-agnostic. No imports from trimul, GPUMode, FlashInfer, or cudeepy.

#### Data Structures

```python
@dataclass
class ProblemSpec:
    name: str                          # e.g. "gemm4096x4096_fp16_acc32"
    reference_fn: Callable             # model.forward or torch.compile'd version
    inputs: list[torch.Tensor]         # concrete GPU tensors
    input_names: list[str]             # ["a", "b"] or ["a", "b", "bias"]
    expected_output: torch.Tensor      # reference output for correctness check
    atol: float                        # correctness tolerance (absolute)
    rtol: float                        # correctness tolerance (relative)
    flops: int | None                  # for TFLOPS calculation (None = skip)
    model_instance: nn.Module | None   # keep alive to prevent GC of parameters

@dataclass
class BenchmarkResult:
    name: str
    correct: bool
    max_abs_err: float
    mean_abs_err: float
    ref_ms: float
    kernel_ms: float
    speedup: float
    ref_tflops: float | None
    kernel_tflops: float | None
    error_msg: str
```

#### Core Functions

- `benchmark_batch_events(fn, warmup, iters) -> float` — Batch CUDA event timing (cudeepy-style). Returns ms per iteration.
- `check_correctness(kernel_out, expected, atol, rtol) -> (bool, max_err, mean_err, msg)` — Element-wise comparison with tolerance.
- `run_benchmark(spec, kernel_fn, warmup, iters) -> BenchmarkResult` — Full correctness + timing for one problem.
- `print_report(results, kernel_label)` — Formatted table with per-problem rows, geometric/arithmetic mean aggregates, TFLOPS columns.
- `compute_tflops(flops, time_ms) -> float | None` — TFLOPS from FLOP count and latency.

### File 2: `scripts/benchmark_cudeepy_style.py` — CLI Tool

#### Usage

```bash
# Single cudeepy problem
python3 scripts/benchmark_cudeepy_style.py \
    --problem path/to/cudeepy/problems/set_0/0_0_gemm4096x4096_fp16_acc32.py \
    --kernel path/to/submission.py

# All problems in a directory
python3 scripts/benchmark_cudeepy_style.py \
    --problem-dir path/to/cudeepy/problems/set_0/ \
    --kernel path/to/submission.py

# From a K-Search solution JSON
python3 scripts/benchmark_cudeepy_style.py \
    --problem path/to/problem.py \
    --solution-json path/to/solution.json

# TriMul backward compatibility
python3 scripts/benchmark_cudeepy_style.py \
    --task trimul --kernel path/to/submission.py

# Options
--warmup N        # warmup iterations (default: 10)
--iters N         # timed iterations (default: 50)
--atol F          # override correctness tolerance
--rtol F          # override correctness tolerance
--flops N         # override FLOP count for TFLOPS calculation
--no-compile-ref  # skip torch.compile on reference (use eager)
--seed N          # random seed (default: 42)
```

#### Problem Loading (`load_cudeepy_problem`)

1. Import the problem `.py` file as a module
2. Find the `nn.Module` subclass (first class defined in the module that inherits `nn.Module`)
3. Execute the `__main__` block in a controlled namespace to capture:
   - Input tensors (all `torch.Tensor` locals on CUDA)
   - Model instance
   - Model constructor kwargs
4. Run reference: `torch.compile(model, mode="max-autotune")` with warmup
5. Compute expected output
6. Auto-detect FLOPS for GEMM-shaped problems: `2 * M * N * K` (for 2D matmul inputs)
7. Auto-detect atol/rtol from dtype (FP16→0.1/1e-5, FP8→0.5/1e-2)
8. Return `ProblemSpec`

#### Kernel Loading

- `--kernel path.py`: Import `custom_kernel` from the file. Called as `custom_kernel(a, b, ...)` — same positional args as reference `forward()`.
- `--solution-json path.json`: Extract submission source from K-Search solution JSON, write to temp file, import.

#### TriMul Mode (`--task trimul`)

When `--task trimul`:
- Load benchmark specs from `task_orig.yml` (7 specs)
- Use `generate_input()` from trimul's `reference.py`
- Build `ProblemSpec` per benchmark spec
- Wrap kernel call: `kernel_fn = lambda *args: custom_kernel(args)` (tuple convention)
- Run all 7 specs, report per-spec + geometric mean

### Kernel Interface Convention

For cudeepy problems, K-Search kernels follow interface **(A)**:

```python
def custom_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Same positional args as cudeepy reference forward()."""
    ...
```

This matches cudeepy naturally. TriMul's legacy `custom_kernel(data)` tuple convention is handled by the `--task trimul` adapter.

### FLOPS Auto-Detection

For standard GEMM problems, FLOPS are auto-detected from input shapes:
- 2D inputs `(M, K)` and `(K, N)` → `2 * M * N * K`
- 3D batched `(B, M, K)` and `(B, K, N)` → `2 * B * M * N * K`
- Non-GEMM: FLOPS not auto-detected; use `--flops` or TFLOPS columns show `—`

### Default Correctness Tolerances

| Input dtype | atol | rtol |
|---|---|---|
| float16 | 0.1 | 1e-5 |
| bfloat16 | 0.1 | 1e-2 |
| float8_e4m3fn | 0.5 | 1e-2 |
| float32 | 1e-4 | 1e-5 |

Overridable via `--atol` / `--rtol`.

### Output Format

```
================================================================================
  Benchmark: cudeepy-style evaluation
  Kernel: submission.py
================================================================================

  CORRECTNESS
  ──────────────────────────────────────────────────────────────────────
  [PASS] gemm4096x4096_fp16_acc32     max_err=0.031250  mean_err=0.004120
  [PASS] gemm4096x4096_fp16_relu      max_err=0.028320  mean_err=0.003890

  PERFORMANCE (batch CUDA event timing, warmup=10, iters=50)
  ──────────────────────────────────────────────────────────────────────
  Problem                    Ref (ms)  Kernel (ms)  Speedup  Ref TFLOPS  Kernel TFLOPS
  gemm4096x4096_fp16_acc32      0.274        0.236   1.161x       501.1          583.5
  gemm4096x4096_fp16_relu       0.275        0.241   1.141x       499.3          570.2
  ─────────────────────────────────────────────────────────────────────────────────────
  Geometric mean                0.274        0.238   1.151x       500.2          576.8
================================================================================
```

## Not Included

- No task registry or plugin system
- No YAML config files
- No automatic kernel generation
- No cudeepy CuTe tensor conversion (K-Search kernels work with PyTorch tensors)
