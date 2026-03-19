"""Task-agnostic GPU kernel benchmark library.

Provides cudeepy-style batch CUDA event timing, correctness checking,
TFLOPS calculation, and formatted reporting. No task-specific code.

Usage:
    from k_search.eval.benchmark import ProblemSpec, run_benchmark, print_report
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProblemSpec:
    """A self-contained benchmark problem: reference, inputs, tolerances."""

    name: str
    reference_fn: Callable          # callable(*inputs) -> Tensor
    inputs: list[torch.Tensor]      # concrete GPU tensors
    input_names: list[str]          # e.g. ["a", "b", "bias"]
    expected_output: torch.Tensor   # reference output for correctness
    atol: float = 0.1
    rtol: float = 1e-5
    flops: int | None = None        # total FLOPs for TFLOPS calc (None = skip)
    model_instance: Any = None      # keep nn.Module alive to prevent GC


@dataclass
class BenchmarkResult:
    """Result of benchmarking one kernel on one problem."""

    name: str
    correct: bool
    max_abs_err: float = 0.0
    mean_abs_err: float = 0.0
    ref_ms: float = 0.0
    kernel_ms: float = 0.0
    speedup: float = 0.0
    ref_tflops: float | None = None
    kernel_tflops: float | None = None
    error_msg: str = ""


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def benchmark_batch_events(
    fn: Callable,
    warmup: int = 10,
    iters: int = 50,
) -> float:
    """Batch CUDA event timing (cudeepy methodology).

    Records a single start event, runs *iters* iterations, records end event.
    No per-iteration synchronization — lets the GPU pipeline freely.

    Returns:
        Mean time per iteration in milliseconds.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def benchmark_per_iter_events(
    fn: Callable,
    warmup: int = 10,
    iters: int = 50,
) -> dict[str, float]:
    """Per-iteration CUDA event timing (K-Search methodology).

    Returns:
        Dict with mean, std, best, worst in milliseconds.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times_ms.append(s.elapsed_time(e))

    mean = sum(times_ms) / len(times_ms)
    variance = sum((t - mean) ** 2 for t in times_ms) / max(len(times_ms) - 1, 1)
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "best": min(times_ms),
        "worst": max(times_ms),
    }


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def check_correctness(
    kernel_out: torch.Tensor,
    expected: torch.Tensor,
    atol: float = 0.1,
    rtol: float = 1e-5,
) -> tuple[bool, float, float, str]:
    """Element-wise comparison with tolerance.

    Returns:
        (passed, max_abs_err, mean_abs_err, message)
    """
    if kernel_out.shape != expected.shape:
        return (
            False, float("inf"), float("inf"),
            f"Shape mismatch: kernel {kernel_out.shape} vs expected {expected.shape}",
        )

    diff = (kernel_out.float() - expected.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()

    # Check tolerance: |diff| <= atol + rtol * |expected|
    tolerance = atol + rtol * expected.float().abs()
    violations = (diff > tolerance).sum().item()

    # Also check NaN/Inf mismatches
    nan_mismatch = torch.logical_xor(
        torch.isnan(kernel_out), torch.isnan(expected)
    ).sum().item()
    inf_mismatch = (
        torch.logical_xor(torch.isinf(kernel_out), torch.isinf(expected))
    ).sum().item()

    total_bad = violations + nan_mismatch + inf_mismatch
    if total_bad > 0:
        msg = f"{int(total_bad)} mismatched elements (max_err={max_err:.6f}, atol={atol}, rtol={rtol})"
        if nan_mismatch:
            msg += f", {int(nan_mismatch)} NaN mismatches"
        if inf_mismatch:
            msg += f", {int(inf_mismatch)} Inf mismatches"
        return False, max_err, mean_err, msg

    return True, max_err, mean_err, ""


# ---------------------------------------------------------------------------
# TFLOPS
# ---------------------------------------------------------------------------

def compute_tflops(flops: int | None, time_ms: float) -> float | None:
    """Compute TFLOPS from FLOP count and time in milliseconds."""
    if flops is None or flops <= 0 or time_ms <= 0:
        return None
    return flops / (time_ms * 1e-3) / 1e12


def estimate_gemm_flops(inputs: list[torch.Tensor]) -> int | None:
    """Auto-detect FLOPS for GEMM-shaped problems from input tensor shapes.

    Recognizes:
        - 2D: (M, K) x (K, N) -> 2*M*N*K
        - 3D batched: (B, M, K) x (B, K, N) -> 2*B*M*N*K

    Returns None if inputs don't look like a GEMM.
    """
    if len(inputs) < 2:
        return None

    a, b = inputs[0], inputs[1]
    if a.ndim == 2 and b.ndim == 2:
        M, K = a.shape
        K2, N = b.shape
        if K == K2:
            return 2 * M * N * K
    elif a.ndim == 3 and b.ndim == 3:
        B, M, K = a.shape
        B2, K2, N = b.shape
        if B == B2 and K == K2:
            return 2 * B * M * N * K
    return None


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def clone_inputs(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    """Deep-clone a list of GPU tensors."""
    return [t.clone() for t in inputs]


def run_benchmark(
    spec: ProblemSpec,
    kernel_fn: Callable,
    warmup: int = 10,
    iters: int = 50,
) -> BenchmarkResult:
    """Run correctness check + batch timing for one problem.

    Args:
        spec: Problem specification with reference, inputs, tolerances.
        kernel_fn: The kernel under test. Called as kernel_fn(*inputs).
        warmup: Warmup iterations for timing.
        iters: Timed iterations.

    Returns:
        BenchmarkResult with correctness, latency, speedup, TFLOPS.
    """
    # Correctness
    try:
        kernel_out = kernel_fn(*clone_inputs(spec.inputs))
        torch.cuda.synchronize()
    except Exception as e:
        return BenchmarkResult(
            name=spec.name, correct=False,
            error_msg=f"Kernel raised {type(e).__name__}: {e}",
        )

    passed, max_err, mean_err, msg = check_correctness(
        kernel_out, spec.expected_output, spec.atol, spec.rtol,
    )
    if not passed:
        return BenchmarkResult(
            name=spec.name, correct=False,
            max_abs_err=max_err, mean_abs_err=mean_err,
            error_msg=msg,
        )

    del kernel_out
    torch.cuda.empty_cache()

    # Time reference
    ref_ms = benchmark_batch_events(
        lambda: spec.reference_fn(*clone_inputs(spec.inputs)),
        warmup=warmup, iters=iters,
    )

    # Time kernel
    kernel_ms = benchmark_batch_events(
        lambda: kernel_fn(*clone_inputs(spec.inputs)),
        warmup=warmup, iters=iters,
    )

    speedup = ref_ms / kernel_ms if kernel_ms > 0 else 0.0
    ref_tflops = compute_tflops(spec.flops, ref_ms)
    kernel_tflops = compute_tflops(spec.flops, kernel_ms)

    return BenchmarkResult(
        name=spec.name, correct=True,
        max_abs_err=max_err, mean_abs_err=mean_err,
        ref_ms=ref_ms, kernel_ms=kernel_ms, speedup=speedup,
        ref_tflops=ref_tflops, kernel_tflops=kernel_tflops,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_tflops(v: float | None) -> str:
    return f"{v:>8.1f}" if v is not None else f"{'—':>8s}"


def print_report(
    results: list[BenchmarkResult],
    *,
    kernel_label: str = "kernel",
    warmup: int = 10,
    iters: int = 50,
) -> None:
    """Print formatted benchmark report to stdout."""
    sep = "=" * 90
    thin = "─" * 90

    print(f"\n{sep}")
    print(f"  Benchmark: cudeepy-style evaluation")
    print(f"  Kernel: {kernel_label}")
    print(sep)

    # Correctness
    print(f"\n{thin}")
    print("  CORRECTNESS")
    print(thin)
    for r in results:
        if r.correct:
            print(f"  [PASS] {r.name:<40s}  max_err={r.max_abs_err:.6f}  mean_err={r.mean_abs_err:.6f}")
        else:
            msg = r.error_msg[:60] if r.error_msg else "FAILED"
            print(f"  [FAIL] {r.name:<40s}  {msg}")

    # Performance
    has_tflops = any(r.kernel_tflops is not None for r in results if r.correct)
    print(f"\n{thin}")
    print(f"  PERFORMANCE (batch CUDA event timing, warmup={warmup}, iters={iters})")
    print(thin)

    if has_tflops:
        hdr = f"  {'Problem':<40s} {'Ref (ms)':>9s} {'Kernel (ms)':>11s} {'Speedup':>8s} {'Ref TFLOPS':>11s} {'Knl TFLOPS':>11s}"
        row_sep = f"  {'─'*40} {'─'*9} {'─'*11} {'─'*8} {'─'*11} {'─'*11}"
    else:
        hdr = f"  {'Problem':<40s} {'Ref (ms)':>9s} {'Kernel (ms)':>11s} {'Speedup':>8s}"
        row_sep = f"  {'─'*40} {'─'*9} {'─'*11} {'─'*8}"

    print(hdr)
    print(row_sep)

    for r in results:
        if r.correct:
            line = f"  {r.name:<40s} {r.ref_ms:>9.3f} {r.kernel_ms:>11.3f} {r.speedup:>7.3f}x"
            if has_tflops:
                line += f" {_fmt_tflops(r.ref_tflops):>11s} {_fmt_tflops(r.kernel_tflops):>11s}"
            print(line)
        else:
            line = f"  {r.name:<40s} {'FAIL':>9s} {'FAIL':>11s} {'—':>8s}"
            if has_tflops:
                line += f" {'—':>11s} {'—':>11s}"
            print(line)

    # Aggregates
    passed = [r for r in results if r.correct]
    if len(passed) > 1:
        geom_kernel = math.exp(sum(math.log(r.kernel_ms) for r in passed) / len(passed))
        geom_ref = math.exp(sum(math.log(r.ref_ms) for r in passed) / len(passed))
        geom_speedup = geom_ref / geom_kernel if geom_kernel > 0 else 0

        arith_kernel = sum(r.kernel_ms for r in passed) / len(passed)
        arith_ref = sum(r.ref_ms for r in passed) / len(passed)
        arith_speedup = arith_ref / arith_kernel if arith_kernel > 0 else 0

        print(row_sep)
        line = f"  {'Geometric mean':<40s} {geom_ref:>9.3f} {geom_kernel:>11.3f} {geom_speedup:>7.3f}x"
        if has_tflops:
            geom_ref_tf = compute_tflops(passed[0].flops, geom_ref) if passed[0].flops else None
            geom_knl_tf = compute_tflops(passed[0].flops, geom_kernel) if passed[0].flops else None
            line += f" {_fmt_tflops(geom_ref_tf):>11s} {_fmt_tflops(geom_knl_tf):>11s}"
        print(line)

        line = f"  {'Arithmetic mean':<40s} {arith_ref:>9.3f} {arith_kernel:>11.3f} {arith_speedup:>7.3f}x"
        if has_tflops:
            arith_ref_tf = compute_tflops(passed[0].flops, arith_ref) if passed[0].flops else None
            arith_knl_tf = compute_tflops(passed[0].flops, arith_kernel) if passed[0].flops else None
            line += f" {_fmt_tflops(arith_ref_tf):>11s} {_fmt_tflops(arith_knl_tf):>11s}"
        print(line)

    failed = [r for r in results if not r.correct]
    if failed:
        print(f"\n  {len(failed)}/{len(results)} benchmarks FAILED (excluded from aggregates)")

    print(f"\n{sep}\n")
