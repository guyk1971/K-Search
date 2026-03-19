#!/usr/bin/env python3
"""Benchmark K-Search kernels using cudeepy-style evaluation methodology.

Supports two modes:
  1. cudeepy problems — load an nn.Module problem file directly
  2. K-Search TriMul — backward-compatible mode for GPUMode tasks

Examples:
    # Benchmark a kernel on a cudeepy GEMM problem:
    python3 scripts/benchmark_cudeepy_style.py \\
        --problem /path/to/cudeepy/problems/set_0/0_0_gemm4096x4096_fp16_acc32.py \\
        --kernel path/to/submission.py

    # Benchmark on all problems in a directory:
    python3 scripts/benchmark_cudeepy_style.py \\
        --problem-dir /path/to/cudeepy/problems/set_0/ \\
        --kernel path/to/submission.py

    # Benchmark from a K-Search solution JSON:
    python3 scripts/benchmark_cudeepy_style.py \\
        --problem /path/to/problem.py \\
        --solution-json path/to/solution.json

    # TriMul backward-compatible mode:
    python3 scripts/benchmark_cudeepy_style.py \\
        --task trimul --kernel path/to/submission.py

    # Custom options:
    python3 scripts/benchmark_cudeepy_style.py \\
        --problem /path/to/problem.py --kernel sub.py \\
        --warmup 20 --iters 100 --atol 0.5 --flops 274877906944
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import json
import os
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

# Ensure project root is on sys.path for k_search imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from k_search.eval.benchmark import (
    BenchmarkResult,
    ProblemSpec,
    benchmark_batch_events,
    benchmark_per_iter_events,
    check_correctness,
    clone_inputs,
    compute_tflops,
    estimate_gemm_flops,
    print_report,
    run_benchmark,
)

TRIMUL_DIR = PROJECT_ROOT / "k_search" / "tasks" / "gpu_mode" / "trimul"


# ═══════════════════════════════════════════════════════════════════════════
# cudeepy problem loading
# ═══════════════════════════════════════════════════════════════════════════

def _load_module_from_file(name: str, path: str) -> Any:
    """Import a Python module from an arbitrary file path."""
    path = str(Path(path).resolve())
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _find_module_class(mod: Any) -> type | None:
    """Find the first nn.Module subclass defined in a module."""
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if issubclass(obj, nn.Module) and obj is not nn.Module and obj.__module__ == mod.__name__:
            return obj
    return None


def _extract_main_block_source(filepath: str) -> str | None:
    """Extract the source code inside `if __name__ == "__main__":` block."""
    source = Path(filepath).read_text()
    tree = ast.parse(source)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.If):
            # Match: if __name__ == "__main__"
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "__main__"
            ):
                lines = source.splitlines()
                # Get the body lines (from first body statement to the end of the if block)
                start = node.body[0].lineno - 1
                end = node.body[-1].end_lineno
                block = "\n".join(lines[start:end])
                return textwrap.dedent(block)
    return None


def _default_tolerances(inputs: list[torch.Tensor]) -> tuple[float, float]:
    """Auto-detect atol/rtol from input dtype."""
    if not inputs:
        return 0.1, 1e-5
    dtype = inputs[0].dtype
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return 0.5, 1e-2
    if dtype == torch.bfloat16:
        return 0.1, 1e-2
    if dtype == torch.float32:
        return 1e-4, 1e-5
    # float16 and default
    return 0.1, 1e-5


def load_cudeepy_problem(
    problem_path: str,
    *,
    compile_ref: bool = True,
    seed: int = 42,
    atol_override: float | None = None,
    rtol_override: float | None = None,
    flops_override: int | None = None,
) -> ProblemSpec:
    """Load a cudeepy problem file into a ProblemSpec.

    1. Import the module and find the nn.Module subclass
    2. Execute the __main__ block to capture inputs + model instance
    3. Optionally torch.compile the reference
    4. Auto-detect FLOPS and tolerances
    """
    problem_path = str(Path(problem_path).resolve())
    mod_name = Path(problem_path).stem

    # Import the module
    mod = _load_module_from_file(f"cudeepy_problem_{mod_name}", problem_path)
    cls = _find_module_class(mod)
    if cls is None:
        raise ValueError(f"No nn.Module subclass found in {problem_path}")

    # Execute __main__ block to capture inputs and model
    main_src = _extract_main_block_source(problem_path)
    if main_src is None:
        raise ValueError(f"No `if __name__ == '__main__':` block found in {problem_path}")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Execute in a namespace that has access to the module's globals
    exec_globals = dict(mod.__dict__)
    exec_globals["__name__"] = "__main__"  # so any __name__ checks pass
    exec_locals: dict[str, Any] = {}
    exec(main_src, exec_globals, exec_locals)

    # Collect tensors and model from exec_locals
    inputs: list[torch.Tensor] = []
    input_names: list[str] = []
    model_instance = None

    for vname, val in exec_locals.items():
        if isinstance(val, nn.Module):
            model_instance = val
        elif isinstance(val, torch.Tensor) and val.is_cuda:
            # Skip output tensors (heuristic: 'out' in name or assigned after model call)
            if vname not in ("out", "output", "result", "ref_out"):
                inputs.append(val)
                input_names.append(vname)

    if model_instance is None:
        # Try to find it in exec_globals (in case it was assigned there)
        for vname, val in exec_locals.items():
            if isinstance(val, cls):
                model_instance = val
                break
    if model_instance is None:
        raise ValueError(f"Could not find {cls.__name__} instance in __main__ block")

    if not inputs:
        raise ValueError(f"No CUDA input tensors found in __main__ block of {problem_path}")

    # Build reference function
    model_instance.eval()
    if compile_ref:
        try:
            ref_compiled = torch.compile(model_instance, mode="max-autotune")
            # Warmup torch.compile
            with torch.no_grad():
                for _ in range(3):
                    ref_compiled(*inputs)
            torch.cuda.synchronize()
            reference_fn = lambda *args: ref_compiled(*args)
        except Exception:
            # Fallback to eager if torch.compile fails
            reference_fn = lambda *args: model_instance(*args)
    else:
        reference_fn = lambda *args: model_instance(*args)

    # Compute expected output
    with torch.no_grad():
        expected_output = reference_fn(*inputs)
    torch.cuda.synchronize()

    # Tolerances
    atol, rtol = _default_tolerances(inputs)
    if atol_override is not None:
        atol = atol_override
    if rtol_override is not None:
        rtol = rtol_override

    # FLOPS
    flops = flops_override
    if flops is None:
        flops = estimate_gemm_flops(inputs)

    return ProblemSpec(
        name=mod_name,
        reference_fn=reference_fn,
        inputs=inputs,
        input_names=input_names,
        expected_output=expected_output,
        atol=atol,
        rtol=rtol,
        flops=flops,
        model_instance=model_instance,
    )


def load_cudeepy_problems_from_dir(
    problem_dir: str,
    **kwargs,
) -> list[ProblemSpec]:
    """Load all .py problem files from a directory."""
    d = Path(problem_dir).resolve()
    specs = []
    for f in sorted(d.glob("*.py")):
        if f.name.startswith("__"):
            continue
        try:
            spec = load_cudeepy_problem(str(f), **kwargs)
            specs.append(spec)
        except Exception as e:
            print(f"  [SKIP] {f.name}: {e}")
    return specs


# ═══════════════════════════════════════════════════════════════════════════
# K-Search kernel loading
# ═══════════════════════════════════════════════════════════════════════════

def load_kernel_from_py(kernel_path: str) -> Callable:
    """Load custom_kernel from a Python submission file.

    The kernel should define: def custom_kernel(*args) -> Tensor
    """
    mod = _load_module_from_file("ksearch_kernel", str(Path(kernel_path).resolve()))
    fn = getattr(mod, "custom_kernel", None)
    if fn is None:
        raise ValueError(f"No `custom_kernel` function found in {kernel_path}")
    return fn


def load_kernel_from_json(json_path: str) -> Callable:
    """Load custom_kernel from a K-Search solution JSON."""
    with open(json_path) as f:
        sol = json.load(f)

    sources = sol.get("sources", [])
    spec = sol.get("spec", {})
    language = str(spec.get("language", "python")).lower()

    if language == "cuda":
        from k_search.tasks.gpu_mode.code_utils import (
            cuda_sources_to_submission_py,
            normalize_cuda_sources,
        )
        files = {s["path"]: s["content"] for s in sources}
        normalized = normalize_cuda_sources(files)
        submission_content = cuda_sources_to_submission_py(normalized)
    else:
        entry_path = str(spec.get("entry_point", "submission.py")).split("::")[0]
        submission_content = None
        for s in sources:
            if s["path"] == entry_path:
                submission_content = s["content"]
                break
        if submission_content is None and sources:
            submission_content = sources[0]["content"]
        if not submission_content:
            raise ValueError(f"No source content found in {json_path}")

    tmpdir = Path(tempfile.mkdtemp(prefix="ksearch_kernel_"))
    submission_file = tmpdir / "submission.py"
    submission_file.write_text(submission_content)
    return load_kernel_from_py(str(submission_file))


# ═══════════════════════════════════════════════════════════════════════════
# TriMul backward compatibility
# ═══════════════════════════════════════════════════════════════════════════

TRIMUL_BENCHMARK_SPECS = [
    {"seqlen": 256,  "bs": 2, "dim": 128, "hiddendim": 128, "seed": 9371,  "nomask": True,  "distribution": "normal"},
    {"seqlen": 768,  "bs": 1, "dim": 128, "hiddendim": 128, "seed": 381,   "nomask": True,  "distribution": "cauchy"},
    {"seqlen": 256,  "bs": 2, "dim": 384, "hiddendim": 128, "seed": 2301,  "nomask": False, "distribution": "normal"},
    {"seqlen": 512,  "bs": 1, "dim": 128, "hiddendim": 128, "seed": 12819, "nomask": True,  "distribution": "normal"},
    {"seqlen": 1024, "bs": 1, "dim": 128, "hiddendim": 128, "seed": 381,   "nomask": True,  "distribution": "cauchy"},
    {"seqlen": 768,  "bs": 1, "dim": 384, "hiddendim": 128, "seed": 481,   "nomask": False, "distribution": "normal"},
    {"seqlen": 1024, "bs": 1, "dim": 384, "hiddendim": 128, "seed": 23291, "nomask": True,  "distribution": "normal"},
]


def _setup_trimul_workdir() -> Path:
    """Create a temp directory with the trimul support files on sys.path."""
    tmpdir = Path(tempfile.mkdtemp(prefix="ksearch_bench_trimul_"))
    for fname in ("reference.py", "utils.py", "task.py"):
        src = TRIMUL_DIR / fname
        if src.exists():
            shutil.copy2(src, tmpdir / fname)
    sys.path.insert(0, str(tmpdir))
    return tmpdir


def _clone_trimul_data(data):
    """Deep-clone TriMul data tuple."""
    if isinstance(data, torch.Tensor):
        return data.clone()
    if isinstance(data, tuple):
        return tuple(_clone_trimul_data(x) for x in data)
    if isinstance(data, list):
        return [_clone_trimul_data(x) for x in data]
    if isinstance(data, dict):
        return {k: _clone_trimul_data(v) for k, v in data.items()}
    return data


def build_trimul_specs(
    kernel_fn: Callable,
    *,
    atol_override: float | None = None,
    rtol_override: float | None = None,
) -> list[ProblemSpec]:
    """Build ProblemSpec list for TriMul benchmark specs."""
    from reference import generate_input, ref_kernel

    atol = atol_override if atol_override is not None else 2e-2
    rtol = rtol_override if rtol_override is not None else 2e-2

    specs = []
    for bm in TRIMUL_BENCHMARK_SPECS:
        data = generate_input(**bm)
        tag = f"trimul_seq{bm['seqlen']}_bs{bm['bs']}_dim{bm['dim']}_{bm['distribution']}"

        # TriMul reference takes a single data tuple
        with torch.no_grad():
            expected = ref_kernel(_clone_trimul_data(data))
        torch.cuda.synchronize()

        # Wrap reference and kernel to match ProblemSpec's *args calling convention.
        # We store the data tuple as a single-element inputs list and wrap the callables.
        ref_data = _clone_trimul_data(data)
        ref_fn = lambda *_args, _d=ref_data: ref_kernel(_clone_trimul_data(_d))
        kern_fn_wrapped = lambda *_args, _d=ref_data, _k=kernel_fn: _k(_clone_trimul_data(_d))

        specs.append(ProblemSpec(
            name=tag,
            reference_fn=ref_fn,
            inputs=[],  # empty — timing lambdas capture data internally
            input_names=[],
            expected_output=expected,
            atol=atol,
            rtol=rtol,
            flops=None,
            model_instance=None,
        ))
        # Stash the wrapped kernel on the spec for the runner to use
        specs[-1]._kernel_fn = kern_fn_wrapped  # type: ignore[attr-defined]

    return specs


# ═══════════════════════════════════════════════════════════════════════════
# Main CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark K-Search kernels with cudeepy-style timing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Problem source (mutually exclusive)
    prob_group = parser.add_mutually_exclusive_group()
    prob_group.add_argument(
        "--problem", type=str, nargs="+",
        help="Path(s) to cudeepy problem .py file(s)",
    )
    prob_group.add_argument(
        "--problem-dir", type=str,
        help="Directory of cudeepy problem .py files (run all)",
    )
    prob_group.add_argument(
        "--task", type=str, choices=["trimul"],
        help="Built-in K-Search task (backward compatibility)",
    )

    # Kernel source (mutually exclusive)
    kern_group = parser.add_mutually_exclusive_group()
    kern_group.add_argument("--kernel", type=str, help="Path to kernel submission.py")
    kern_group.add_argument("--solution-json", type=str, help="Path to K-Search solution JSON")

    # Options
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations (default: 50)")
    parser.add_argument("--atol", type=float, default=None, help="Override absolute tolerance")
    parser.add_argument("--rtol", type=float, default=None, help="Override relative tolerance")
    parser.add_argument("--flops", type=int, default=None, help="Override FLOP count for TFLOPS")
    parser.add_argument("--no-compile-ref", action="store_true", help="Skip torch.compile on reference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--per-iter", action="store_true", help="Also show per-iteration timing stats")

    args = parser.parse_args()

    # Validate: need either a problem or a task
    if not args.problem and not args.problem_dir and not args.task:
        parser.error("Provide --problem, --problem-dir, or --task")

    if args.task == "trimul":
        _run_trimul(args)
    else:
        _run_cudeepy(args)


def _run_cudeepy(args):
    """Run benchmark on cudeepy problem(s)."""
    # Load problem specs
    common_kwargs = dict(
        compile_ref=not args.no_compile_ref,
        seed=args.seed,
        atol_override=args.atol,
        rtol_override=args.rtol,
        flops_override=args.flops,
    )

    if args.problem_dir:
        print(f"Loading problems from {args.problem_dir}...")
        problem_specs = load_cudeepy_problems_from_dir(args.problem_dir, **common_kwargs)
    else:
        problem_specs = []
        for p in args.problem:
            print(f"Loading problem: {p}")
            problem_specs.append(load_cudeepy_problem(p, **common_kwargs))

    if not problem_specs:
        print("No problems loaded.")
        sys.exit(1)

    # Load kernel
    if args.solution_json:
        kernel_fn = load_kernel_from_json(args.solution_json)
        kernel_label = Path(args.solution_json).stem
    elif args.kernel:
        kernel_fn = load_kernel_from_py(args.kernel)
        kernel_label = Path(args.kernel).stem
    else:
        # No kernel specified — benchmark reference against itself (sanity check)
        kernel_fn = None
        kernel_label = "reference (self-check)"

    print(f"Kernel: {kernel_label}")
    print(f"Timing: warmup={args.warmup}, iters={args.iters}")
    print(f"Problems: {len(problem_specs)}")

    # Run benchmarks
    results: list[BenchmarkResult] = []
    for spec in problem_specs:
        fn = kernel_fn if kernel_fn is not None else spec.reference_fn
        r = run_benchmark(spec, fn, warmup=args.warmup, iters=args.iters)
        results.append(r)
        status = "PASS" if r.correct else "FAIL"
        tflops_str = f"  {r.kernel_tflops:.1f} TFLOPS" if r.kernel_tflops else ""
        print(f"  [{status}] {r.name} — {r.kernel_ms:.3f}ms ({r.speedup:.3f}x){tflops_str}")

    print_report(results, kernel_label=kernel_label, warmup=args.warmup, iters=args.iters)

    # Optional per-iteration comparison
    if args.per_iter:
        print("Per-iteration timing (K-Search style):")
        for spec, r in zip(problem_specs, results):
            if not r.correct:
                continue
            fn = kernel_fn if kernel_fn is not None else spec.reference_fn
            stats = benchmark_per_iter_events(
                lambda _fn=fn, _s=spec: _fn(*clone_inputs(_s.inputs)),
                warmup=args.warmup, iters=args.iters,
            )
            print(
                f"  {r.name}  mean={stats['mean']:.3f}ms  std={stats['std']:.3f}ms  "
                f"best={stats['best']:.3f}ms  worst={stats['worst']:.3f}ms  "
                f"(batch={r.kernel_ms:.3f}ms, delta={stats['mean']-r.kernel_ms:+.3f}ms)"
            )


def _run_trimul(args):
    """Run benchmark in TriMul backward-compatible mode."""
    workdir = _setup_trimul_workdir()
    try:
        # Load kernel
        if args.solution_json:
            # Need to copy to workdir for trimul imports
            kernel_fn = load_kernel_from_json(args.solution_json)
            kernel_label = Path(args.solution_json).stem
        elif args.kernel:
            kernel_fn = load_kernel_from_py(args.kernel)
            kernel_label = Path(args.kernel).stem
        else:
            # Default: baseline
            kernel_fn = load_kernel_from_py(str(TRIMUL_DIR / "submission.py"))
            kernel_label = "trimul baseline"

        print(f"Kernel: {kernel_label}")
        print(f"Mode: TriMul (7 benchmark specs)")
        print(f"Timing: warmup={args.warmup}, iters={args.iters}")

        trimul_specs = build_trimul_specs(
            kernel_fn, atol_override=args.atol, rtol_override=args.rtol,
        )

        # For TriMul, the ProblemSpec has wrapped lambdas — use the stashed kernel_fn
        results: list[BenchmarkResult] = []
        for spec in trimul_specs:
            wrapped_kernel = getattr(spec, "_kernel_fn", spec.reference_fn)

            # TriMul specs have empty inputs (data captured in lambdas).
            # Run correctness manually, then time.
            try:
                kernel_out = wrapped_kernel()
                torch.cuda.synchronize()
            except Exception as e:
                results.append(BenchmarkResult(
                    name=spec.name, correct=False,
                    error_msg=f"{type(e).__name__}: {e}",
                ))
                print(f"  [FAIL] {spec.name}: {e}")
                continue

            passed, max_err, mean_err, msg = check_correctness(
                kernel_out, spec.expected_output, spec.atol, spec.rtol,
            )
            if not passed:
                results.append(BenchmarkResult(
                    name=spec.name, correct=False,
                    max_abs_err=max_err, mean_abs_err=mean_err, error_msg=msg,
                ))
                print(f"  [FAIL] {spec.name}: {msg[:60]}")
                continue

            del kernel_out
            torch.cuda.empty_cache()

            ref_ms = benchmark_batch_events(
                spec.reference_fn, warmup=args.warmup, iters=args.iters,
            )
            kernel_ms = benchmark_batch_events(
                wrapped_kernel, warmup=args.warmup, iters=args.iters,
            )
            speedup = ref_ms / kernel_ms if kernel_ms > 0 else 0
            r = BenchmarkResult(
                name=spec.name, correct=True,
                max_abs_err=max_err, mean_abs_err=mean_err,
                ref_ms=ref_ms, kernel_ms=kernel_ms, speedup=speedup,
            )
            results.append(r)
            print(f"  [PASS] {spec.name} — {kernel_ms:.3f}ms ({speedup:.3f}x)")

        print_report(results, kernel_label=kernel_label, warmup=args.warmup, iters=args.iters)
    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
