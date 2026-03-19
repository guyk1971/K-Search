"""CudeepyTask — Task implementation for cudeepy problem definitions.

Loads a cudeepy problem file (nn.Module with forward() reference + __main__
block showing input shapes) and exposes it through K-Search's Task protocol,
enabling the LLM optimization loop to generate and optimize kernels for it.

Usage:
    task = CudeepyTask(problem_path="/path/to/0_0_gemm4096x4096_fp16_acc32.py")
    # Pass to generate_and_evaluate() as the task argument
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import tempfile
import textwrap
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

from k_search.tasks.task_base import (
    BuildSpec,
    EvalResult,
    Solution,
    SourceFile,
    SupportedLanguages,
    load_ksearch_solution_json,
    solution_from_json_dict,
)
from k_search.tasks.gpu_mode.code_utils import (
    normalize_cuda_sources,
    cuda_sources_to_submission_py,
)
from k_search.eval.benchmark import (
    BenchmarkResult,
    ProblemSpec,
    benchmark_batch_events,
    check_correctness,
    clone_inputs,
    compute_tflops,
    estimate_gemm_flops,
)


# ---------------------------------------------------------------------------
# Problem loading (shared with scripts/benchmark_cudeepy_style.py)
# ---------------------------------------------------------------------------

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
    for _name, obj in inspect.getmembers(mod, inspect.isclass):
        if (
            issubclass(obj, nn.Module)
            and obj is not nn.Module
            and obj.__module__ == mod.__name__
        ):
            return obj
    return None


def _extract_main_block_source(filepath: str) -> str | None:
    """Extract source code inside `if __name__ == '__main__':` block."""
    import ast

    source = Path(filepath).read_text()
    tree = ast.parse(source)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.If):
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
    return 0.1, 1e-5


def _extract_forward_source(cls: type) -> str:
    """Extract the source code of the forward() method."""
    try:
        return inspect.getsource(cls.forward)
    except (OSError, TypeError):
        return "(source unavailable)"


def _describe_tensor(name: str, t: torch.Tensor) -> str:
    """Human-readable one-line tensor description."""
    shape = "x".join(str(d) for d in t.shape)
    return f"  {name}: ({shape}) {t.dtype}"


# ---------------------------------------------------------------------------
# CudeepyTask
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CudeepyTaskConfig:
    problem_path: str
    seed: int = 42
    warmup: int = 10
    iters: int = 50
    compile_ref: bool = True
    atol: float | None = None
    rtol: float | None = None


class CudeepyTask:
    """Task wrapper for cudeepy problem definitions.

    Loads a cudeepy problem .py file containing an nn.Module with a forward()
    reference implementation and a __main__ block that creates concrete input
    tensors. Exposes the Task protocol so K-Search generators can optimize
    kernels for it.
    """

    def __init__(
        self,
        *,
        problem_path: str,
        seed: int = 42,
        warmup: int = 10,
        iters: int = 50,
        compile_ref: bool = True,
        atol: float | None = None,
        rtol: float | None = None,
        artifacts_dir: str | None = None,
    ) -> None:
        self._cfg = CudeepyTaskConfig(
            problem_path=str(Path(problem_path).resolve()),
            seed=seed,
            warmup=warmup,
            iters=iters,
            compile_ref=compile_ref,
            atol=atol,
            rtol=rtol,
        )
        self._ksearch_artifacts_dir = str(artifacts_dir) if artifacts_dir else None
        self._solutions: dict[str, Solution] = {}

        # Feedback hooks (populated by run_benchmark)
        self._last_round_trace_logs: str = ""
        self._last_round_passed_count: int = 0
        self._last_round_total_workloads: int = 0

        # Load the problem
        self._problem_spec = self._load_problem()

    @property
    def name(self) -> str:
        return self._problem_spec.name

    # ------------------------------------------------------------------
    # Problem loading
    # ------------------------------------------------------------------

    def _load_problem(self) -> ProblemSpec:
        """Load cudeepy problem file into a ProblemSpec."""
        problem_path = self._cfg.problem_path
        mod_name = Path(problem_path).stem

        mod = _load_module_from_file(f"cudeepy_problem_{mod_name}", problem_path)
        cls = _find_module_class(mod)
        if cls is None:
            raise ValueError(f"No nn.Module subclass found in {problem_path}")

        main_src = _extract_main_block_source(problem_path)
        if main_src is None:
            raise ValueError(f"No `if __name__ == '__main__':` block in {problem_path}")

        torch.manual_seed(self._cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self._cfg.seed)

        exec_globals = dict(mod.__dict__)
        exec_globals["__name__"] = "__main__"
        exec_locals: dict[str, Any] = {}
        exec(main_src, exec_globals, exec_locals)

        inputs: list[torch.Tensor] = []
        input_names: list[str] = []
        model_instance = None

        for vname, val in exec_locals.items():
            if isinstance(val, nn.Module):
                model_instance = val
            elif isinstance(val, torch.Tensor) and val.is_cuda:
                if vname not in ("out", "output", "result", "ref_out"):
                    inputs.append(val)
                    input_names.append(vname)

        if model_instance is None:
            for vname, val in exec_locals.items():
                if isinstance(val, cls):
                    model_instance = val
                    break
        if model_instance is None:
            raise ValueError(f"Could not find {cls.__name__} instance in __main__ block")
        if not inputs:
            raise ValueError(f"No CUDA input tensors found in __main__ block of {problem_path}")

        # Store class and source for prompt generation
        self._problem_class = cls
        self._problem_module = mod
        self._forward_source = _extract_forward_source(cls)

        model_instance.eval()

        # Build reference
        if self._cfg.compile_ref:
            try:
                ref_compiled = torch.compile(model_instance, mode="max-autotune")
                with torch.no_grad():
                    for _ in range(3):
                        ref_compiled(*inputs)
                torch.cuda.synchronize()
                # Clone output to avoid CUDAGraphs buffer reuse issues
                reference_fn = lambda *args, _r=ref_compiled: _r(*args).clone()
            except Exception:
                reference_fn = lambda *args, _m=model_instance: _m(*args)
        else:
            reference_fn = lambda *args, _m=model_instance: _m(*args)

        with torch.no_grad():
            expected_output = reference_fn(*inputs)
        torch.cuda.synchronize()
        # Clone to avoid CUDAGraphs buffer reuse issues — torch.compile with
        # max-autotune may use CUDAGraphs which reuses output buffers.
        expected_output = expected_output.clone()

        atol, rtol = _default_tolerances(inputs)
        if self._cfg.atol is not None:
            atol = self._cfg.atol
        if self._cfg.rtol is not None:
            rtol = self._cfg.rtol

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

    # ------------------------------------------------------------------
    # Task protocol: definition text
    # ------------------------------------------------------------------

    def get_definition_text(self, language: str | None = None) -> str:
        """Build LLM-facing prompt describing the cudeepy problem."""
        spec = self._problem_spec
        cls = self._problem_class
        lang = str(language or "triton").strip().lower()

        # Input descriptions
        input_lines = []
        for name, tensor in zip(spec.input_names, spec.inputs):
            input_lines.append(_describe_tensor(name, tensor))

        # Output description
        out = spec.expected_output
        out_shape = "x".join(str(d) for d in out.shape)
        out_desc = f"  output: ({out_shape}) {out.dtype}"

        # Forward signature
        fwd = inspect.signature(cls.forward)
        params = [p for p in fwd.parameters if p != "self"]

        # FLOPS info
        flops_line = ""
        if spec.flops:
            flops_line = f"\nFLOPS: {spec.flops:,} ({spec.flops / 1e12:.1f} TFLOPS at 1ms)\n"

        # Language constraint
        if lang == "cuda":
            lang_constraint = (
                "IMPORTANT: You MUST write a custom CUDA kernel (not Triton, not Python-only). "
                "Use raw CUDA C++ with __global__ kernels. "
                "All GPU computation must happen in CUDA kernels you write.\n\n"
                "CRITICAL: Do NOT call cuBLAS, cuDNN, CUTLASS, or any vendor library for the "
                "core computation. Do NOT use torch::matmul, torch::mm, at::matmul, or any "
                "ATen/PyTorch operator that delegates to cuBLAS. The entire matrix multiplication "
                "(or whatever the core operation is) must be implemented in your own __global__ "
                "kernel using raw CUDA. The point is to write a custom high-performance kernel, "
                "not to wrap an existing library call."
            )
        elif lang == "triton":
            lang_constraint = (
                "IMPORTANT: You MUST write a Triton kernel using the triton library "
                "(import triton, triton.language as tl). Do not use raw CUDA.\n\n"
                "CRITICAL: Do NOT call torch.matmul, torch.mm, or any PyTorch operator that "
                "delegates to cuBLAS for the core computation. The entire operation must be "
                "implemented in your own Triton kernel using tl.dot or equivalent Triton primitives."
            )
        else:
            lang_constraint = f"Target language: {lang}"

        definition = f"""Task: {spec.name}

Implement an optimized GPU kernel for the following operation.
Target language: {lang.upper()}

{lang_constraint}

Operation: {cls.__name__}

Reference implementation (PyTorch):
```python
{self._forward_source.strip()}
```

Inputs (all on CUDA):
{chr(10).join(input_lines)}

Output:
{out_desc}
{flops_line}
Interface requirement:
Your submission must define a function with this exact signature:
```python
def custom_kernel({', '.join(params)}) -> torch.Tensor:
    ...
```

The function receives the same positional tensor arguments as the reference forward() method
(excluding `self`). It must return a tensor matching the reference output shape and dtype.

Correctness tolerance: atol={spec.atol}, rtol={spec.rtol}

Optimize for minimum latency on the target GPU. The baseline reference uses
torch.compile(mode="max-autotune") which delegates to cuBLAS. Your custom
kernel must implement the computation from scratch (no library calls for the
core operation) and aim to match or exceed the cuBLAS baseline throughput."""

        return definition.strip()

    def get_code_format_text(self, *, language: str, target_gpu: str) -> str:
        """Return code format instructions for the world model prompts.

        For CUDA: the 3-file XML format (kernel.h / kernel.cu / main.cpp).
        For Triton: single-file Python guidelines.
        """
        lang = str(language or "").strip().lower()
        tg = str(target_gpu or "H100")
        spec = self._problem_spec

        # Build the interface hint for main.cpp
        params = spec.input_names
        params_str = ", ".join(params)

        if lang == "cuda":
            return f"""IMPORTANT: Generate code in XML format with exactly 3 files with these strict names:

<header_file name="kernel.h">
- All CUDA kernel function declarations
- Host function declarations
- Any necessary struct/type definitions
- Include guards and necessary headers
</header_file>

<cuda_file name="kernel.cu">
- All __global__ kernel implementations
- All __device__ helper functions
- CUDA-specific optimizations and memory patterns
</cuda_file>

<cpp_file name="main.cpp">
- Host function that launches kernels
- Entry point function named "run" that takes PyTorch tensors ({params_str}) and returns the output tensor
- MUST include PyTorch C++ extension bindings using PYBIND11_MODULE
- The "run" function must be exposed to Python through the binding
- Include proper tensor type conversion between PyTorch tensors and CUDA pointers
- Include all necessary PyTorch headers: #include <torch/extension.h>
- The "run" function signature: torch::Tensor run({', '.join(f'torch::Tensor {p}' for p in params)})
</cpp_file>

The generated CUDA extension will be loaded via torch.utils.cpp_extension.load_inline and
called as: output = ext.run({params_str})
A wrapper Python file (submission.py) will be auto-generated to define:
    def custom_kernel({params_str}): return _EXT.run({params_str})"""
        elif lang in ("triton", "python"):
            return f"""Output a single Python file containing:
- All necessary imports (torch, triton, triton.language as tl)
- Your optimized kernel implementation
- A function named custom_kernel({params_str}) -> torch.Tensor that serves as the entry point
- Return only the code (no explanations, no markdown formatting)"""
        return ""

    # ------------------------------------------------------------------
    # Task protocol: run_benchmark
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        *,
        solution: Solution,
        config: Any = None,
        dump_traces: bool = False,
        round_num: int | None = None,
    ) -> EvalResult:
        """Evaluate a solution using cudeepy-style batch CUDA event timing."""
        spec = self._problem_spec

        # Extract custom_kernel from solution
        try:
            kernel_fn = self._load_kernel_from_solution(solution)
        except Exception as e:
            self._update_feedback(passed=False, log=f"Failed to load kernel: {e}")
            return EvalResult(
                status="failed",
                log_excerpt=f"Load error: {type(e).__name__}: {e}",
                metrics={"score_name": "speedup", "score": None},
            )

        # Correctness check
        try:
            kernel_out = kernel_fn(*clone_inputs(spec.inputs))
            torch.cuda.synchronize()
        except Exception as e:
            tb = traceback.format_exc()
            self._update_feedback(passed=False, log=f"Runtime error:\n{tb}")
            return EvalResult(
                status="runtime_error",
                log_excerpt=f"Runtime error: {type(e).__name__}: {e}\n{tb[-2000:]}",
                metrics={"score_name": "speedup", "score": None},
            )

        passed, max_err, mean_err, err_msg = check_correctness(
            kernel_out, spec.expected_output, spec.atol, spec.rtol,
        )

        if not passed:
            self._update_feedback(
                passed=False,
                log=f"Correctness failed: {err_msg}\nmax_abs_err={max_err:.6f}, mean_abs_err={mean_err:.6f}",
            )
            return EvalResult(
                status="failed",
                log_excerpt=f"Correctness: {err_msg}",
                metrics={
                    "score_name": "speedup",
                    "score": None,
                    "max_abs_err": max_err,
                    "mean_abs_err": mean_err,
                },
            )

        del kernel_out
        torch.cuda.empty_cache()

        # Timing
        ref_ms = benchmark_batch_events(
            lambda: spec.reference_fn(*clone_inputs(spec.inputs)),
            warmup=self._cfg.warmup, iters=self._cfg.iters,
        )
        kernel_ms = benchmark_batch_events(
            lambda: kernel_fn(*clone_inputs(spec.inputs)),
            warmup=self._cfg.warmup, iters=self._cfg.iters,
        )

        speedup = ref_ms / kernel_ms if kernel_ms > 0 else 0.0
        ref_tflops = compute_tflops(spec.flops, ref_ms)
        kernel_tflops = compute_tflops(spec.flops, kernel_ms)

        # Build EvalResult
        er = EvalResult(
            status="passed",
            latency_ms=kernel_ms,
            reference_latency_ms=ref_ms,
            mean_vs_baseline_factor=speedup,
            speedup_factor=speedup,
            log_excerpt="",
            metrics={
                "score_name": "speedup",
                "score": speedup,
                "max_abs_err": max_err,
                "mean_abs_err": mean_err,
                "ref_ms": ref_ms,
                "kernel_ms": kernel_ms,
                "ref_tflops": ref_tflops,
                "kernel_tflops": kernel_tflops,
            },
        )

        self._update_feedback(passed=True, log="")

        # Print summary
        rn = str(int(round_num)) if round_num is not None else "?"
        tflops_str = f"  {kernel_tflops:.1f} TFLOPS" if kernel_tflops else ""
        print(
            f"[{self.name}] Round {rn}: PASSED | "
            f"kernel={kernel_ms:.3f}ms ref={ref_ms:.3f}ms "
            f"speedup={speedup:.3f}x{tflops_str}",
            flush=True,
        )

        return er

    # ------------------------------------------------------------------
    # Task protocol: solution construction
    # ------------------------------------------------------------------

    def make_solution_from_generated_code(
        self,
        *,
        cleaned_code: Any,
        raw_code: Any,
        round_num: int,
        model_name: str,
        target_gpu: str,
        language: str,
    ) -> Solution:
        """Construct a Solution from LLM-generated code."""
        sol_name = f"{model_name}_{self.name}_{language}_r{round_num}"
        lang = str(language or "").strip().lower()

        if lang == "cuda" and isinstance(cleaned_code, dict):
            # Multi-file CUDA: kernel.h, kernel.cu, main.cpp
            files = {str(k): str(v) for k, v in cleaned_code.items()}
            sources = [
                SourceFile(path="kernel.h", content=str(files.get("kernel.h", ""))),
                SourceFile(path="kernel.cu", content=str(files.get("kernel.cu", ""))),
                SourceFile(path="main.cpp", content=str(files.get("main.cpp", ""))),
            ]
            return Solution(
                name=sol_name,
                definition=self.name,
                author=model_name,
                spec=BuildSpec(
                    language=SupportedLanguages.CUDA,
                    target_hardware=[target_gpu],
                    entry_point="main.cpp::run",
                ),
                sources=sources,
                description=f"Optimized CUDA kernel for {self.name} (round {round_num})",
            )
        else:
            # Single-file: Triton/Python
            code_text = str(cleaned_code or "")
            if not code_text.strip():
                code_text = str(raw_code or "")

            lang_map = {
                "triton": SupportedLanguages.TRITON,
                "cuda": SupportedLanguages.CUDA,
                "python": SupportedLanguages.PYTHON,
            }
            lang_enum = lang_map.get(lang, SupportedLanguages.PYTHON)

            return Solution(
                name=sol_name,
                definition=self.name,
                author=model_name,
                spec=BuildSpec(
                    language=lang_enum,
                    target_hardware=[target_gpu],
                    entry_point="submission.py::custom_kernel",
                ),
                sources=[SourceFile(path="submission.py", content=code_text)],
                description=f"Optimized kernel for {self.name} (round {round_num})",
            )

    # ------------------------------------------------------------------
    # Task protocol: other required methods
    # ------------------------------------------------------------------

    def get_solution(self, solution_name: str) -> Solution | None:
        name = str(solution_name)
        if name in self._solutions:
            return self._solutions[name]
        try:
            d = load_ksearch_solution_json(
                solution_ref=name,
                definition_name=self.name,
                artifacts_dir=self._ksearch_artifacts_dir,
            )
            sol = solution_from_json_dict(d)
            self._solutions[sol.name] = sol
            return sol
        except (FileNotFoundError, Exception):
            return None

    def build_remote_eval_package(
        self, *, solution: Solution, config: Any = None, round_num: int | None = None,
    ) -> Any:
        raise NotImplementedError("CudeepyTask does not support remote evaluation yet")

    def code_for_world_model_from_raw(self, *, raw: Any, language: str) -> str:
        lang = str(language or "").strip().lower()
        if lang == "cuda":
            # For CUDA, extract kernel.cu content for the WM (less noise than full XML)
            try:
                files = normalize_cuda_sources(raw)
                return str(files.get("kernel.cu", "") or "")
            except Exception:
                return str(raw or "")
        return str(raw or "")

    def seed_eval_for_base_solution(
        self, *, base_solution: Solution, config: Any = None,
    ) -> EvalResult:
        return self.run_benchmark(solution=base_solution, config=config, round_num=None)

    def get_config_for_logging(self) -> Dict[str, Any]:
        spec = self._problem_spec
        return {
            "task_type": "cudeepy",
            "task_name": self.name,
            "problem_path": self._cfg.problem_path,
            "input_shapes": [
                {"name": n, "shape": list(t.shape), "dtype": str(t.dtype)}
                for n, t in zip(spec.input_names, spec.inputs)
            ],
            "flops": spec.flops,
            "atol": spec.atol,
            "rtol": spec.rtol,
            "warmup": self._cfg.warmup,
            "iters": self._cfg.iters,
        }

    def run_final_evaluation(
        self,
        *,
        solutions: list[Solution],
        config: Any = None,
        dump_traces: bool = False,
        workload_limit: int | None = None,
    ) -> dict[str, Any]:
        """Final evaluation of best solutions (called at end of optimization loop)."""
        out: list[dict[str, Any]] = []
        for sol in solutions or []:
            if sol is None:
                continue
            er = self.run_benchmark(solution=sol, config=config, round_num=None)
            out.append({
                "solution": str(getattr(sol, "name", "") or ""),
                "status": str(er.status or ""),
                "latency_ms": er.latency_ms,
                "speedup": er.speedup_factor,
                "score_name": (er.metrics.get("score_name") if isinstance(er.metrics, dict) else None),
                "score": (er.metrics.get("score") if isinstance(er.metrics, dict) else None),
            })
        return {
            "task": self.name,
            "solutions": out,
        }

    # Optional feedback hooks (generators use getattr)
    def get_last_round_trace_logs_for_prompt(self) -> str:
        return self._last_round_trace_logs

    def get_last_round_passed_count(self) -> int:
        return self._last_round_passed_count

    def get_last_round_total_workloads(self) -> int:
        return self._last_round_total_workloads

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_kernel_from_solution(self, solution: Solution) -> Callable:
        """Extract custom_kernel callable from a Solution's sources.

        For CUDA solutions (kernel.h + kernel.cu + main.cpp): converts to a
        submission.py using cuda_sources_to_submission_py(), which wraps the
        CUDA code in a torch.utils.cpp_extension.load_inline call exposing
        a `run()` function. We then wrap `run()` as `custom_kernel`.

        For Python/Triton solutions: loads the entry source directly.
        """
        _lang_raw = getattr(solution.spec, "language", "") or ""
        lang = str(_lang_raw.value if hasattr(_lang_raw, "value") else _lang_raw).strip().lower()

        tmpdir = Path(tempfile.mkdtemp(prefix="ksearch_cudeepy_eval_"))

        if lang == "cuda":
            # Multi-file CUDA solution → build submission.py with load_inline wrapper
            sources_dict = {sf.path: sf.content for sf in (solution.sources or [])}
            normalized = normalize_cuda_sources(sources_dict)
            submission_content = cuda_sources_to_submission_py(normalized)

            # The generated submission.py defines custom_kernel(data) with the
            # GPUMode tuple convention. We need to adapt it for cudeepy's
            # positional-args convention.
            # Actually, cuda_sources_to_submission_py wraps it as:
            #   def custom_kernel(data): return _EXT.run(...)
            # But for cudeepy we need: custom_kernel(a, b) → _EXT.run(a, b)
            #
            # Instead of using the GPUMode wrapper, we build our own thin wrapper.
            import json as _json
            params = self._problem_spec.input_names
            params_str = ", ".join(params)
            sources_json = _json.dumps(normalized)

            wrapper = (
                "import json, os, hashlib\n"
                "from pathlib import Path\n"
                "import torch\n"
                "from torch.utils.cpp_extension import load\n"
                "\n"
                f"_SOURCES = json.loads(r'''{sources_json}''')\n"
                "_EXT = None\n"
                "\n"
                "def _build_ext():\n"
                '    build_root = Path(os.environ.get("FIB_CACHE_PATH", str(Path.home() / ".cache" / "k_search" / "cache")))\n'
                '    digest = hashlib.md5(json.dumps(_SOURCES, sort_keys=True).encode()).hexdigest()[:12]\n'
                '    build_dir = build_root / "cudeepy_cuda" / digest\n'
                "    build_dir.mkdir(parents=True, exist_ok=True)\n"
                "    for name, content in _SOURCES.items():\n"
                "        (build_dir / name).write_text(content)\n"
                '    sources = [str(build_dir / "main.cpp"), str(build_dir / "kernel.cu")]\n'
                "    ext = load(\n"
                '        name=f"cudeepy_cuda_{digest}",\n'
                "        sources=sources,\n"
                "        extra_include_paths=[str(build_dir)],\n"
                "        with_cuda=True,\n"
                "        build_directory=str(build_dir),\n"
                "        verbose=False,\n"
                "    )\n"
                "    return ext\n"
                "\n"
                f"def custom_kernel({params_str}):\n"
                "    global _EXT\n"
                "    if _EXT is None:\n"
                "        _EXT = _build_ext()\n"
                f"    return _EXT.run({params_str})\n"
            )
            submission_file = tmpdir / "submission.py"
            submission_file.write_text(wrapper)
        else:
            # Python/Triton: use entry source directly
            entry_src = solution.get_entry_source()
            if entry_src is None:
                raise ValueError(f"No entry source in solution {solution.name}")
            submission_file = tmpdir / "submission.py"
            submission_file.write_text(entry_src.content)

        modspec = importlib.util.spec_from_file_location(
            "cudeepy_submission", str(submission_file),
        )
        if modspec is None or modspec.loader is None:
            raise ImportError(f"Cannot load submission from {submission_file}")
        mod = importlib.util.module_from_spec(modspec)
        modspec.loader.exec_module(mod)

        fn = getattr(mod, "custom_kernel", None)
        if fn is None:
            raise ValueError(
                f"No `custom_kernel` function in submission for solution {solution.name}"
            )
        return fn

    def _update_feedback(self, *, passed: bool, log: str) -> None:
        """Update feedback hooks for the generator loop."""
        self._last_round_trace_logs = log
        self._last_round_total_workloads = 1
        self._last_round_passed_count = 1 if passed else 0
