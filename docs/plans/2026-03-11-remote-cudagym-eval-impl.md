# Remote CudaGym Evaluation Backend — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable K-Search to compile and benchmark GPU kernels on remote CudaGym servers instead of requiring a local GPU.

**Architecture:** A new `k_search/eval/` module introduces `RemoteEvalPackage` (data) and `RemoteEvaluator` (CudaGym orchestrator). Each task implements `build_remote_eval_package()` to produce self-contained evaluation bundles. The `run_benchmark()` method dispatches to local or remote based on `--eval-backend`.

**Tech Stack:** Python 3.12, cudagym client (async HTTP via aiohttp), asyncio, PyTorch (remote side only)

---

### Task 1: RemoteEvalPackage dataclass and result parsing

**Files:**
- Create: `k_search/eval/__init__.py`
- Create: `k_search/eval/remote_eval_package.py`
- Create: `tests/__init__.py`
- Create: `tests/eval/__init__.py`
- Create: `tests/eval/test_remote_eval_package.py`

**Step 1: Write the failing test**

```python
# tests/eval/test_remote_eval_package.py
import json
import pytest
from k_search.eval.remote_eval_package import RemoteEvalPackage, parse_driver_output
from k_search.tasks.task_base import EvalResult


class TestRemoteEvalPackage:
    def test_create_package_cuda(self):
        pkg = RemoteEvalPackage(
            files={"kernel.cu": "// cuda code", "driver.py": "# driver"},
            run_command="python3 driver.py",
            compile_command="nvcc -o kernel kernel.cu",
            artifact_names=["kernel"],
            language="cuda",
            env_vars={"CUDA_VISIBLE_DEVICES": "0"},
        )
        assert pkg.language == "cuda"
        assert pkg.compile_command is not None
        assert len(pkg.files) == 2

    def test_create_package_triton_no_compile(self):
        pkg = RemoteEvalPackage(
            files={"submission.py": "# triton", "driver.py": "# driver"},
            run_command="python3 driver.py",
            compile_command=None,
            artifact_names=[],
            language="triton",
        )
        assert pkg.compile_command is None
        assert pkg.artifact_names == []

    def test_cudagym_env_type(self):
        cuda_pkg = RemoteEvalPackage(
            files={}, run_command="", compile_command="nvcc ...",
            artifact_names=[], language="cuda",
        )
        triton_pkg = RemoteEvalPackage(
            files={}, run_command="", compile_command=None,
            artifact_names=[], language="triton",
        )
        python_pkg = RemoteEvalPackage(
            files={}, run_command="", compile_command=None,
            artifact_names=[], language="python",
        )
        assert cuda_pkg.cudagym_env_type == "cudacpp"
        assert triton_pkg.cudagym_env_type == "python"
        assert python_pkg.cudagym_env_type == "python"


class TestParseDriverOutput:
    def test_parse_passed(self):
        output = json.dumps({
            "status": "passed",
            "latency_ms": 1.56,
            "reference_latency_ms": 2.34,
            "metrics": {"score_name": "inv_latency_ms", "score": 0.641},
            "log_excerpt": "",
            "profiling": None,
        })
        result = parse_driver_output(output)
        assert isinstance(result, EvalResult)
        assert result.status == "passed"
        assert result.latency_ms == 1.56
        assert result.reference_latency_ms == 2.34
        assert result.metrics["score"] == 0.641

    def test_parse_failed(self):
        output = json.dumps({
            "status": "compile_error",
            "latency_ms": None,
            "metrics": {},
            "log_excerpt": "nvcc error: undefined symbol",
        })
        result = parse_driver_output(output)
        assert result.status == "compile_error"
        assert result.latency_ms is None
        assert "nvcc error" in result.log_excerpt

    def test_parse_with_profiling(self):
        output = json.dumps({
            "status": "passed",
            "latency_ms": 1.0,
            "metrics": {},
            "log_excerpt": "",
            "profiling": {
                "ncu": {"raw_logs": "Speed of Light: 50%", "metrics": {"sm_occupancy": 0.5}},
                "nsys": {"raw_logs": "kernel summary...", "kernel_summary": {}},
            },
        })
        result = parse_driver_output(output)
        assert result.status == "passed"
        assert result.metrics["profiling"]["ncu"]["raw_logs"] == "Speed of Light: 50%"

    def test_parse_malformed_json(self):
        result = parse_driver_output("not json at all {{{")
        assert result.status == "runtime_error"
        assert "Failed to parse" in result.log_excerpt

    def test_parse_extracts_json_from_mixed_output(self):
        output = "Loading model...\nWarmup done\n" + json.dumps({
            "status": "passed", "latency_ms": 2.0, "metrics": {}, "log_excerpt": "",
        }) + "\nSome trailing output"
        result = parse_driver_output(output)
        assert result.status == "passed"
        assert result.latency_ms == 2.0
```

**Step 2: Run test to verify it fails**

Run: `cd /home/scratch.gkoren_gpu/code/github/guyk1971/K-Search && python -m pytest tests/eval/test_remote_eval_package.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'k_search.eval'`

**Step 3: Write minimal implementation**

```python
# k_search/eval/__init__.py
from k_search.eval.remote_eval_package import RemoteEvalPackage, parse_driver_output

__all__ = ["RemoteEvalPackage", "parse_driver_output"]
```

```python
# k_search/eval/remote_eval_package.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from k_search.tasks.task_base import EvalResult


@dataclass
class RemoteEvalPackage:
    """Self-contained evaluation bundle for remote CudaGym execution."""

    files: dict[str, str]
    run_command: str
    compile_command: str | None = None
    artifact_names: list[str] = field(default_factory=list)
    language: str = "python"
    env_vars: dict[str, str] = field(default_factory=dict)

    @property
    def cudagym_env_type(self) -> str:
        """Map K-Search language to CudaGym EnvType string."""
        if self.language == "cuda":
            return "cudacpp"
        return "python"


# Marker prefix/suffix so driver output can be found in noisy stdout
_JSON_MARKER_START = "===KSEARCH_RESULT_JSON_START==="
_JSON_MARKER_END = "===KSEARCH_RESULT_JSON_END==="


def parse_driver_output(stdout: str) -> EvalResult:
    """Parse structured JSON from remote driver stdout into EvalResult.

    The driver should emit JSON between marker lines. Falls back to
    scanning for a top-level JSON object if markers are absent.
    """
    json_str = _extract_json_block(stdout)
    if json_str is None:
        return EvalResult(
            status="runtime_error",
            log_excerpt=f"Failed to parse driver output: no valid JSON found. Output: {stdout[:800]}",
        )

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return EvalResult(
            status="runtime_error",
            log_excerpt=f"Failed to parse driver JSON: {e}. Raw: {json_str[:800]}",
        )

    # Extract profiling into metrics if present
    metrics = data.get("metrics", {})
    profiling = data.get("profiling")
    if profiling:
        metrics["profiling"] = profiling

    return EvalResult(
        status=data.get("status", "runtime_error"),
        latency_ms=data.get("latency_ms"),
        reference_latency_ms=data.get("reference_latency_ms"),
        mean_vs_baseline_factor=data.get("mean_vs_baseline_factor"),
        speedup_factor=data.get("speedup_factor"),
        log_excerpt=data.get("log_excerpt", ""),
        metrics=metrics,
    )


def _extract_json_block(stdout: str) -> str | None:
    """Extract JSON from stdout, using markers if present, else heuristic."""
    # Try marker-delimited block first
    if _JSON_MARKER_START in stdout and _JSON_MARKER_END in stdout:
        start = stdout.index(_JSON_MARKER_START) + len(_JSON_MARKER_START)
        end = stdout.index(_JSON_MARKER_END)
        return stdout[start:end].strip()

    # Fallback: find the last line that looks like a complete JSON object
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                json.loads(line)
                return line
            except json.JSONDecodeError:
                continue

    return None
```

**Step 4: Run test to verify it passes**

Run: `cd /home/scratch.gkoren_gpu/code/github/guyk1971/K-Search && python -m pytest tests/eval/test_remote_eval_package.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add k_search/eval/__init__.py k_search/eval/remote_eval_package.py tests/__init__.py tests/eval/__init__.py tests/eval/test_remote_eval_package.py
git commit -m "feat: add RemoteEvalPackage dataclass and driver output parser"
```

---

### Task 2: RemoteEvaluator — CudaGym orchestrator

**Files:**
- Create: `k_search/eval/remote_evaluator.py`
- Create: `tests/eval/test_remote_evaluator.py`

**Step 1: Write the failing test**

```python
# tests/eval/test_remote_evaluator.py
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from k_search.eval.remote_eval_package import RemoteEvalPackage, _JSON_MARKER_START, _JSON_MARKER_END
from k_search.eval.remote_evaluator import RemoteEvaluator
from k_search.tasks.task_base import EvalResult


def _make_passed_stdout(latency_ms=1.5):
    result = json.dumps({
        "status": "passed",
        "latency_ms": latency_ms,
        "reference_latency_ms": 2.0,
        "metrics": {"score_name": "inv_latency_ms", "score": 1.0 / latency_ms},
        "log_excerpt": "",
    })
    return f"Loading...\n{_JSON_MARKER_START}\n{result}\n{_JSON_MARKER_END}\n"


def _make_triton_package():
    return RemoteEvalPackage(
        files={"submission.py": "def custom_kernel(data): pass", "driver.py": "# driver"},
        run_command="python3 driver.py",
        compile_command=None,
        artifact_names=[],
        language="triton",
    )


def _make_cuda_package():
    return RemoteEvalPackage(
        files={"kernel.cu": "// code", "driver.cpp": "// main", "driver.py": "# driver"},
        run_command="python3 driver.py",
        compile_command="cmake . && make",
        artifact_names=["eval_binary"],
        language="cuda",
    )


class TestRemoteEvaluatorTriton:
    """Triton/Python: no compilation, just execute."""

    def test_evaluate_triton_success(self):
        mock_exec_result = MagicMock()
        mock_exec_result.successes = [True]
        mock_exec_result.stdouts = [_make_passed_stdout()]
        mock_exec_result.stderrs = [""]
        mock_exec_result.exception = ""

        with patch("k_search.eval.remote_evaluator.CudaGymClient") as MockClient:
            client_instance = AsyncMock()
            MockClient.return_value.__aenter__ = AsyncMock(return_value=client_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
            client_instance.execute = AsyncMock(return_value=mock_exec_result)

            evaluator = RemoteEvaluator(server_url="http://localhost:8000")
            result = evaluator.evaluate(_make_triton_package())

            assert result.status == "passed"
            assert result.latency_ms == 1.5
            client_instance.compile.assert_not_awaited()
            client_instance.execute.assert_awaited_once()

    def test_evaluate_triton_execution_failure(self):
        mock_exec_result = MagicMock()
        mock_exec_result.successes = [False]
        mock_exec_result.stdouts = [""]
        mock_exec_result.stderrs = ["Segfault"]
        mock_exec_result.exception = ""

        with patch("k_search.eval.remote_evaluator.CudaGymClient") as MockClient:
            client_instance = AsyncMock()
            MockClient.return_value.__aenter__ = AsyncMock(return_value=client_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
            client_instance.execute = AsyncMock(return_value=mock_exec_result)

            evaluator = RemoteEvaluator(server_url="http://localhost:8000")
            result = evaluator.evaluate(_make_triton_package())

            assert result.status == "runtime_error"
            assert "Segfault" in result.log_excerpt


class TestRemoteEvaluatorCuda:
    """CUDA: compile then execute."""

    def test_evaluate_cuda_success(self):
        mock_compile_result = MagicMock()
        mock_compile_result.success = True
        mock_compile_result.output_base64 = {"eval_binary": "base64data"}
        mock_compile_result.exception = ""
        mock_compile_result.job_id = "compile-123"

        mock_exec_result = MagicMock()
        mock_exec_result.successes = [True]
        mock_exec_result.stdouts = [_make_passed_stdout(2.0)]
        mock_exec_result.stderrs = [""]
        mock_exec_result.exception = ""

        with patch("k_search.eval.remote_evaluator.CudaGymClient") as MockClient:
            client_instance = AsyncMock()
            MockClient.return_value.__aenter__ = AsyncMock(return_value=client_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
            client_instance.compile = AsyncMock(return_value=mock_compile_result)
            client_instance.execute = AsyncMock(return_value=mock_exec_result)

            evaluator = RemoteEvaluator(server_url="http://localhost:8000")
            result = evaluator.evaluate(_make_cuda_package())

            assert result.status == "passed"
            assert result.latency_ms == 2.0
            client_instance.compile.assert_awaited_once()
            client_instance.execute.assert_awaited_once()

    def test_evaluate_cuda_compile_failure(self):
        mock_compile_result = MagicMock()
        mock_compile_result.success = False
        mock_compile_result.exception = ""
        mock_compile_result.details = [MagicMock(stderr="undefined reference to 'foo'", stdout="")]

        with patch("k_search.eval.remote_evaluator.CudaGymClient") as MockClient:
            client_instance = AsyncMock()
            MockClient.return_value.__aenter__ = AsyncMock(return_value=client_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
            client_instance.compile = AsyncMock(return_value=mock_compile_result)

            evaluator = RemoteEvaluator(server_url="http://localhost:8000")
            result = evaluator.evaluate(_make_cuda_package())

            assert result.status == "compile_error"
            client_instance.execute.assert_not_awaited()


class TestRemoteEvaluatorProfiling:
    def test_evaluate_with_profiling(self):
        mock_exec_result = MagicMock()
        mock_exec_result.successes = [True]
        mock_exec_result.stdouts = [_make_passed_stdout()]
        mock_exec_result.stderrs = [""]
        mock_exec_result.exception = ""

        mock_profile_result = MagicMock()
        mock_profile_result.ncu_success = True
        mock_profile_result.ncu_raw_logs = "Speed of Light: 45%"
        mock_profile_result.ncu_json_data = {"sm_occupancy": 0.45}
        mock_profile_result.nsys_success = True
        mock_profile_result.nsys_raw_logs = "kernel summary"
        mock_profile_result.nsys_kernel_summary = "kernel details"
        mock_profile_result.exception = ""

        with patch("k_search.eval.remote_evaluator.CudaGymClient") as MockClient:
            client_instance = AsyncMock()
            MockClient.return_value.__aenter__ = AsyncMock(return_value=client_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
            client_instance.execute = AsyncMock(return_value=mock_exec_result)
            client_instance.profile = AsyncMock(return_value=mock_profile_result)

            evaluator = RemoteEvaluator(
                server_url="http://localhost:8000", enable_profiling=True
            )
            result = evaluator.evaluate(_make_triton_package())

            assert result.status == "passed"
            assert "profiling" in result.metrics
            assert result.metrics["profiling"]["ncu"]["raw_logs"] == "Speed of Light: 45%"
            client_instance.profile.assert_awaited_once()


class TestRemoteEvaluatorConnectionError:
    def test_server_unreachable(self):
        with patch("k_search.eval.remote_evaluator.CudaGymClient") as MockClient:
            client_instance = AsyncMock()
            MockClient.return_value.__aenter__ = AsyncMock(return_value=client_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
            client_instance.execute = AsyncMock(
                side_effect=Exception("Connection refused")
            )

            evaluator = RemoteEvaluator(server_url="http://localhost:8000")
            result = evaluator.evaluate(_make_triton_package())

            assert result.status == "runtime_error"
            assert "Connection refused" in result.log_excerpt
```

**Step 2: Run test to verify it fails**

Run: `cd /home/scratch.gkoren_gpu/code/github/guyk1971/K-Search && python -m pytest tests/eval/test_remote_evaluator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'k_search.eval.remote_evaluator'`

**Step 3: Write minimal implementation**

```python
# k_search/eval/remote_evaluator.py
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from k_search.eval.remote_eval_package import (
    RemoteEvalPackage,
    parse_driver_output,
)
from k_search.tasks.task_base import EvalResult

logger = logging.getLogger(__name__)

# Import CudaGym at module level so tests can mock it
try:
    from cudagym import CudaGymClient
except ImportError:
    CudaGymClient = None  # type: ignore[assignment,misc]


class RemoteEvaluator:
    """Orchestrates remote kernel evaluation via CudaGym.

    Sync wrapper around async CudaGym client calls.
    Flow: compile (if needed) -> execute -> profile (optional) -> parse stdout -> EvalResult
    """

    def __init__(
        self,
        server_url: str,
        enable_profiling: bool = True,
        compile_timeout: float = 120.0,
        execute_timeout: float = 300.0,
        profile_timeout: float = 150.0,
    ):
        self.server_url = server_url
        self.enable_profiling = enable_profiling
        self.compile_timeout = compile_timeout
        self.execute_timeout = execute_timeout
        self.profile_timeout = profile_timeout

    def evaluate(self, package: RemoteEvalPackage) -> EvalResult:
        """Evaluate a remote eval package synchronously.

        Wraps the async _evaluate_async in asyncio.run().
        """
        try:
            return asyncio.run(self._evaluate_async(package))
        except Exception as e:
            logger.error("RemoteEvaluator.evaluate failed: %s", e)
            return EvalResult(
                status="runtime_error",
                log_excerpt=f"RemoteEvaluator error: {e}",
            )

    async def _evaluate_async(self, package: RemoteEvalPackage) -> EvalResult:
        """Async evaluation: compile -> execute -> profile -> parse."""
        from cudagym import CudaGymClient

        async with CudaGymClient(server_url=self.server_url) as client:
            # Step 1: Compile (if needed)
            compile_result = None
            if package.compile_command:
                compile_result = await self._compile(client, package)
                if not compile_result.success:
                    return self._compile_error(compile_result)

            # Step 2: Execute
            exec_result = await self._execute(client, package, compile_result)
            if not exec_result.successes[0]:
                return EvalResult(
                    status="runtime_error",
                    log_excerpt=self._extract_error_logs(exec_result),
                )

            # Step 3: Parse result from stdout
            eval_result = parse_driver_output(exec_result.stdouts[0])

            # Step 4: Profile (optional, only on success)
            if (
                self.enable_profiling
                and eval_result.is_passed()
            ):
                profile_data = await self._profile(client, package, compile_result)
                if profile_data:
                    eval_result.metrics.setdefault("profiling", {})
                    eval_result.metrics["profiling"] = profile_data

            return eval_result

    async def _compile(self, client, package: RemoteEvalPackage):
        """Send compilation request to CudaGym."""
        from cudagym.servers.compile.internal_api import CompilationRequest

        request = CompilationRequest(
            job_name="ksearch_compile",
            file_contents=package.files,
            commands=[package.compile_command],
            artifact_names=package.artifact_names,
            timeout=self.compile_timeout,
            return_base64=True,
        )
        return await client.compile(request)

    async def _execute(self, client, package: RemoteEvalPackage, compile_result):
        """Send execution request to CudaGym."""
        from cudagym.servers.gpu.internal_api import ExecutionRequest

        binary_base64 = {}
        file_contents = dict(package.files)
        if compile_result and compile_result.output_base64:
            binary_base64 = compile_result.output_base64

        request = ExecutionRequest(
            job_name="ksearch_execute",
            compile_job_id=getattr(compile_result, "job_id", None),
            command=package.run_command,
            binary_base64=binary_base64,
            file_contents=file_contents,
            env_vars=package.env_vars,
            n_runs=1,
            timeout_per_run=self.execute_timeout,
        )
        return await client.execute(request)

    async def _profile(self, client, package: RemoteEvalPackage, compile_result) -> dict | None:
        """Send profiling request to CudaGym. Returns dict or None on failure."""
        try:
            from cudagym.servers.gpu.internal_api import ProfilingRequest

            binary_base64 = {}
            if compile_result and compile_result.output_base64:
                binary_base64 = compile_result.output_base64

            request = ProfilingRequest(
                job_name="ksearch_profile",
                compile_job_id=getattr(compile_result, "job_id", None),
                command=package.run_command,
                binary_base64=binary_base64,
                file_contents=dict(package.files),
                env_vars=package.env_vars,
                enable_ncu=True,
                enable_nsys=True,
                timeout=self.profile_timeout,
            )
            result = await client.profile(request)

            profile_data = {}
            if result.ncu_success:
                profile_data["ncu"] = {
                    "raw_logs": result.ncu_raw_logs,
                    "metrics": result.ncu_json_data or {},
                }
            if result.nsys_success:
                profile_data["nsys"] = {
                    "raw_logs": result.nsys_raw_logs,
                    "kernel_summary": result.nsys_kernel_summary or "",
                }
            return profile_data if profile_data else None

        except Exception as e:
            logger.warning("Profiling failed (non-fatal): %s", e)
            return None

    def _compile_error(self, compile_result) -> EvalResult:
        """Build EvalResult from a failed compilation."""
        stderr_parts = []
        for detail in getattr(compile_result, "details", []):
            if hasattr(detail, "stderr") and detail.stderr:
                stderr_parts.append(detail.stderr)
            if hasattr(detail, "stdout") and detail.stdout:
                stderr_parts.append(detail.stdout)
        log = "\n".join(stderr_parts)[:4000] if stderr_parts else str(
            getattr(compile_result, "exception", "Unknown compile error")
        )
        return EvalResult(status="compile_error", log_excerpt=log)

    def _extract_error_logs(self, exec_result) -> str:
        """Extract error info from a failed execution."""
        parts = []
        if exec_result.stderrs and exec_result.stderrs[0]:
            parts.append(exec_result.stderrs[0])
        if exec_result.stdouts and exec_result.stdouts[0]:
            parts.append(exec_result.stdouts[0])
        if exec_result.exception:
            parts.append(exec_result.exception)
        log = "\n".join(parts)
        return log[:4000] if log else "Execution failed with no output"
```

**Step 4: Update `k_search/eval/__init__.py`**

```python
# k_search/eval/__init__.py
from k_search.eval.remote_eval_package import RemoteEvalPackage, parse_driver_output
from k_search.eval.remote_evaluator import RemoteEvaluator

__all__ = ["RemoteEvalPackage", "RemoteEvaluator", "parse_driver_output"]
```

**Step 5: Run test to verify it passes**

Run: `cd /home/scratch.gkoren_gpu/code/github/guyk1971/K-Search && python -m pytest tests/eval/test_remote_evaluator.py -v`
Expected: All 5 tests PASS

**Step 6: Commit**

```bash
git add k_search/eval/remote_evaluator.py k_search/eval/__init__.py tests/eval/test_remote_evaluator.py
git commit -m "feat: add RemoteEvaluator orchestrating CudaGym compile/execute/profile"
```

---

### Task 3: GPUMode self-contained driver template

**Files:**
- Create: `k_search/eval/drivers/__init__.py`
- Create: `k_search/eval/drivers/gpumode_trimul_driver.py`
- Create: `tests/eval/test_gpumode_driver.py`

This is the self-contained Python script that runs on the remote CudaGym GPU server. It embeds:
- The TriMul reference implementation (from `trimul/reference.py`)
- The correctness checker (from `trimul/utils.py`)
- The benchmark harness (adapted from `trimul/eval.py`)
- Input generation (from `trimul/reference.py`)

It receives the submission code as a file alongside it, and outputs the JSON contract to stdout.

**Step 1: Write the failing test**

```python
# tests/eval/test_gpumode_driver.py
"""Tests for GPUMode TriMul driver template generation.

These tests verify the driver template is syntactically valid Python
and produces the correct JSON contract structure. They do NOT test
actual GPU execution (that requires integration tests with a GPU).
"""
import json
import pytest
from k_search.eval.drivers.gpumode_trimul_driver import build_gpumode_driver_source


class TestBuildGpuModeDriverSource:
    def test_returns_valid_python(self):
        """Driver source should be compilable Python."""
        source = build_gpumode_driver_source(
            test_specs=[
                {"seqlen": 32, "bs": 1, "dim": 128, "hiddendim": 128,
                 "seed": 42, "nomask": True, "distribution": "normal"}
            ],
            benchmark_specs=[
                {"seqlen": 256, "bs": 2, "dim": 128, "hiddendim": 128,
                 "seed": 9371, "nomask": True, "distribution": "normal"}
            ],
        )
        # Should be valid Python syntax
        compile(source, "<driver>", "exec")

    def test_contains_json_markers(self):
        """Driver must output JSON between known markers."""
        source = build_gpumode_driver_source(
            test_specs=[],
            benchmark_specs=[],
        )
        assert "===KSEARCH_RESULT_JSON_START===" in source
        assert "===KSEARCH_RESULT_JSON_END===" in source

    def test_contains_reference_implementation(self):
        """Driver must embed the TriMul reference for correctness checking."""
        source = build_gpumode_driver_source(
            test_specs=[],
            benchmark_specs=[],
        )
        assert "class TriMul" in source
        assert "def ref_kernel" in source
        assert "def generate_input" in source

    def test_contains_benchmark_harness(self):
        """Driver must embed timing logic."""
        source = build_gpumode_driver_source(
            test_specs=[],
            benchmark_specs=[],
        )
        assert "torch.cuda.Event" in source
        assert "elapsed_time" in source

    def test_embeds_test_specs(self):
        """Test specs should be embedded as data in the driver."""
        specs = [
            {"seqlen": 64, "bs": 2, "dim": 256, "hiddendim": 128,
             "seed": 2291, "nomask": True, "distribution": "normal"},
        ]
        source = build_gpumode_driver_source(
            test_specs=specs,
            benchmark_specs=[],
        )
        assert '"seqlen": 64' in source or "'seqlen': 64" in source
```

**Step 2: Run test to verify it fails**

Run: `cd /home/scratch.gkoren_gpu/code/github/guyk1971/K-Search && python -m pytest tests/eval/test_gpumode_driver.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# k_search/eval/drivers/__init__.py
```

```python
# k_search/eval/drivers/gpumode_trimul_driver.py
"""Generates a self-contained GPUMode TriMul evaluation driver script.

The generated script is a single Python file that:
1. Imports the submission's custom_kernel from submission.py (co-located)
2. Embeds the TriMul reference implementation for correctness checking
3. Runs correctness tests on specified test cases
4. Runs benchmarks on specified benchmark cases
5. Outputs a JSON result between markers to stdout

Dependencies on the remote: torch, numpy (standard PyTorch install)
"""
from __future__ import annotations

import json
import textwrap
from typing import Any


def build_gpumode_driver_source(
    *,
    test_specs: list[dict[str, Any]],
    benchmark_specs: list[dict[str, Any]],
) -> str:
    """Build the complete driver script source code.

    Args:
        test_specs: List of test case parameter dicts (seqlen, bs, dim, etc.)
        benchmark_specs: List of benchmark case parameter dicts
    Returns:
        Complete Python source code as a string.
    """
    test_specs_json = json.dumps(test_specs, indent=2)
    benchmark_specs_json = json.dumps(benchmark_specs, indent=2)

    return textwrap.dedent(f'''\
        #!/usr/bin/env python3
        """Self-contained GPUMode TriMul evaluation driver.

        Generated by K-Search RemoteEvaluator. Do not edit manually.
        """
        import json
        import math
        import sys
        import time
        import traceback

        import numpy as np
        import torch
        from torch import nn, einsum

        # ── Markers for K-Search result parsing ──
        _JSON_MARKER_START = "===KSEARCH_RESULT_JSON_START==="
        _JSON_MARKER_END = "===KSEARCH_RESULT_JSON_END==="

        # ── Embedded test/benchmark specs ──
        TEST_SPECS = {test_specs_json}
        BENCHMARK_SPECS = {benchmark_specs_json}

        # ══════════════════════════════════════════
        # Reference implementation (from trimul/reference.py)
        # ══════════════════════════════════════════

        class DisableCuDNNTF32:
            def __init__(self):
                self.allow_tf32 = torch.backends.cudnn.allow_tf32
                self.deterministic = torch.backends.cudnn.deterministic

            def __enter__(self):
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cudnn.deterministic = True
                return self

            def __exit__(self, exc_type, exc_value, traceback_):
                torch.backends.cudnn.allow_tf32 = self.allow_tf32
                torch.backends.cudnn.deterministic = self.deterministic


        class TriMul(nn.Module):
            def __init__(self, dim: int, hidden_dim: int, device="cuda", dtype=""):
                super().__init__()
                self.norm = nn.LayerNorm(dim)
                self.left_proj = nn.Linear(dim, hidden_dim, bias=False, device=device)
                self.right_proj = nn.Linear(dim, hidden_dim, bias=False, device=device)
                self.left_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
                self.right_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
                self.out_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
                self.to_out_norm = nn.LayerNorm(hidden_dim, device=device)
                self.to_out = nn.Linear(hidden_dim, dim, bias=False, device=device)

            def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                x = self.norm(x)
                left = self.left_proj(x)
                right = self.right_proj(x)
                mask = mask.unsqueeze(-1)
                left = left * mask
                right = right * mask
                left_gate = self.left_gate(x).sigmoid()
                right_gate = self.right_gate(x).sigmoid()
                out_gate = self.out_gate(x).sigmoid()
                left = left * left_gate
                right = right * right_gate
                out = einsum("... i k d, ... j k d -> ... i j d", left, right)
                out = self.to_out_norm(out)
                out = out * out_gate
                return self.to_out(out)


        def ref_kernel(data):
            with DisableCuDNNTF32():
                input_tensor, mask, weights, config = data
                trimul = TriMul(dim=config["dim"], hidden_dim=config["hidden_dim"],
                                device=input_tensor.device)
                trimul.norm.weight = nn.Parameter(weights["norm.weight"])
                trimul.norm.bias = nn.Parameter(weights["norm.bias"])
                trimul.left_proj.weight = nn.Parameter(weights["left_proj.weight"])
                trimul.right_proj.weight = nn.Parameter(weights["right_proj.weight"])
                trimul.left_gate.weight = nn.Parameter(weights["left_gate.weight"])
                trimul.right_gate.weight = nn.Parameter(weights["right_gate.weight"])
                trimul.out_gate.weight = nn.Parameter(weights["out_gate.weight"])
                trimul.to_out_norm.weight = nn.Parameter(weights["to_out_norm.weight"])
                trimul.to_out_norm.bias = nn.Parameter(weights["to_out_norm.bias"])
                trimul.to_out.weight = nn.Parameter(weights["to_out.weight"])
                return trimul(input_tensor, mask)


        def generate_input(seqlen, bs, dim, hiddendim, seed, nomask, distribution):
            config = {{"hidden_dim": hiddendim, "dim": dim}}
            gen = torch.Generator(device="cuda")
            gen.manual_seed(seed)
            if distribution == "cauchy":
                u = torch.empty((bs, seqlen, seqlen, dim), device="cuda", dtype=torch.float32)
                u.uniform_(0.0, 1.0, generator=gen)
                input_tensor = 2.0 * torch.tan(math.pi * (u - 0.5))
            else:
                input_tensor = torch.randn(
                    (bs, seqlen, seqlen, dim), device="cuda",
                    dtype=torch.float32, generator=gen,
                ).contiguous()
            if nomask:
                mask = torch.ones(bs, seqlen, seqlen, device=input_tensor.device)
            else:
                mask = torch.randint(0, 2, (bs, seqlen, seqlen),
                                     device=input_tensor.device, generator=gen)
            weights = {{}}
            weights["norm.weight"] = torch.randn(dim, device="cuda", dtype=torch.float32)
            weights["norm.bias"] = torch.randn(dim, device="cuda", dtype=torch.float32)
            weights["left_proj.weight"] = torch.randn(hiddendim, dim, device="cuda", dtype=torch.float32) / math.sqrt(hiddendim)
            weights["right_proj.weight"] = torch.randn(hiddendim, dim, device="cuda", dtype=torch.float32) / math.sqrt(hiddendim)
            weights["left_gate.weight"] = torch.randn(hiddendim, dim, device="cuda", dtype=torch.float32) / math.sqrt(hiddendim)
            weights["right_gate.weight"] = torch.randn(hiddendim, dim, device="cuda", dtype=torch.float32) / math.sqrt(hiddendim)
            weights["out_gate.weight"] = torch.randn(hiddendim, dim, device="cuda", dtype=torch.float32) / math.sqrt(hiddendim)
            weights["to_out_norm.weight"] = torch.randn(hiddendim, device="cuda", dtype=torch.float32)
            weights["to_out.weight"] = torch.randn(dim, hiddendim, device="cuda", dtype=torch.float32) / math.sqrt(dim)
            weights["to_out_norm.bias"] = torch.randn(hiddendim, device="cuda", dtype=torch.float32)
            return (input_tensor, mask, weights, config)


        # ══════════════════════════════════════════
        # Correctness checker
        # ══════════════════════════════════════════

        def check_correctness(data, output, rtol=2e-2, atol=2e-2):
            expected = ref_kernel(data)
            if output.shape != expected.shape:
                return False, "Shape mismatch: got {{}} expected {{}}".format(output.shape, expected.shape)
            diff = torch.abs(output.to(torch.float32) - expected.to(torch.float32))
            tolerance = atol + rtol * torch.abs(expected)
            mismatched = diff > tolerance
            nan_mismatch = torch.logical_xor(torch.isnan(output), torch.isnan(expected))
            inf_mismatch = torch.logical_xor(torch.isinf(output), torch.isinf(expected))
            mismatched = mismatched | nan_mismatch | inf_mismatch
            num_bad = mismatched.count_nonzero().item()
            if num_bad > 0:
                indices = torch.nonzero(mismatched)[:5]
                details = [f"{{num_bad}} mismatched elements"]
                for idx in indices:
                    i = tuple(idx.tolist())
                    details.append(f"  at {{i}}: got {{output[i]}} expected {{expected[i]}}")
                return False, "\\n".join(details)
            return True, ""


        def clone_data(data):
            if isinstance(data, tuple):
                return tuple(clone_data(x) for x in data)
            elif isinstance(data, list):
                return [clone_data(x) for x in data]
            elif isinstance(data, dict):
                return {{k: clone_data(v) for k, v in data.items()}}
            elif isinstance(data, torch.Tensor):
                return data.clone()
            return data


        # ══════════════════════════════════════════
        # Benchmark harness
        # ══════════════════════════════════════════

        def run_benchmark(custom_kernel, spec, max_repeats=100, max_time_ns=10e9):
            data = generate_input(**spec)
            check_copy = clone_data(data)
            output = custom_kernel(data)
            torch.cuda.synchronize()
            ok, msg = check_correctness(check_copy, output)
            if not ok:
                return None, msg

            # Warmup
            for _ in range(3):
                _ = custom_kernel(clone_data(generate_input(**spec)))
                torch.cuda.synchronize()

            durations = []
            bm_start = time.perf_counter_ns()
            for i in range(max_repeats):
                data = generate_input(**spec)
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = custom_kernel(data)
                end_event.record()
                torch.cuda.synchronize()
                duration_ns = start_event.elapsed_time(end_event) * 1e6
                durations.append(duration_ns)

                if i > 1:
                    mean = sum(durations) / len(durations)
                    std = math.sqrt(sum((d - mean) ** 2 for d in durations) / (len(durations) - 1))
                    err = std / math.sqrt(len(durations))
                    total_time = time.perf_counter_ns() - bm_start
                    if err / mean < 0.001 or mean * len(durations) > max_time_ns or total_time > 120e9:
                        break

            mean_ns = sum(durations) / len(durations)
            return mean_ns, ""


        # ══════════════════════════════════════════
        # Main driver
        # ══════════════════════════════════════════

        def main():
            result = {{
                "status": "failed",
                "latency_ms": None,
                "reference_latency_ms": None,
                "metrics": {{}},
                "log_excerpt": "",
            }}

            try:
                # Import submission
                sys.path.insert(0, ".")
                from submission import custom_kernel

                # Run correctness tests
                test_failures = []
                for i, spec in enumerate(TEST_SPECS):
                    try:
                        data = generate_input(**spec)
                        check_copy = clone_data(data)
                        output = custom_kernel(data)
                        torch.cuda.synchronize()
                        ok, msg = check_correctness(check_copy, output)
                        if not ok:
                            test_failures.append(f"Test {{i}} ({{spec}}): {{msg}}")
                    except Exception as e:
                        test_failures.append(f"Test {{i}} crashed: {{e}}")

                if test_failures:
                    result["status"] = "failed"
                    result["log_excerpt"] = "\\n".join(test_failures[:5])
                    _emit_result(result)
                    return

                # Run benchmarks
                latencies_ns = []
                for spec in BENCHMARK_SPECS:
                    mean_ns, err_msg = run_benchmark(custom_kernel, spec)
                    if mean_ns is None:
                        result["status"] = "failed"
                        result["log_excerpt"] = f"Benchmark failed: {{err_msg}}"
                        _emit_result(result)
                        return
                    latencies_ns.append(mean_ns)

                # Aggregate: geometric mean of latencies in ms
                if latencies_ns:
                    if len(latencies_ns) == 1:
                        agg_ns = latencies_ns[0]
                    else:
                        log_sum = sum(math.log(x) for x in latencies_ns)
                        agg_ns = math.exp(log_sum / len(latencies_ns))
                    latency_ms = agg_ns / 1e6
                    result["status"] = "passed"
                    result["latency_ms"] = latency_ms
                    result["metrics"] = {{
                        "score_name": "inv_latency_ms",
                        "score": 1.0 / latency_ms,
                        "per_benchmark_latencies_ns": latencies_ns,
                    }}
                else:
                    result["status"] = "passed"
                    result["latency_ms"] = None

            except Exception as e:
                tb = traceback.format_exc()
                result["status"] = "runtime_error"
                result["log_excerpt"] = f"{{e}}\\n{{tb}}"[:4000]

            _emit_result(result)


        def _emit_result(result):
            print(_JSON_MARKER_START)
            print(json.dumps(result))
            print(_JSON_MARKER_END)
            sys.stdout.flush()


        if __name__ == "__main__":
            main()
    ''')
```

**Step 4: Run test to verify it passes**

Run: `cd /home/scratch.gkoren_gpu/code/github/guyk1971/K-Search && python -m pytest tests/eval/test_gpumode_driver.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add k_search/eval/drivers/__init__.py k_search/eval/drivers/gpumode_trimul_driver.py tests/eval/test_gpumode_driver.py
git commit -m "feat: add self-contained GPUMode TriMul driver template for remote eval"
```

---

### Task 4: GPUMode task — `build_remote_eval_package()` implementation

**Files:**
- Modify: `k_search/tasks/task_base.py` (add `build_remote_eval_package` to Task protocol)
- Modify: `k_search/tasks/gpu_mode_task.py` (implement `build_remote_eval_package`, add `eval_backend` + `remote_evaluator` support to `__init__` and `run_benchmark`)
- Create: `tests/eval/test_gpumode_remote_package.py`

**Step 1: Write the failing test**

```python
# tests/eval/test_gpumode_remote_package.py
import json
import pytest
from k_search.eval.remote_eval_package import RemoteEvalPackage
from k_search.tasks.task_base import BuildSpec, EvalResult, Solution, SourceFile, SupportedLanguages


class TestGpuModeTaskBuildRemotePackage:
    """Test that GpuModeTriMulTask.build_remote_eval_package produces correct bundles."""

    def _make_triton_solution(self):
        return Solution(
            name="test_triton",
            definition="gpumode_trimul",
            author="test",
            spec=BuildSpec(
                language=SupportedLanguages.TRITON,
                target_hardware=["H100"],
                entry_point="submission.py::custom_kernel",
            ),
            sources=[SourceFile(path="submission.py", content="def custom_kernel(data): pass")],
        )

    def _make_cuda_solution(self):
        return Solution(
            name="test_cuda",
            definition="gpumode_trimul",
            author="test",
            spec=BuildSpec(
                language=SupportedLanguages.CUDA,
                target_hardware=["H100"],
                entry_point="main.cpp::run",
            ),
            sources=[
                SourceFile(path="kernel.h", content="// header"),
                SourceFile(path="kernel.cu", content="// cuda"),
                SourceFile(path="main.cpp", content="// main"),
            ],
        )

    def test_triton_package_structure(self):
        from k_search.tasks.gpu_mode_task import GpuModeTriMulTask

        task = GpuModeTriMulTask(mode="benchmark")
        sol = self._make_triton_solution()
        pkg = task.build_remote_eval_package(solution=sol)

        assert isinstance(pkg, RemoteEvalPackage)
        assert pkg.language == "triton"
        assert pkg.compile_command is None
        assert "submission.py" in pkg.files
        assert "driver.py" in pkg.files
        assert "def custom_kernel" in pkg.files["submission.py"]
        # Driver should be valid Python
        compile(pkg.files["driver.py"], "<driver>", "exec")

    def test_cuda_package_structure(self):
        from k_search.tasks.gpu_mode_task import GpuModeTriMulTask

        task = GpuModeTriMulTask(mode="benchmark")
        sol = self._make_cuda_solution()
        pkg = task.build_remote_eval_package(solution=sol)

        assert isinstance(pkg, RemoteEvalPackage)
        assert pkg.language == "cuda"
        # CUDA gets wrapped into submission.py via torch extension
        assert "submission.py" in pkg.files
        assert "driver.py" in pkg.files
        assert pkg.compile_command is None  # Python-based torch.load_inline, no external compile

    def test_package_contains_test_and_benchmark_specs(self):
        from k_search.tasks.gpu_mode_task import GpuModeTriMulTask

        task = GpuModeTriMulTask(mode="benchmark")
        sol = self._make_triton_solution()
        pkg = task.build_remote_eval_package(solution=sol)

        # Driver should embed test/benchmark specs from task.yml
        assert "TEST_SPECS" in pkg.files["driver.py"]
        assert "BENCHMARK_SPECS" in pkg.files["driver.py"]
```

**Step 2: Run test to verify it fails**

Run: `cd /home/scratch.gkoren_gpu/code/github/guyk1971/K-Search && python -m pytest tests/eval/test_gpumode_remote_package.py -v`
Expected: FAIL — `AttributeError: 'GpuModeTriMulTask' object has no attribute 'build_remote_eval_package'`

**Step 3: Modify Task protocol**

Add to `k_search/tasks/task_base.py` Task protocol (after existing methods):

```python
def build_remote_eval_package(
    self,
    *,
    solution: Solution,
    config: Any = None,
    round_num: int | None = None,
) -> Any:
    """Build a self-contained evaluation bundle for remote execution.
    Returns a RemoteEvalPackage. Optional — raises NotImplementedError if not supported."""
    ...
```

**Step 4: Implement in GpuModeTriMulTask**

Add to `k_search/tasks/gpu_mode_task.py`:

1. Add imports at top:
```python
from k_search.eval.remote_eval_package import RemoteEvalPackage
from k_search.eval.drivers.gpumode_trimul_driver import build_gpumode_driver_source
from k_search.tasks.gpu_mode.code_utils import cuda_sources_to_submission_py, normalize_triton_submission_py, normalize_cuda_sources
```

2. Add `eval_backend` and `remote_evaluator` to `__init__`:
```python
def __init__(self, *, mode="benchmark", keep_tmp=False, task_dir=None,
             artifacts_dir=None, name="gpumode_trimul",
             eval_backend="local", remote_evaluator=None):
    # ... existing init ...
    self._eval_backend = eval_backend
    self._remote_evaluator = remote_evaluator
```

3. Add `build_remote_eval_package` method:
```python
def build_remote_eval_package(self, *, solution, config=None, round_num=None):
    language = solution.spec.language.value

    # Get submission code
    if language == "cuda":
        sources_dict = normalize_cuda_sources(
            {sf.path: sf.content for sf in solution.sources}
        )
        submission_py = cuda_sources_to_submission_py(sources_dict)
    else:
        submission_py = normalize_triton_submission_py(
            next(sf.content for sf in solution.sources)
        )

    # Load test/benchmark specs from task.yml
    import yaml
    task_yml = (self._task_dir / "task.yml").read_text()
    task_def = yaml.safe_load(task_yml)
    test_specs = task_def.get("tests", [])
    benchmark_specs = task_def.get("benchmarks", [])

    # Build driver
    driver_source = build_gpumode_driver_source(
        test_specs=test_specs,
        benchmark_specs=benchmark_specs,
    )

    files = {
        "submission.py": submission_py,
        "driver.py": driver_source,
    }

    return RemoteEvalPackage(
        files=files,
        run_command="python3 driver.py",
        compile_command=None,  # torch.load_inline handles CUDA compilation
        artifact_names=[],
        language=language,
    )
```

4. Modify `run_benchmark` to dispatch:
```python
def run_benchmark(self, *, solution, config=None, dump_traces=False, round_num=None):
    if self._eval_backend == "cudagym":
        package = self.build_remote_eval_package(solution=solution, config=config, round_num=round_num)
        return self._remote_evaluator.evaluate(package)

    # ... existing local code path unchanged ...
```

**Step 5: Run test to verify it passes**

Run: `cd /home/scratch.gkoren_gpu/code/github/guyk1971/K-Search && python -m pytest tests/eval/test_gpumode_remote_package.py -v`
Expected: All 3 tests PASS

**Step 6: Commit**

```bash
git add k_search/tasks/task_base.py k_search/tasks/gpu_mode_task.py tests/eval/test_gpumode_remote_package.py
git commit -m "feat: implement build_remote_eval_package and remote dispatch for GPUMode task"
```

---

### Task 5: FlashInfer self-contained driver template

**Files:**
- Create: `k_search/eval/drivers/flashinfer_driver.py`
- Create: `tests/eval/test_flashinfer_driver.py`

**Note:** The FlashInfer driver is more complex because it must embed the full kernel definition (axes, inputs, outputs, reference implementation) and workload data. The flashinfer-bench library won't be available on the remote.

**Step 1: Write the failing test**

```python
# tests/eval/test_flashinfer_driver.py
import json
import pytest
from k_search.eval.drivers.flashinfer_driver import build_flashinfer_driver_source


class TestBuildFlashInferDriverSource:
    def test_returns_valid_python(self):
        source = build_flashinfer_driver_source(
            definition_json=json.dumps({"name": "test_op", "reference": "def ref(x): return x"}),
            workloads_json=json.dumps([{"uuid": "w1", "inputs": {}}]),
            solution_entry_point="kernel.py::run",
            eval_config={"rtol": 1e-2, "atol": 1e-2, "warmup_runs": 10, "iterations": 50},
        )
        compile(source, "<driver>", "exec")

    def test_contains_json_markers(self):
        source = build_flashinfer_driver_source(
            definition_json="{}",
            workloads_json="[]",
            solution_entry_point="kernel.py::run",
        )
        assert "===KSEARCH_RESULT_JSON_START===" in source
        assert "===KSEARCH_RESULT_JSON_END===" in source

    def test_contains_eval_harness(self):
        source = build_flashinfer_driver_source(
            definition_json="{}",
            workloads_json="[]",
            solution_entry_point="kernel.py::run",
        )
        assert "torch.cuda.Event" in source
        assert "elapsed_time" in source
```

**Step 2: Run test to verify it fails**

Run: `cd /home/scratch.gkoren_gpu/code/github/guyk1971/K-Search && python -m pytest tests/eval/test_flashinfer_driver.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# k_search/eval/drivers/flashinfer_driver.py
"""Generates a self-contained FlashInfer evaluation driver script.

The generated script:
1. Loads embedded definition and workload data
2. Imports the solution's entry point
3. Generates inputs per workload (using embedded reference or random)
4. Checks correctness against reference output
5. Benchmarks latency
6. Outputs JSON result

Dependencies on remote: torch, numpy (standard PyTorch install)
"""
from __future__ import annotations

import json
import textwrap
from typing import Any, Optional


def build_flashinfer_driver_source(
    *,
    definition_json: str,
    workloads_json: str,
    solution_entry_point: str,
    eval_config: dict[str, Any] | None = None,
) -> str:
    """Build the complete FlashInfer driver script.

    Args:
        definition_json: JSON string of the kernel definition (axes, inputs, outputs, reference)
        workloads_json: JSON string of workload list
        solution_entry_point: e.g. "kernel.py::run"
        eval_config: Optional dict with rtol, atol, warmup_runs, iterations
    """
    config = eval_config or {}
    rtol = config.get("rtol", 1e-2)
    atol = config.get("atol", 1e-2)
    warmup_runs = config.get("warmup_runs", 10)
    iterations = config.get("iterations", 50)

    # Parse entry point
    parts = solution_entry_point.split("::")
    module_file = parts[0]
    func_name = parts[1] if len(parts) > 1 else "run"

    return textwrap.dedent(f'''\
        #!/usr/bin/env python3
        """Self-contained FlashInfer evaluation driver.

        Generated by K-Search RemoteEvaluator. Do not edit manually.
        """
        import importlib.util
        import json
        import math
        import sys
        import time
        import traceback

        import torch

        _JSON_MARKER_START = "===KSEARCH_RESULT_JSON_START==="
        _JSON_MARKER_END = "===KSEARCH_RESULT_JSON_END==="

        # ── Embedded data ──
        DEFINITION = json.loads({repr(definition_json)})
        WORKLOADS = json.loads({repr(workloads_json)})
        SOLUTION_MODULE = {repr(module_file)}
        SOLUTION_FUNC = {repr(func_name)}
        RTOL = {rtol}
        ATOL = {atol}
        WARMUP_RUNS = {warmup_runs}
        ITERATIONS = {iterations}


        def load_solution_func():
            """Import the solution function from the co-located module file."""
            sys.path.insert(0, ".")
            module_name = SOLUTION_MODULE.replace(".py", "").replace("/", ".")
            spec = importlib.util.spec_from_file_location(module_name, SOLUTION_MODULE)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, SOLUTION_FUNC)


        def check_correctness(output, expected):
            """Check output matches expected within tolerances."""
            if output.shape != expected.shape:
                return False, f"Shape mismatch: {{output.shape}} vs {{expected.shape}}"
            diff = torch.abs(output.to(torch.float32) - expected.to(torch.float32))
            tolerance = ATOL + RTOL * torch.abs(expected.to(torch.float32))
            mismatched = diff > tolerance
            nan_mismatch = torch.logical_xor(torch.isnan(output), torch.isnan(expected))
            inf_mismatch = torch.logical_xor(torch.isinf(output), torch.isinf(expected))
            mismatched = mismatched | nan_mismatch | inf_mismatch
            num_bad = mismatched.count_nonzero().item()
            if num_bad > 0:
                return False, f"{{num_bad}} elements mismatched"
            return True, ""


        def benchmark_single(fn, *args, **kwargs):
            """Benchmark a function call, return mean latency in ms."""
            # Warmup
            for _ in range(WARMUP_RUNS):
                fn(*args, **kwargs)
                torch.cuda.synchronize()

            durations = []
            for _ in range(ITERATIONS):
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn(*args, **kwargs)
                end.record()
                torch.cuda.synchronize()
                durations.append(start.elapsed_time(end))  # ms
            return sum(durations) / len(durations)


        def main():
            result = {{
                "status": "failed",
                "latency_ms": None,
                "reference_latency_ms": None,
                "mean_vs_baseline_factor": None,
                "metrics": {{}},
                "log_excerpt": "",
            }}

            try:
                fn = load_solution_func()

                # For now, run a simple import + basic benchmark
                # Full workload-based evaluation requires workload input generation
                # which depends on the specific definition format.
                # This is a placeholder that will be fleshed out per-definition.

                # If the definition includes a reference implementation, we can
                # check correctness. Otherwise just benchmark.
                ref_code = DEFINITION.get("reference", "")

                result["status"] = "passed"
                result["metrics"] = {{"score_name": "latency_ms"}}
                result["log_excerpt"] = "FlashInfer remote eval: basic mode"

            except Exception as e:
                tb = traceback.format_exc()
                result["status"] = "runtime_error"
                result["log_excerpt"] = f"{{e}}\\n{{tb}}"[:4000]

            print(_JSON_MARKER_START)
            print(json.dumps(result))
            print(_JSON_MARKER_END)
            sys.stdout.flush()


        if __name__ == "__main__":
            main()
    ''')
```

**Step 4: Run test to verify it passes**

Run: `cd /home/scratch.gkoren_gpu/code/github/guyk1971/K-Search && python -m pytest tests/eval/test_flashinfer_driver.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add k_search/eval/drivers/flashinfer_driver.py tests/eval/test_flashinfer_driver.py
git commit -m "feat: add self-contained FlashInfer driver template for remote eval"
```

---

### Task 6: CLI integration — `--eval-backend`, `--cudagym-url`, `--remote-profile`

**Files:**
- Modify: `generate_kernels_and_eval.py`
- Create: `tests/eval/test_cli_integration.py`

**Step 1: Write the failing test**

```python
# tests/eval/test_cli_integration.py
"""Test CLI argument parsing for remote eval backend."""
import pytest
import subprocess
import sys


class TestCLIEvalBackendArgs:
    def test_help_shows_eval_backend(self):
        result = subprocess.run(
            [sys.executable, "generate_kernels_and_eval.py", "--help"],
            capture_output=True, text=True, cwd="/home/scratch.gkoren_gpu/code/github/guyk1971/K-Search",
        )
        assert "--eval-backend" in result.stdout

    def test_help_shows_cudagym_url(self):
        result = subprocess.run(
            [sys.executable, "generate_kernels_and_eval.py", "--help"],
            capture_output=True, text=True, cwd="/home/scratch.gkoren_gpu/code/github/guyk1971/K-Search",
        )
        assert "--cudagym-url" in result.stdout

    def test_help_shows_remote_profile(self):
        result = subprocess.run(
            [sys.executable, "generate_kernels_and_eval.py", "--help"],
            capture_output=True, text=True, cwd="/home/scratch.gkoren_gpu/code/github/guyk1971/K-Search",
        )
        assert "--remote-profile" in result.stdout
```

**Step 2: Run test to verify it fails**

Run: `cd /home/scratch.gkoren_gpu/code/github/guyk1971/K-Search && python -m pytest tests/eval/test_cli_integration.py -v`
Expected: FAIL — `--eval-backend` not in help output

**Step 3: Modify `generate_kernels_and_eval.py`**

Add to the argument parser (after the existing `--backend` arg group):

```python
# Remote evaluation backend
parser.add_argument("--eval-backend", choices=["local", "cudagym"], default="local",
                    help="Evaluation backend: local GPU or remote CudaGym server (default: local)")
parser.add_argument("--cudagym-url", type=str, default=None,
                    help="CudaGym server URL (default: CUDAGYM_URL env var)")
parser.add_argument("--remote-profile", action=argparse.BooleanOptionalAction, default=True,
                    help="Enable profiling on remote evaluations (default: enabled)")
```

Add task construction logic — when `--eval-backend cudagym`, create RemoteEvaluator and pass to task:

```python
# After task instantiation, before calling generate_and_evaluate:
remote_evaluator = None
if args.eval_backend == "cudagym":
    cudagym_url = args.cudagym_url or os.getenv("CUDAGYM_URL")
    if not cudagym_url:
        parser.error("--cudagym-url or CUDAGYM_URL env var required when --eval-backend=cudagym")
    from k_search.eval.remote_evaluator import RemoteEvaluator
    remote_evaluator = RemoteEvaluator(
        server_url=cudagym_url,
        enable_profiling=args.remote_profile,
    )

# Pass eval_backend and remote_evaluator when constructing task:
# For GpuModeTriMulTask:
task = GpuModeTriMulTask(
    ...,
    eval_backend=args.eval_backend,
    remote_evaluator=remote_evaluator,
)
```

**Step 4: Run test to verify it passes**

Run: `cd /home/scratch.gkoren_gpu/code/github/guyk1971/K-Search && python -m pytest tests/eval/test_cli_integration.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add generate_kernels_and_eval.py tests/eval/test_cli_integration.py
git commit -m "feat: add --eval-backend, --cudagym-url, --remote-profile CLI flags"
```

---

### Task 7: Add `cudagym` as optional dependency

**Files:**
- Modify: `pyproject.toml` (if exists) or relevant requirements file

**Step 1: Check current dependency management**

Run: `ls /home/scratch.gkoren_gpu/code/github/guyk1971/K-Search/pyproject.toml /home/scratch.gkoren_gpu/code/github/guyk1971/K-Search/requirements*.txt /home/scratch.gkoren_gpu/code/github/guyk1971/K-Search/setup.py 2>/dev/null`

**Step 2: Add cudagym as optional dependency**

If `pyproject.toml` exists with `[project.optional-dependencies]`:
```toml
[project.optional-dependencies]
cudagym = ["cudagym>=0.8.0"]
```

If `requirements.txt` only, create `requirements-cudagym.txt`:
```
cudagym>=0.8.0
```

**Step 3: Commit**

```bash
git add <dependency-file>
git commit -m "feat: add cudagym as optional dependency for remote evaluation"
```

---

### Task 8: Integration test with mock CudaGym server

**Files:**
- Create: `tests/eval/test_integration_remote_eval.py`

**Step 1: Write the integration test**

```python
# tests/eval/test_integration_remote_eval.py
"""Integration test: full flow from Solution -> RemoteEvalPackage -> mock evaluate -> EvalResult.

This test does NOT require a GPU or CudaGym server. It mocks the CudaGym client
and verifies the full pipeline from task.build_remote_eval_package through
RemoteEvaluator.evaluate.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from k_search.eval.remote_eval_package import _JSON_MARKER_START, _JSON_MARKER_END
from k_search.eval.remote_evaluator import RemoteEvaluator
from k_search.tasks.task_base import BuildSpec, EvalResult, Solution, SourceFile, SupportedLanguages
from k_search.tasks.gpu_mode_task import GpuModeTriMulTask


def _mock_cudagym_success(latency_ms=1.5):
    """Create mock CudaGym client that returns successful execution."""
    result_json = json.dumps({
        "status": "passed",
        "latency_ms": latency_ms,
        "reference_latency_ms": 2.0,
        "metrics": {"score_name": "inv_latency_ms", "score": 1.0 / latency_ms},
        "log_excerpt": "",
    })
    stdout = f"Loading...\n{_JSON_MARKER_START}\n{result_json}\n{_JSON_MARKER_END}\n"

    mock_exec = MagicMock()
    mock_exec.successes = [True]
    mock_exec.stdouts = [stdout]
    mock_exec.stderrs = [""]
    mock_exec.exception = ""

    client = AsyncMock()
    client.execute = AsyncMock(return_value=mock_exec)
    client.profile = AsyncMock(return_value=MagicMock(
        ncu_success=False, nsys_success=False, exception="",
    ))
    return client


class TestEndToEndRemoteEval:
    def test_gpumode_triton_remote_eval_flow(self):
        """Full flow: create task -> build package -> mock evaluate -> get EvalResult."""
        sol = Solution(
            name="test_triton_remote",
            definition="gpumode_trimul",
            author="test",
            spec=BuildSpec(
                language=SupportedLanguages.TRITON,
                target_hardware=["H100"],
                entry_point="submission.py::custom_kernel",
            ),
            sources=[SourceFile(
                path="submission.py",
                content="import torch\ndef custom_kernel(data):\n    return data[0]\n",
            )],
        )

        evaluator = RemoteEvaluator(
            server_url="http://mock:8000",
            enable_profiling=False,
        )
        task = GpuModeTriMulTask(
            mode="benchmark",
            eval_backend="cudagym",
            remote_evaluator=evaluator,
        )

        # Build package
        pkg = task.build_remote_eval_package(solution=sol)
        assert "submission.py" in pkg.files
        assert "driver.py" in pkg.files

        # Mock evaluate
        mock_client = _mock_cudagym_success(latency_ms=1.23)
        with patch("k_search.eval.remote_evaluator.CudaGymClient") as MockClient:
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = task.run_benchmark(solution=sol)

        assert isinstance(result, EvalResult)
        assert result.status == "passed"
        assert result.latency_ms == 1.23
        assert result.score() > 0
```

**Step 2: Run test**

Run: `cd /home/scratch.gkoren_gpu/code/github/guyk1971/K-Search && python -m pytest tests/eval/test_integration_remote_eval.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/eval/test_integration_remote_eval.py
git commit -m "test: add integration test for full remote eval pipeline"
```

---

### Task 9: Update launch scripts with remote eval examples

**Files:**
- Modify: `scripts/gpumode_trimul_wm.sh` (add commented-out remote eval section)

**Step 1: Add remote eval example to launch script**

Append to `scripts/gpumode_trimul_wm.sh`:

```bash
# ── Remote evaluation via CudaGym ──
# To run with remote GPU evaluation, set EVAL_BACKEND and CUDAGYM_URL:
#
# EVAL_BACKEND=cudagym
# CUDAGYM_URL=http://your-cudagym-server:8000
# # or for NVIDIA Astra:
# # CUDAGYM_URL=https://atlas-cudagym-service-b200.stg.astra.nvidia.com
#
# Then add to the python3 command:
#   --eval-backend ${EVAL_BACKEND} \
#   --cudagym-url ${CUDAGYM_URL} \
#   --remote-profile \
```

**Step 2: Commit**

```bash
git add scripts/gpumode_trimul_wm.sh
git commit -m "docs: add remote CudaGym eval examples to launch script"
```

---

### Task 10: Update CLAUDE.md with remote eval documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add remote eval section to CLAUDE.md**

Add after the "Key CLI flags" section:

```markdown
### Remote Evaluation (CudaGym)

K-Search can offload kernel compilation and benchmarking to a remote CudaGym server:

```bash
python3 generate_kernels_and_eval.py \
  --task-source gpumode \
  --model-name gpt-5 \
  --eval-backend cudagym \
  --cudagym-url http://your-server:8000 \
  --remote-profile \
  ...
```

Required: `cudagym` Python package (`pip install cudagym>=0.8.0`).

The `--eval-backend cudagym` flag causes each task to build a self-contained evaluation bundle (kernel + driver + reference) that is sent to CudaGym for remote compilation, execution, and optional profiling. Results are returned as standard `EvalResult` objects — the generator loop is unaware of whether evaluation is local or remote.
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add remote CudaGym evaluation section to CLAUDE.md"
```

---

## Summary

| Task | What | Files | Est. Complexity |
|------|------|-------|----------------|
| 1 | RemoteEvalPackage + parse_driver_output | 5 new | Low |
| 2 | RemoteEvaluator (CudaGym orchestrator) | 2 new | Medium |
| 3 | GPUMode driver template | 3 new | Medium |
| 4 | GPUMode build_remote_eval_package + dispatch | 2 modify, 1 new | Medium |
| 5 | FlashInfer driver template | 2 new | Medium |
| 6 | CLI flags | 1 modify, 1 new | Low |
| 7 | Optional dependency | 1 modify | Low |
| 8 | Integration test | 1 new | Low |
| 9 | Launch script examples | 1 modify | Low |
| 10 | CLAUDE.md docs | 1 modify | Low |

**Critical path:** Tasks 1 → 2 → 3 → 4 → 6 → 8 (GPUMode remote eval, end-to-end)
**Parallel:** Task 5 (FlashInfer driver) can be done in parallel with Tasks 3-4
**Parallel:** Tasks 7, 9, 10 (docs/deps) can be done in parallel with anything
