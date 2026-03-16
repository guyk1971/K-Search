"""Tests for RemoteEvaluator — all CudaGym interactions are mocked."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from k_search.eval.remote_eval_package import (
    RemoteEvalPackage,
    _JSON_MARKER_END,
    _JSON_MARKER_START,
)
from k_search.eval.remote_evaluator import RemoteEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stdout(result_dict: dict) -> str:
    """Build a driver stdout string with JSON markers."""
    payload = json.dumps(result_dict)
    return (
        f"Loading...\n"
        f"{_JSON_MARKER_START}\n"
        f"{payload}\n"
        f"{_JSON_MARKER_END}\n"
    )


def _triton_package() -> RemoteEvalPackage:
    return RemoteEvalPackage(
        files={"solution.py": "import triton"},
        run_command="python solution.py",
        compile_command=None,
        artifact_names=[],
        language="triton",
    )


def _cuda_package() -> RemoteEvalPackage:
    return RemoteEvalPackage(
        files={
            "kernel.cu": "// cuda",
            "kernel.h": "// header",
            "main.cpp": "// main",
        },
        run_command="./main",
        compile_command="nvcc -o main kernel.cu main.cpp",
        artifact_names=["main"],
        language="cuda",
    )


def _build_mock_client(
    compile_resp=None,
    exec_resp=None,
    profile_resp=None,
    enter_side_effect=None,
):
    """Create a mock CudaGymClient class whose instances are async context managers."""
    instance = AsyncMock()

    if compile_resp is not None:
        instance.compile.return_value = compile_resp
    if exec_resp is not None:
        instance.execute.return_value = exec_resp
    if profile_resp is not None:
        instance.profile.return_value = profile_resp

    if enter_side_effect is not None:
        instance.__aenter__ = AsyncMock(side_effect=enter_side_effect)
    else:
        instance.__aenter__ = AsyncMock(return_value=instance)
    instance.__aexit__ = AsyncMock(return_value=False)

    client_cls = MagicMock(return_value=instance)
    return client_cls, instance


# We patch all four CudaGym symbols at module level.  ``_ensure_cudagym_imports``
# sees a non-None CudaGymClient and short-circuits, so the mocks take effect.
_PATCH_PREFIX = "k_search.eval.remote_evaluator"


def _patches(client_cls):
    """Return a list of ``patch`` context managers for the four CudaGym symbols."""
    return [
        patch(f"{_PATCH_PREFIX}.CudaGymClient", client_cls),
        patch(f"{_PATCH_PREFIX}.CompilationRequest", MagicMock()),
        patch(f"{_PATCH_PREFIX}.ExecutionRequest", MagicMock()),
        patch(f"{_PATCH_PREFIX}.ProfilingRequest", MagicMock()),
    ]


def _apply_patches(patches):
    """Enter a list of patch context managers, return list of mocks."""
    return [p.__enter__() for p in patches]


def _stop_patches(patches):
    for p in patches:
        p.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Mock response factories matching real CudaGym API field names
# ---------------------------------------------------------------------------

def _exec_success(stdout_str: str):
    """ExecutionResult: successes (plural), stdouts (plural), stderrs (plural)."""
    return SimpleNamespace(
        successes=[True],
        stdouts=[stdout_str],
        stderrs=[""],
        exception="",
    )


def _exec_failure(stderr_str: str = "Segfault"):
    return SimpleNamespace(
        successes=[False],
        stdouts=[""],
        stderrs=[stderr_str],
        exception="",
    )


def _compile_success(output_base64: dict | None = None):
    """CompilationResult: success (singular), output_base64, details."""
    return SimpleNamespace(
        success=True,
        output_base64=output_base64 or {},
        details=[],
        exception="",
    )


def _compile_failure(stderr_str: str = "error: undeclared identifier"):
    return SimpleNamespace(
        success=False,
        output_base64={},
        details=[SimpleNamespace(stderr=stderr_str, stdout="")],
        exception="",
    )


def _profile_success(ncu=True, nsys=True):
    """ProfilingResult with real field names."""
    return SimpleNamespace(
        ncu_success=ncu,
        ncu_raw_logs="Speed of Light: 50%" if ncu else None,
        ncu_json_data={"sm_occupancy": 0.5} if ncu else None,
        ncu_cycles=1000 if ncu else None,
        ncu_duration_us=100.0 if ncu else None,
        nsys_success=nsys,
        nsys_raw_logs="kernel summary" if nsys else None,
        nsys_kernel_summary="kernel details" if nsys else None,
        nsys_nvtx_summary=None,
        nsys_api_summary=None,
        exception="",
    )


# ---------------------------------------------------------------------------
# Triton tests (no compilation step)
# ---------------------------------------------------------------------------

class TestRemoteEvaluatorTriton:
    def test_evaluate_triton_success(self):
        stdout = _make_stdout({
            "status": "passed",
            "latency_ms": 1.23,
            "speedup_factor": 2.5,
        })
        client_cls, client_inst = _build_mock_client(
            exec_resp=_exec_success(stdout),
        )

        patches = _patches(client_cls)
        _apply_patches(patches)
        try:
            evaluator = RemoteEvaluator("http://localhost:8080", enable_profiling=False)
            result = evaluator.evaluate(_triton_package())

            assert result.is_passed()
            assert result.latency_ms == 1.23
            assert result.speedup_factor == 2.5
            # compile should NOT have been called
            client_inst.compile.assert_not_awaited()
        finally:
            _stop_patches(patches)

    def test_evaluate_triton_execution_failure(self):
        client_cls, _ = _build_mock_client(
            exec_resp=_exec_failure("Segfault"),
        )

        patches = _patches(client_cls)
        _apply_patches(patches)
        try:
            evaluator = RemoteEvaluator("http://localhost:8080", enable_profiling=False)
            result = evaluator.evaluate(_triton_package())

            assert result.status == "runtime_error"
            assert "Segfault" in result.log_excerpt
        finally:
            _stop_patches(patches)


# ---------------------------------------------------------------------------
# CUDA tests (compile + execute)
# ---------------------------------------------------------------------------

class TestRemoteEvaluatorCuda:
    def test_evaluate_cuda_success(self):
        stdout = _make_stdout({
            "status": "passed",
            "latency_ms": 0.5,
            "reference_latency_ms": 1.0,
            "speedup_factor": 2.0,
        })
        client_cls, client_inst = _build_mock_client(
            compile_resp=_compile_success({"main": "AAAA=="}),
            exec_resp=_exec_success(stdout),
        )

        patches = _patches(client_cls)
        _apply_patches(patches)
        try:
            evaluator = RemoteEvaluator("http://localhost:8080", enable_profiling=False)
            result = evaluator.evaluate(_cuda_package())

            assert result.is_passed()
            assert result.latency_ms == 0.5
            assert result.speedup_factor == 2.0
            client_inst.compile.assert_awaited_once()
        finally:
            _stop_patches(patches)

    def test_evaluate_cuda_compile_failure(self):
        client_cls, client_inst = _build_mock_client(
            compile_resp=_compile_failure("error: undeclared identifier"),
        )

        patches = _patches(client_cls)
        _apply_patches(patches)
        try:
            evaluator = RemoteEvaluator("http://localhost:8080", enable_profiling=False)
            result = evaluator.evaluate(_cuda_package())

            assert result.status == "compile_error"
            assert "undeclared identifier" in result.log_excerpt
            client_inst.execute.assert_not_awaited()
        finally:
            _stop_patches(patches)


# ---------------------------------------------------------------------------
# Profiling tests
# ---------------------------------------------------------------------------

class TestRemoteEvaluatorProfiling:
    def test_evaluate_with_profiling(self):
        stdout = _make_stdout({
            "status": "passed",
            "latency_ms": 1.0,
            "speedup_factor": 1.5,
        })
        client_cls, client_inst = _build_mock_client(
            exec_resp=_exec_success(stdout),
            profile_resp=_profile_success(ncu=True, nsys=True),
        )

        patches = _patches(client_cls)
        _apply_patches(patches)
        try:
            evaluator = RemoteEvaluator("http://localhost:8080", enable_profiling=True)
            result = evaluator.evaluate(_triton_package())

            assert result.is_passed()
            assert "profiling" in result.metrics
            assert "ncu" in result.metrics["profiling"]
            assert result.metrics["profiling"]["ncu"]["raw_logs"] == "Speed of Light: 50%"
            assert result.metrics["profiling"]["ncu"]["metrics"]["sm_occupancy"] == 0.5
            assert "nsys" in result.metrics["profiling"]
            assert result.metrics["profiling"]["nsys"]["kernel_summary"] == "kernel details"
            client_inst.profile.assert_awaited_once()
        finally:
            _stop_patches(patches)

    def test_profiling_skipped_on_failure(self):
        """Profiling should not be attempted when execution fails."""
        client_cls, client_inst = _build_mock_client(
            exec_resp=_exec_failure("OOM"),
        )

        patches = _patches(client_cls)
        _apply_patches(patches)
        try:
            evaluator = RemoteEvaluator("http://localhost:8080", enable_profiling=True)
            result = evaluator.evaluate(_triton_package())

            assert result.status == "runtime_error"
            client_inst.profile.assert_not_awaited()
        finally:
            _stop_patches(patches)


# ---------------------------------------------------------------------------
# Connection error tests
# ---------------------------------------------------------------------------

class TestRemoteEvaluatorConnectionError:
    def test_server_unreachable(self):
        client_cls, _ = _build_mock_client(
            enter_side_effect=ConnectionError("Connection refused"),
        )

        patches = _patches(client_cls)
        _apply_patches(patches)
        try:
            evaluator = RemoteEvaluator("http://localhost:9999", enable_profiling=False)
            result = evaluator.evaluate(_triton_package())

            assert result.status == "runtime_error"
            assert "Connection refused" in result.log_excerpt
        finally:
            _stop_patches(patches)
