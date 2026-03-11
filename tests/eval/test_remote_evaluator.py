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
# Triton tests (no compilation step)
# ---------------------------------------------------------------------------

class TestRemoteEvaluatorTriton:
    def test_evaluate_triton_success(self):
        stdout = _make_stdout({
            "status": "passed",
            "latency_ms": 1.23,
            "speedup_factor": 2.5,
        })
        exec_resp = SimpleNamespace(successes=[True], stdout=stdout, stderr="")
        client_cls, client_inst = _build_mock_client(exec_resp=exec_resp)

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
        exec_resp = SimpleNamespace(successes=[False], stdout="", stderr="Segfault")
        client_cls, _ = _build_mock_client(exec_resp=exec_resp)

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
        compile_resp = SimpleNamespace(success=True, binary_base64="AAAA==", stderr="")
        stdout = _make_stdout({
            "status": "passed",
            "latency_ms": 0.5,
            "reference_latency_ms": 1.0,
            "speedup_factor": 2.0,
        })
        exec_resp = SimpleNamespace(successes=[True], stdout=stdout, stderr="")
        client_cls, client_inst = _build_mock_client(
            compile_resp=compile_resp, exec_resp=exec_resp,
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
        compile_resp = SimpleNamespace(
            success=False, binary_base64=None, stderr="error: undeclared identifier",
        )
        client_cls, client_inst = _build_mock_client(compile_resp=compile_resp)

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
        exec_resp = SimpleNamespace(successes=[True], stdout=stdout, stderr="")
        profile_resp = SimpleNamespace(
            profiling={"kernel_time_ms": 0.8, "memory_bw_gb_s": 500.0},
        )
        client_cls, client_inst = _build_mock_client(
            exec_resp=exec_resp, profile_resp=profile_resp,
        )

        patches = _patches(client_cls)
        _apply_patches(patches)
        try:
            evaluator = RemoteEvaluator("http://localhost:8080", enable_profiling=True)
            result = evaluator.evaluate(_triton_package())

            assert result.is_passed()
            assert "profiling" in result.metrics
            assert result.metrics["profiling"]["kernel_time_ms"] == 0.8
            client_inst.profile.assert_awaited_once()
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
