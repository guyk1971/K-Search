"""End-to-end integration test: Solution -> build_remote_eval_package -> mock CudaGym -> EvalResult.

No real GPU or CudaGym server needed — CudaGym is mocked entirely.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from k_search.eval.remote_eval_package import _JSON_MARKER_START, _JSON_MARKER_END
from k_search.eval.remote_evaluator import RemoteEvaluator
from k_search.tasks.task_base import BuildSpec, EvalResult, Solution, SourceFile, SupportedLanguages
from k_search.tasks.gpu_mode_task import GpuModeTriMulTask


def _mock_cudagym_success(latency_ms=1.5):
    """Return a mock CudaGym async client whose execute() returns a successful result."""
    result_json = json.dumps({
        "status": "passed",
        "latency_ms": latency_ms,
        "reference_latency_ms": 2.0,
        "metrics": {"score_name": "inv_latency_ms", "score": 1.0 / latency_ms},
        "log_excerpt": "",
    })
    stdout = f"Loading...\n{_JSON_MARKER_START}\n{result_json}\n{_JSON_MARKER_END}\n"

    # The RemoteEvaluator reads exec_resp via getattr:
    #   getattr(exec_resp, "successes", [])  -> [True]
    #   getattr(exec_resp, "stdout", "")     -> the full stdout string
    mock_exec = MagicMock()
    mock_exec.successes = [True]
    mock_exec.stdout = stdout
    mock_exec.stderr = ""
    mock_exec.exception = ""

    client = AsyncMock()
    client.execute = AsyncMock(return_value=mock_exec)
    client.profile = AsyncMock(return_value=MagicMock(
        ncu_success=False, nsys_success=False, exception="",
    ))
    return client


class TestEndToEndRemoteEval:
    def test_gpumode_triton_remote_eval_flow(self):
        """Full: create task -> build package -> mock evaluate -> get EvalResult."""
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

        evaluator = RemoteEvaluator(server_url="http://mock:8000", enable_profiling=False)
        task = GpuModeTriMulTask(mode="benchmark", eval_backend="cudagym", remote_evaluator=evaluator)

        # Build package
        pkg = task.build_remote_eval_package(solution=sol)
        assert "submission.py" in pkg.files
        assert "driver.py" in pkg.files

        # Mock evaluate via run_benchmark dispatch.
        # We need to:
        #   1. Bypass _ensure_cudagym_imports (cudagym is not installed)
        #   2. Mock the CudaGymClient async context manager
        #   3. Mock ExecutionRequest (called inside _evaluate_async)
        mock_client = _mock_cudagym_success(latency_ms=1.23)

        # Create a mock CudaGymClient class that works as an async context manager
        mock_client_cls = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client_instance

        with patch("k_search.eval.remote_evaluator._ensure_cudagym_imports"), \
             patch("k_search.eval.remote_evaluator.CudaGymClient", mock_client_cls), \
             patch("k_search.eval.remote_evaluator.ExecutionRequest", MagicMock()):
            result = task.run_benchmark(solution=sol)

        assert isinstance(result, EvalResult)
        assert result.status == "passed"
        assert result.latency_ms == 1.23
        assert result.score() > 0
