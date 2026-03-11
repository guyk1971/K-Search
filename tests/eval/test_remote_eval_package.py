"""Tests for RemoteEvalPackage and parse_driver_output."""

import json
import pytest

from k_search.eval.remote_eval_package import RemoteEvalPackage, parse_driver_output


# ---------------------------------------------------------------------------
# RemoteEvalPackage construction tests
# ---------------------------------------------------------------------------

class TestRemoteEvalPackage:
    def test_create_package_cuda(self):
        pkg = RemoteEvalPackage(
            files={"kernel.cu": "// cuda code", "kernel.h": "// header", "main.cpp": "// main"},
            run_command="./main",
            compile_command="nvcc -o main kernel.cu main.cpp",
            artifact_names=["main"],
            language="cuda",
            env_vars={"CUDA_VISIBLE_DEVICES": "0"},
        )
        assert pkg.language == "cuda"
        assert pkg.compile_command == "nvcc -o main kernel.cu main.cpp"
        assert len(pkg.files) == 3
        assert pkg.env_vars["CUDA_VISIBLE_DEVICES"] == "0"

    def test_create_package_triton_no_compile(self):
        pkg = RemoteEvalPackage(
            files={"solution.py": "import triton"},
            run_command="python solution.py",
            compile_command=None,
            artifact_names=[],
            language="triton",
            env_vars={},
        )
        assert pkg.compile_command is None
        assert pkg.artifact_names == []
        assert pkg.language == "triton"

    def test_cudagym_env_type(self):
        cuda_pkg = RemoteEvalPackage(
            files={}, run_command="./main", compile_command=None,
            artifact_names=[], language="cuda", env_vars={},
        )
        assert cuda_pkg.cudagym_env_type == "cudacpp"

        triton_pkg = RemoteEvalPackage(
            files={}, run_command="python sol.py", compile_command=None,
            artifact_names=[], language="triton", env_vars={},
        )
        assert triton_pkg.cudagym_env_type == "python"

        python_pkg = RemoteEvalPackage(
            files={}, run_command="python sol.py", compile_command=None,
            artifact_names=[], language="python", env_vars={},
        )
        assert python_pkg.cudagym_env_type == "python"


# ---------------------------------------------------------------------------
# parse_driver_output tests
# ---------------------------------------------------------------------------

class TestParseDriverOutput:
    def test_parse_passed(self):
        data = {
            "status": "passed",
            "latency_ms": 1.23,
            "reference_latency_ms": 2.0,
            "speedup_factor": 1.63,
        }
        stdout = json.dumps(data)
        result = parse_driver_output(stdout)
        assert result.status == "passed"
        assert result.latency_ms == pytest.approx(1.23)
        assert result.reference_latency_ms == pytest.approx(2.0)
        assert result.speedup_factor == pytest.approx(1.63)

    def test_parse_failed(self):
        data = {
            "status": "runtime_error",
            "log_excerpt": "segfault at 0xdead",
        }
        stdout = json.dumps(data)
        result = parse_driver_output(stdout)
        assert result.status == "runtime_error"
        assert result.latency_ms is None
        assert "segfault" in result.log_excerpt

    def test_parse_with_profiling(self):
        data = {
            "status": "passed",
            "latency_ms": 0.5,
            "profiling": {"ncu_raw": "some profiling data"},
        }
        stdout = json.dumps(data)
        result = parse_driver_output(stdout)
        assert result.status == "passed"
        assert result.metrics["profiling"] == {"ncu_raw": "some profiling data"}

    def test_parse_malformed_json(self):
        stdout = "this is not json at all"
        result = parse_driver_output(stdout)
        assert result.status == "runtime_error"
        assert "Failed to parse" in result.log_excerpt

    def test_parse_extracts_json_from_mixed_output(self):
        data = {
            "status": "passed",
            "latency_ms": 3.14,
        }
        marker_start = "===KSEARCH_RESULT_JSON_START==="
        marker_end = "===KSEARCH_RESULT_JSON_END==="
        stdout = (
            "Loading model...\n"
            "Warming up GPU...\n"
            f"{marker_start}\n"
            f"{json.dumps(data)}\n"
            f"{marker_end}\n"
            "Cleanup done.\n"
        )
        result = parse_driver_output(stdout)
        assert result.status == "passed"
        assert result.latency_ms == pytest.approx(3.14)

    def test_parse_fallback_to_last_json_line(self):
        """When no markers, parser should find the last JSON line."""
        data = {"status": "passed", "latency_ms": 2.71}
        stdout = (
            "some log output\n"
            "more logs\n"
            f"{json.dumps(data)}\n"
        )
        result = parse_driver_output(stdout)
        assert result.status == "passed"
        assert result.latency_ms == pytest.approx(2.71)
