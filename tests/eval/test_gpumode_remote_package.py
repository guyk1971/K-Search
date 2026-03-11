"""Tests for GpuModeTriMulTask.build_remote_eval_package."""

from __future__ import annotations

import pytest

from k_search.tasks.task_base import (
    BuildSpec,
    Solution,
    SourceFile,
    SupportedLanguages,
)
from k_search.tasks.gpu_mode_task import GpuModeTriMulTask


def _make_triton_solution() -> Solution:
    return Solution(
        name="test",
        definition="gpumode_trimul",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.TRITON,
            target_hardware=["H100"],
            entry_point="submission.py::custom_kernel",
        ),
        sources=[
            SourceFile(
                path="submission.py",
                content="def custom_kernel(data):\n    return data[0]\n",
            ),
        ],
    )


def _make_cuda_solution() -> Solution:
    return Solution(
        name="test",
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


def _make_task() -> GpuModeTriMulTask:
    return GpuModeTriMulTask()


class TestTritonPackageStructure:
    def test_triton_package_structure(self) -> None:
        task = _make_task()
        solution = _make_triton_solution()
        package = task.build_remote_eval_package(solution=solution)

        assert "submission.py" in package.files
        assert "driver.py" in package.files
        assert package.language == "triton"
        assert package.compile_command is None
        assert package.run_command == "python3 driver.py"


class TestCudaPackageStructure:
    def test_cuda_package_structure(self) -> None:
        task = _make_task()
        solution = _make_cuda_solution()
        package = task.build_remote_eval_package(solution=solution)

        assert "submission.py" in package.files
        assert "driver.py" in package.files
        assert package.language == "cuda"
        assert package.compile_command is None


class TestPackageContainsSpecs:
    def test_package_contains_test_and_benchmark_specs(self) -> None:
        task = _make_task()
        solution = _make_triton_solution()
        package = task.build_remote_eval_package(solution=solution)

        driver_source = package.files["driver.py"]
        assert "TEST_SPECS" in driver_source
        assert "BENCHMARK_SPECS" in driver_source
