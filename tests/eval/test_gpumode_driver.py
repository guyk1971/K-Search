"""Tests for the GPUMode TriMul self-contained driver template."""

import pytest

from k_search.eval.drivers.gpumode_trimul_driver import (
    RESULT_JSON_END,
    RESULT_JSON_START,
    build_gpumode_driver_source,
)

# Sample specs used across tests.
SAMPLE_TEST_SPECS = [
    {
        "seqlen": 64,
        "bs": 2,
        "dim": 32,
        "hiddendim": 16,
        "seed": 42,
        "nomask": False,
        "distribution": "normal",
    },
]

SAMPLE_BENCHMARK_SPECS = [
    {
        "seqlen": 128,
        "bs": 4,
        "dim": 64,
        "hiddendim": 32,
        "seed": 123,
        "nomask": True,
        "distribution": "normal",
    },
]


@pytest.fixture
def driver_source() -> str:
    return build_gpumode_driver_source(
        test_specs=SAMPLE_TEST_SPECS,
        benchmark_specs=SAMPLE_BENCHMARK_SPECS,
    )


class TestBuildGpuModeDriverSource:
    def test_returns_valid_python(self, driver_source: str):
        """The generated source must be syntactically valid Python."""
        compile(driver_source, "<driver>", "exec")

    def test_contains_json_markers(self, driver_source: str):
        """Driver must print result between the expected markers."""
        assert RESULT_JSON_START in driver_source
        assert RESULT_JSON_END in driver_source

    def test_contains_reference_implementation(self, driver_source: str):
        """Driver must embed the TriMul reference components."""
        assert "class TriMul" in driver_source
        assert "def ref_kernel" in driver_source
        assert "def generate_input" in driver_source

    def test_contains_benchmark_harness(self, driver_source: str):
        """Driver must contain CUDA event-based timing."""
        assert "torch.cuda.Event" in driver_source
        assert "elapsed_time" in driver_source

    def test_embeds_test_specs(self, driver_source: str):
        """Spec values from test_specs must appear in the source."""
        # Check a few distinctive values from the sample specs.
        assert '"seqlen": 64' in driver_source or '"seqlen":64' in driver_source
        assert '"bs": 2' in driver_source or '"bs":2' in driver_source
        assert '"dim": 32' in driver_source or '"dim":32' in driver_source

    def test_embeds_benchmark_specs(self, driver_source: str):
        """Spec values from benchmark_specs must appear in the source."""
        assert '"seqlen": 128' in driver_source or '"seqlen":128' in driver_source
        assert '"bs": 4' in driver_source or '"bs":4' in driver_source

    def test_contains_correctness_checker(self, driver_source: str):
        """Driver must embed verbose_allclose and DisableCuDNNTF32."""
        assert "verbose_allclose" in driver_source
        assert "DisableCuDNNTF32" in driver_source

    def test_contains_submission_import(self, driver_source: str):
        """Driver must import custom_kernel from submission."""
        assert "from submission import custom_kernel" in driver_source

    def test_contains_main_guard(self, driver_source: str):
        """Driver must have an if __name__ == '__main__' block."""
        assert '__name__ == "__main__"' in driver_source or "__name__ == '__main__'" in driver_source
