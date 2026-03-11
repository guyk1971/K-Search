"""Tests for the FlashInfer remote evaluation driver template."""

import json

from k_search.eval.drivers.flashinfer_driver import build_flashinfer_driver_source


_DUMMY_DEFINITION = json.dumps({"name": "test_kernel", "inputs": [], "outputs": []})
_DUMMY_WORKLOADS = json.dumps([{"batch_size": 1}])
_ENTRY_POINT = "kernel.py::run"


def _build(**overrides):
    kwargs = {
        "definition_json": _DUMMY_DEFINITION,
        "workloads_json": _DUMMY_WORKLOADS,
        "solution_entry_point": _ENTRY_POINT,
    }
    kwargs.update(overrides)
    return build_flashinfer_driver_source(**kwargs)


class TestFlashinferDriver:
    def test_returns_valid_python(self):
        source = _build()
        # Should compile without syntax errors
        compile(source, "<driver>", "exec")

    def test_contains_json_markers(self):
        source = _build()
        assert "===KSEARCH_RESULT_JSON_START===" in source
        assert "===KSEARCH_RESULT_JSON_END===" in source

    def test_contains_eval_harness(self):
        source = _build()
        assert "torch.cuda.Event" in source
        assert "elapsed_time" in source

    def test_embeds_definition_and_workloads(self):
        source = _build()
        assert "test_kernel" in source
        assert "batch_size" in source

    def test_custom_eval_config(self):
        source = _build(eval_config={"rtol": 1e-5, "atol": 1e-5, "warmup_runs": 5, "iterations": 20})
        assert "1e-05" in source
        assert "_WARMUP_RUNS = 5" in source
        assert "_ITERATIONS = 20" in source

    def test_entry_point_without_double_colon(self):
        source = _build(solution_entry_point="my_kernel.py")
        assert "my_kernel" in source
        # Default function name should be "run"
        assert "'run'" in source

    def test_entry_point_with_custom_function(self):
        source = _build(solution_entry_point="solver.py::solve")
        assert "'solver'" in source or '"solver"' in source
        assert "'solve'" in source or '"solve"' in source
