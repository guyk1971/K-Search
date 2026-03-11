"""RemoteEvalPackage dataclass and driver output parsing utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from k_search.tasks.task_base import EvalResult

# ---------------------------------------------------------------------------
# JSON marker constants used by the remote driver to delimit the result blob
# ---------------------------------------------------------------------------
_JSON_MARKER_START = "===KSEARCH_RESULT_JSON_START==="
_JSON_MARKER_END = "===KSEARCH_RESULT_JSON_END==="


# ---------------------------------------------------------------------------
# RemoteEvalPackage
# ---------------------------------------------------------------------------

@dataclass
class RemoteEvalPackage:
    """Everything needed to evaluate a kernel on a remote CudaGym worker."""

    files: dict[str, str]
    run_command: str
    compile_command: str | None
    artifact_names: list[str]
    language: str
    env_vars: dict[str, str] = field(default_factory=dict)

    @property
    def cudagym_env_type(self) -> str:
        """Map language to CudaGym environment type."""
        if self.language == "cuda":
            return "cudacpp"
        return "python"


# ---------------------------------------------------------------------------
# Driver output parsing
# ---------------------------------------------------------------------------

def _extract_json_block(stdout: str) -> str | None:
    """Extract the JSON string from driver stdout.

    First tries to find content between marker lines. Falls back to the last
    line that looks like a JSON object.
    """
    # Strategy 1: markers
    if _JSON_MARKER_START in stdout and _JSON_MARKER_END in stdout:
        start_idx = stdout.index(_JSON_MARKER_START) + len(_JSON_MARKER_START)
        end_idx = stdout.index(_JSON_MARKER_END)
        candidate = stdout[start_idx:end_idx].strip()
        if candidate:
            return candidate

    # Strategy 2: last JSON-like line (scanning from the end)
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped

    return None


def parse_driver_output(stdout: str) -> EvalResult:
    """Parse remote driver stdout into an EvalResult."""
    json_str = _extract_json_block(stdout)
    if json_str is None:
        return EvalResult(
            status="runtime_error",
            log_excerpt=f"Failed to parse driver output: no JSON found in stdout ({len(stdout)} chars)",
        )

    try:
        data: dict[str, Any] = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return EvalResult(
            status="runtime_error",
            log_excerpt=f"Failed to parse driver output: {exc}",
        )

    metrics: dict[str, Any] = {}
    if "profiling" in data:
        metrics["profiling"] = data["profiling"]

    return EvalResult(
        status=data.get("status", "runtime_error"),
        latency_ms=data.get("latency_ms"),
        reference_latency_ms=data.get("reference_latency_ms"),
        mean_vs_baseline_factor=data.get("mean_vs_baseline_factor"),
        speedup_factor=data.get("speedup_factor"),
        log_excerpt=data.get("log_excerpt", ""),
        metrics=metrics,
    )
