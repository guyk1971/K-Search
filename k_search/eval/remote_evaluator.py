"""RemoteEvaluator — orchestrates compile/execute/profile via CudaGym."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from k_search.eval.remote_eval_package import RemoteEvalPackage, parse_driver_output
from k_search.tasks.task_base import EvalResult

logger = logging.getLogger(__name__)

# Lazy imports — CudaGym may not be installed.  The names are populated on
# first use inside ``_evaluate_async`` so they can be patched in tests via
# ``patch("k_search.eval.remote_evaluator.CudaGymClient")`` etc.
CudaGymClient: Any = None
CompilationRequest: Any = None
ExecutionRequest: Any = None
ProfilingRequest: Any = None


def _ensure_cudagym_imports() -> None:
    """Import CudaGym symbols into module globals (once)."""
    global CudaGymClient, CompilationRequest, ExecutionRequest, ProfilingRequest  # noqa: PLW0603
    if CudaGymClient is not None:
        return
    from cudagym import CudaGymClient as _Client
    from cudagym.servers.compile.internal_api import CompilationRequest as _CompReq
    from cudagym.servers.gpu.internal_api import ExecutionRequest as _ExecReq
    from cudagym.servers.gpu.internal_api import ProfilingRequest as _ProfReq
    CudaGymClient = _Client
    CompilationRequest = _CompReq
    ExecutionRequest = _ExecReq
    ProfilingRequest = _ProfReq


class RemoteEvaluator:
    """Sends a RemoteEvalPackage to a CudaGym server for evaluation.

    Orchestrates up to three steps:
      1. Compilation (CUDA only — skipped when ``package.compile_command`` is None)
      2. Execution
      3. Profiling (optional, only when the execution passes)

    CudaGym is imported lazily so the rest of K-Search can run without it
    installed.
    """

    def __init__(
        self,
        server_url: str,
        enable_profiling: bool = True,
        compile_timeout: float = 120.0,
        execute_timeout: float = 300.0,
        profile_timeout: float = 150.0,
    ) -> None:
        self.server_url = server_url
        self.enable_profiling = enable_profiling
        self.compile_timeout = compile_timeout
        self.execute_timeout = execute_timeout
        self.profile_timeout = profile_timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, package: RemoteEvalPackage) -> EvalResult:
        """Synchronous wrapper around :meth:`_evaluate_async`."""
        return asyncio.run(self._evaluate_async(package))

    # ------------------------------------------------------------------
    # Async implementation
    # ------------------------------------------------------------------

    async def _evaluate_async(self, package: RemoteEvalPackage) -> EvalResult:
        """Full compile -> execute -> profile pipeline."""
        _ensure_cudagym_imports()

        try:
            async with CudaGymClient(self.server_url) as client:
                compile_resp = None

                # --- Step 1: Compile (CUDA) --------------------------------
                if package.compile_command is not None:
                    compile_req = CompilationRequest(
                        job_name="ksearch_compile",
                        file_contents=package.files,
                        commands=[package.compile_command],
                        artifact_names=package.artifact_names,
                        return_base64=True,
                        timeout=self.compile_timeout,
                    )
                    compile_resp = await client.compile(compile_req)

                    if not compile_resp.success:
                        stderr = _extract_compile_errors(compile_resp)
                        logger.info("Compilation failed: %s", stderr[:200])
                        return EvalResult(
                            status="compile_error",
                            log_excerpt=stderr,
                        )

                # --- Step 2: Execute ----------------------------------------
                exec_req = ExecutionRequest(
                    job_name="ksearch_execute",
                    command=package.run_command,
                    file_contents=package.files,
                    binary_base64=(
                        compile_resp.output_base64 if compile_resp else {}
                    ),
                    env_vars=package.env_vars,
                    n_runs=1,
                    timeout_per_run=self.execute_timeout,
                )
                exec_resp = await client.execute(exec_req)

                if not exec_resp.successes or not exec_resp.successes[0]:
                    stderr = exec_resp.stderrs[0] if exec_resp.stderrs else ""
                    stdout = exec_resp.stdouts[0] if exec_resp.stdouts else ""
                    log = (stderr or stdout or exec_resp.exception or
                           "Execution failed with no output")
                    logger.info("Execution failed: %s", log[:200])
                    return EvalResult(
                        status="runtime_error",
                        log_excerpt=log[:4000],
                    )

                # --- Step 3: Parse output -----------------------------------
                stdout = exec_resp.stdouts[0] if exec_resp.stdouts else ""
                result = parse_driver_output(stdout)

                # --- Step 4: Profile (optional) -----------------------------
                if self.enable_profiling and result.is_passed():
                    try:
                        profile_req = ProfilingRequest(
                            job_name="ksearch_profile",
                            command=package.run_command,
                            file_contents=package.files,
                            binary_base64=(
                                compile_resp.output_base64 if compile_resp else {}
                            ),
                            env_vars=package.env_vars,
                            enable_ncu=True,
                            enable_nsys=True,
                            timeout=self.profile_timeout,
                        )
                        profile_resp = await client.profile(profile_req)
                        profiling_data = _extract_profiling_data(profile_resp)
                        if profiling_data:
                            result.metrics["profiling"] = profiling_data
                    except Exception:
                        logger.warning("Profiling step failed; skipping.", exc_info=True)

                return result

        except ImportError:
            raise
        except ConnectionError as exc:
            logger.error("CudaGym server unreachable at %s: %s", self.server_url, exc)
            return EvalResult(
                status="runtime_error",
                log_excerpt=f"CudaGym server unreachable: {exc}",
            )
        except Exception as exc:
            logger.error("RemoteEvaluator error: %s", exc, exc_info=True)
            return EvalResult(
                status="runtime_error",
                log_excerpt=f"RemoteEvaluator error: {exc}",
            )


def _extract_compile_errors(compile_resp) -> str:
    """Extract stderr/stdout from CompilationResult.details."""
    parts = []
    for detail in getattr(compile_resp, "details", []):
        if getattr(detail, "stderr", ""):
            parts.append(detail.stderr)
        if getattr(detail, "stdout", ""):
            parts.append(detail.stdout)
    if parts:
        return "\n".join(parts)[:4000]
    return getattr(compile_resp, "exception", "") or "Unknown compile error"


def _extract_profiling_data(profile_resp) -> dict | None:
    """Extract NCU/nsys data from ProfilingResult into a flat dict."""
    data = {}
    if getattr(profile_resp, "ncu_success", False):
        data["ncu"] = {
            "raw_logs": getattr(profile_resp, "ncu_raw_logs", "") or "",
            "metrics": getattr(profile_resp, "ncu_json_data", {}) or {},
            "cycles": getattr(profile_resp, "ncu_cycles", None),
            "duration_us": getattr(profile_resp, "ncu_duration_us", None),
        }
    if getattr(profile_resp, "nsys_success", False):
        data["nsys"] = {
            "raw_logs": getattr(profile_resp, "nsys_raw_logs", "") or "",
            "kernel_summary": getattr(profile_resp, "nsys_kernel_summary", "") or "",
            "nvtx_summary": getattr(profile_resp, "nsys_nvtx_summary", "") or "",
            "api_summary": getattr(profile_resp, "nsys_api_summary", "") or "",
        }
    return data if data else None
