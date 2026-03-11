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
                binary_base64: str | None = None

                # --- Step 1: Compile (CUDA) --------------------------------
                if package.compile_command is not None:
                    compile_req = CompilationRequest(
                        files=package.files,
                        compile_command=package.compile_command,
                        artifact_names=package.artifact_names,
                        return_base64=True,
                        timeout=self.compile_timeout,
                    )
                    compile_resp = await client.compile(compile_req)

                    if not compile_resp.success:
                        stderr = getattr(compile_resp, "stderr", "") or ""
                        logger.info("Compilation failed: %s", stderr[:200])
                        return EvalResult(
                            status="compile_error",
                            log_excerpt=stderr,
                        )

                    binary_base64 = getattr(compile_resp, "binary_base64", None)

                # --- Step 2: Execute ----------------------------------------
                exec_kwargs: dict[str, Any] = dict(
                    files=package.files,
                    run_command=package.run_command,
                    env_type=package.cudagym_env_type,
                    env_vars=package.env_vars,
                    timeout=self.execute_timeout,
                )
                if binary_base64 is not None:
                    exec_kwargs["binary_base64"] = binary_base64

                exec_req = ExecutionRequest(**exec_kwargs)
                exec_resp = await client.execute(exec_req)

                successes = getattr(exec_resp, "successes", [])
                if not successes or not successes[0]:
                    stderr = getattr(exec_resp, "stderr", "") or ""
                    logger.info("Execution failed: %s", stderr[:200])
                    return EvalResult(
                        status="runtime_error",
                        log_excerpt=stderr,
                    )

                # --- Step 3: Parse output -----------------------------------
                stdout = getattr(exec_resp, "stdout", "") or ""
                result = parse_driver_output(stdout)

                # --- Step 4: Profile (optional) -----------------------------
                if self.enable_profiling and result.is_passed():
                    try:
                        profile_req = ProfilingRequest(
                            files=package.files,
                            run_command=package.run_command,
                            env_type=package.cudagym_env_type,
                            env_vars=package.env_vars,
                            timeout=self.profile_timeout,
                        )
                        if binary_base64 is not None:
                            profile_req.binary_base64 = binary_base64

                        profile_resp = await client.profile(profile_req)
                        profiling_data = getattr(profile_resp, "profiling", None)
                        if profiling_data is not None:
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
