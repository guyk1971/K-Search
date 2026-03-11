# Design: Remote CudaGym Evaluation Backend for K-Search

**Date:** 2026-03-11
**Status:** Approved

## Problem

K-Search currently assumes compilation and profiling run on the local machine. We need to support remote GPU evaluation via CudaGym to enable running K-Search on machines without GPUs, leveraging remote GPU clusters (NVIDIA Astra B200s or self-hosted).

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Remote service | CudaGym (not custom Docker) | Cleaner, more scalable, existing Python client |
| Task scope | Task-agnostic | Any task can support remote execution |
| Evaluation bundle | Fully self-contained | No task-specific libraries needed on remote; only PyTorch/Triton/CUDA toolchain assumed |
| Reference/correctness | Task decides | Each task ships its own reference impl or pre-computed data |
| Local vs remote | CLI flag `--eval-backend` | Explicit, consistent with existing `--backend` pattern |
| Deployment | Configurable (hosted or self-hosted) | Via `--cudagym-url` or `CUDAGYM_URL` env var |
| Profiling | CLI flag `--remote-profile` (default: enabled) | Profiling data helps world model; can disable for speed |
| Async handling | Sync wrapper around async CudaGym client | Minimal architectural change; async refactor deferred |

## Architecture

### Data Flow

```
Generator -> Task.run_benchmark(solution)
                    |
            +-------+--------+
            | eval_backend?   |
            +----+-------+----+
            |local|      |cudagym
            v            v
        (existing)   RemoteEvaluator
                        |
                   +----+-----+
                   | Task builds|
                   | eval bundle|
                   +----+------+
                        |
                CudaGymClient
                compile -> execute -> (profile)
                        |
                Parse stdout JSON -> EvalResult
```

### Key Components

#### 1. `RemoteEvalPackage` dataclass

Produced by each task via `build_remote_eval_package()`:

```python
@dataclasses.dataclass
class RemoteEvalPackage:
    files: dict[str, str]          # All source files including driver script
    compile_command: str | None     # Build command (None for Python/Triton)
    run_command: str                # Execution command
    artifact_names: list[str]      # Compiled binaries to transfer (empty for Python)
    language: str                   # "cuda" | "triton" | "python"
    env_vars: dict[str, str]       # Runtime environment variables
```

#### 2. Task Protocol Extension

One new method added to the `Task` protocol:

```python
def build_remote_eval_package(
    self,
    solution: Solution,
    config: Any = None,
    round_num: int = 0,
) -> RemoteEvalPackage:
    """Build a self-contained evaluation bundle for remote execution."""
```

Each task implements this to produce a standalone package that:
- Contains the kernel source files
- Contains a driver script that compiles, runs, benchmarks, checks correctness
- Outputs structured JSON to stdout
- Has no dependencies beyond PyTorch/Triton/CUDA toolchain

#### 3. `RemoteEvaluator` class

Orchestrates the CudaGym interaction:

```python
class RemoteEvaluator:
    def __init__(self, server_url: str, enable_profiling: bool = True):
        self.server_url = server_url
        self.enable_profiling = enable_profiling

    def evaluate(self, package: RemoteEvalPackage) -> EvalResult:
        """Sync wrapper: compile -> execute -> (profile) -> parse -> EvalResult."""
```

Internally:
1. Maps `RemoteEvalPackage` to CudaGym `CompilationRequest` (if compile_command is set)
2. Maps to `ExecutionRequest` with compiled artifacts + source files
3. If profiling enabled and execution succeeded: sends `ProfilingRequest`
4. Parses stdout JSON into `EvalResult`

#### 4. `run_benchmark()` Dispatch

In each task's `run_benchmark()`:

```python
def run_benchmark(self, solution, config=None, dump_traces=False, round_num=0):
    if self.eval_backend == "cudagym":
        package = self.build_remote_eval_package(solution, config, round_num)
        return self.remote_evaluator.evaluate(package)
    else:
        # existing local code path
        ...
```

#### 5. Stdout JSON Contract

All remote driver scripts must emit this to stdout:

```json
{
  "status": "passed|failed|compile_error|runtime_error",
  "latency_ms": 1.56,
  "reference_latency_ms": 2.34,
  "metrics": {
    "score_name": "inv_latency_ms",
    "score": 0.641
  },
  "log_excerpt": "error details if failed",
  "profiling": {
    "ncu": {
      "raw_logs": "...",
      "metrics": {}
    },
    "nsys": {
      "raw_logs": "...",
      "kernel_summary": {}
    }
  }
}
```

The `profiling` field is only populated when profiling is enabled and runs successfully. The `metrics` dict is task-specific.

#### 6. Self-Contained Driver Scripts

Each task provides a driver script template that gets bundled into the `RemoteEvalPackage`.

**GPUMode TriMul driver** (`gpumode_driver.py`):
- Compiles CUDA via subprocess (nvcc) or loads Triton kernel via Python import
- Runs benchmark: torch.cuda.Event timing with warmup + iterations
- Correctness check: compares against PyTorch reference (`torch.matmul` or task-specific)
- Outputs JSON to stdout

**FlashInfer driver** (`flashinfer_driver.py`):
- Loads workload definition (embedded in driver or as companion JSON file)
- Runs kernel across workloads with timing
- Correctness check against embedded PyTorch reference
- Aggregates results and outputs JSON to stdout

### CLI Changes

Added to `generate_kernels_and_eval.py`:

```
--eval-backend {local,cudagym}   # Default: local
--cudagym-url URL                # Default: env CUDAGYM_URL
--remote-profile / --no-remote-profile  # Default: enabled
```

### File Layout

```
k_search/
  eval/
    __init__.py
    remote_eval_package.py          # RemoteEvalPackage dataclass
    remote_evaluator.py             # CudaGymClient orchestrator + JSON parsing
    drivers/
      __init__.py
      gpumode_driver_template.py    # Self-contained GPUMode eval driver
      flashinfer_driver_template.py # Self-contained FlashInfer eval driver
```

### Dependencies

- `cudagym` Python package (for `CudaGymClient`) — added as optional dependency
- No changes to existing dependencies for local execution

### CudaGym Environment Mapping

| K-Search language | CudaGym env | Notes |
|-------------------|-------------|-------|
| CUDA | `cudacpp` | CMake build with kernel.cu, kernel.h, main.cpp |
| Triton | `python` | Python DSL, JIT compilation on GPU server |
| Python | `python` | Direct Python execution |

### Error Handling

- CudaGym server unreachable: `EvalResult(status="runtime_error", log_excerpt="CudaGym server unreachable: ...")`
- Compilation failure: `EvalResult(status="compile_error", log_excerpt=<compile stderr>)`
- Execution timeout: `EvalResult(status="runtime_error", log_excerpt="Execution timed out")`
- Malformed JSON output: `EvalResult(status="runtime_error", log_excerpt="Failed to parse driver output")`

All errors are returned as `EvalResult` — the generator loop handles them the same as local failures.

### Testing Strategy

- Unit tests: mock CudaGym client, verify request construction and response parsing
- Integration tests: run against local CudaGym server with simple kernels
- End-to-end: run K-Search optimization loop with `--eval-backend cudagym` on a test task
