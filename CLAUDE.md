# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

K-Search is an automated GPU kernel optimization system that uses LLMs (GPT-5, Gemini, etc.) to iteratively generate and optimize GPU kernels. It maintains a **co-evolving world model** — a structured search tree encoding hypotheses about kernel bottlenecks and optimization strategies — guiding multi-round, evidence-driven search over the kernel design space.

## Running

Entry point is `generate_kernels_and_eval.py`. Requires `sudo` for GPU access. Launch scripts are in `scripts/`.

```bash
# GPUMode TriMul (Triton kernels)
bash scripts/gpumode_trimul_wm.sh

# FlashInfer-Bench (CUDA kernels, requires flashinfer-trace dataset)
bash scripts/mla_decode_wm.sh
```

Required environment variables: `KSEARCH_ROOT`, `WANDB_API_KEY` (optional), and one of:
- `API_KEY` (or `LLM_API_KEY`) for direct OpenAI/Gemini backends
- `INFERENCE_API_KEY` for the NVIDIA Inference API backend (can be set in a `.env` file)

### LLM Backends

Two backends are supported via `--backend`:

- **`openai`** (default) — Direct OpenAI-compatible API. Requires `--api-key` / `LLM_API_KEY` and optionally `--base-url`.
- **`inference`** — NVIDIA Inference API (`https://inference-api.nvidia.com`). Provides unified access to GPT, Claude, and Gemini models through a single endpoint. Requires `INFERENCE_API_KEY` (set in `.env` or environment). Short model names (e.g., `gpt-5`, `claude-sonnet-4-6`, `gemini-3-pro`) are auto-resolved to provider paths via `INFERENCE_MODEL_MAP` in `generate_kernels_and_eval.py`.

To use the inference backend:
1. Create a `.env` file in the project root: `INFERENCE_API_KEY=your-key-here`
2. Set `BACKEND=inference` in the launch script, or pass `--backend inference` on the CLI.

### Key CLI flags

```bash
python3 generate_kernels_and_eval.py \
  --task-source flashinfer|gpumode \   # Task backend
  --model-name MODEL \                 # LLM model identifier (required)
  --backend openai|inference \         # LLM backend (default: openai)
  --language triton|cuda \             # Target language
  --world-model \                      # Enable world model (the main mode)
  --max-opt-rounds N \                 # Optimization rounds
  --save-solutions \                   # Persist solutions to disk
  --artifacts-dir DIR                  # Output directory (default: .ksearch)
```

### Remote Evaluation (CudaGym)

K-Search can offload kernel compilation and benchmarking to a remote CudaGym server:

```bash
python3 generate_kernels_and_eval.py \
  --task-source gpumode \
  --model-name gpt-5 \
  --eval-backend cudagym \
  --cudagym-url http://your-server:8000 \
  --remote-profile \
  ...
```

Required: `cudagym` Python package (`pip install cudagym>=0.8.0`).

The `--eval-backend cudagym` flag causes each task to build a self-contained evaluation bundle (kernel + driver + reference) that is sent to CudaGym for remote compilation, execution, and optional profiling. Results are returned as standard `EvalResult` objects — the generator loop is unaware of whether evaluation is local or remote.

### Dependencies

Base deps include `ninja` (required by PyTorch for JIT C++/CUDA extensions in gpumode).

```bash
# Using uv sync (preferred)
uv sync                        # base dependencies
uv sync --extra flashinfer     # + flashinfer-bench for flashinfer tasks

# Or manually
uv pip install openai python-dotenv wandb ninja
uv pip install git+https://github.com/caoshiyi/flashinfer-bench-ksearch.git  # for flashinfer tasks
```

## Architecture

### Core loop (`generate_kernels_and_eval.py`)

1. Instantiates a **Task** (FlashInfer or GPUMode) and a **KernelGenerator** (baseline or world-model)
2. Generator iteratively: generates/optimizes code via LLM → evaluates via task backend → refines world model
3. Final evaluation runs on the best solution; artifacts are persisted

### Key abstractions

- **`Task` protocol** (`k_search/tasks/task_base.py`): Task-agnostic interface that generators depend on. Defines `get_definition_text()`, `run_benchmark()`, `get_solution()`, etc. Two implementations:
  - `FlashInferBenchTask` — wraps flashinfer-bench dataset with workload-parallel evaluation
  - `GpuModeTriMulTask` — wraps GPUMode TriMul with vendored evaluator harness

- **`KernelGenerator`** (`k_search/kernel_generators/kernel_generator.py`): Base LLM-driven generator. Calls OpenAI-compatible API, parses code from responses, creates `Solution` objects, runs benchmark feedback loops.

- **`WorldModelKernelGeneratorWithBaseline`** (`k_search/kernel_generators/kernel_generator_world_model.py`): Extends base generator with world model. Manages action selection (refine/explore/debug), stagnation detection, and world model state persistence. This is the primary generator used in practice.

- **`WorldModelManager`** (`k_search/kernel_generators/world_model_manager.py`): Manages world model lifecycle — initialization, refinement after each round, action node selection via `WorldModelSelectionPolicy`.

- **`Solution` / `EvalResult`** (`k_search/tasks/task_base.py`): Task-agnostic data containers. `Solution` holds source files + build spec. `EvalResult` holds latency, speedup, status, and a generic `metrics` dict. The `score()` method provides a comparable scalar (higher is better).

- **`SolutionDB`** (`k_search/utils/solution_db.py`): JSONL-based solution persistence for tracking all generated solutions across rounds.

### Code flow for CUDA vs Triton

CUDA solutions use multi-file source (`kernel.h`, `kernel.cu`, `main.cpp`) serialized as XML blocks in prompts. Triton/Python solutions use a single source file. The `code_from_solution()` function in `task_base.py` handles this distinction.

### Prompt templates

- `kernel_generator_prompts.py` — Base prompts for initial generation and optimization rounds
- `world_model_prompts.py` — World-model-injected prompt variants (action-driven generation, debug, improve)
- `k_search/tasks/flashinfer_bench/prompts.py` — FlashInfer-specific kernel spec prompts
- `k_search/tasks/gpu_mode/` — GPUMode task spec and evaluator

### Artifacts layout

```
<artifacts-dir>/<task_name>/
├── solutions/       # Persisted Solution JSONs
├── eval/            # Evaluation report JSONs
└── world_model/     # World model state snapshots (world_model.json)
```

## Baselines

`baselines/openevolve/` and `baselines/shinkaevolve/` contain adapter configs for comparison with evolutionary kernel optimization systems. Each has YAML configs per kernel target and shell launch scripts.
