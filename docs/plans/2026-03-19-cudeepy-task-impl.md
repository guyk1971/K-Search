# CudeepyTask Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable K-Search's LLM optimization loop to generate and optimize kernels for cudeepy problem definitions (nn.Module files).

**Architecture:** A new `CudeepyTask` class implementing the `Task` protocol. It loads a cudeepy problem file, extracts the reference implementation and input specs, builds LLM-facing prompts from the problem definition, and evaluates generated kernels using the `k_search.eval.benchmark` library. Wired into `generate_kernels_and_eval.py` via `--task-source cudeepy`.

**Tech Stack:** Python, PyTorch, existing K-Search Task protocol, `k_search.eval.benchmark` library.

---

### Task 1: Create `k_search/tasks/cudeepy_task.py`

**Files:**
- Create: `k_search/tasks/cudeepy_task.py`

The class implements the Task protocol:
- `name` — problem filename stem
- `get_definition_text(language)` — builds LLM prompt from problem file (reference source, input shapes/dtypes, forward() signature, interface spec)
- `run_benchmark(solution)` — extracts `custom_kernel` from Solution sources, calls it with captured inputs, checks correctness, times with batch CUDA events, returns EvalResult
- `make_solution_from_generated_code(...)` — wraps generated code as `submission.py` with entry_point `submission.py::custom_kernel`
- `get_solution(name)` — resolves from artifacts dir
- `code_for_world_model_from_raw(raw, language)` — returns raw code string
- `seed_eval_for_base_solution(base_solution)` — runs `run_benchmark` on the base solution
- `get_config_for_logging()` — problem metadata
- Feedback hooks: `get_last_round_trace_logs_for_prompt()`, `get_last_round_passed_count()`, `get_last_round_total_workloads()`

### Task 2: Wire into `generate_kernels_and_eval.py`

**Files:**
- Modify: `generate_kernels_and_eval.py`

Add `--task-source cudeepy` choice and `--cudeepy-problem` argument. In the task construction block, import and instantiate `CudeepyTask`.

### Task 3: Verify syntax and basic imports

Run syntax check and a dry import to verify no import errors.

---
