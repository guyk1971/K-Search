# Action Space in K-Search

## Overview

K-Search's action space is **open-ended and LLM-generated**, not a fixed catalog of optimization patterns. There is no enum or registry of predefined actions. Instead, the LLM proposes actions as free-text descriptions within a structured JSON decision tree, drawing on its pretrained knowledge of GPU optimization as the prior over the space.

## Decision Tree Structure

The world model is organized as a **prefix tree** (decision tree):

- **Root node** (depth 0) — A dummy anchor (`decision=null`, `choice=null`).
- **Depth-1 nodes** ("kernel families") — High-level, end-to-end mapping/parallelization strategies. These must represent distinct architectural approaches (e.g., "row-major tiled reduction" vs "warp-cooperative streaming"). Naming them after implementation tactics is explicitly forbidden.
- **Deeper nodes** (depth >= 2) — Progressively more specific refinements and tactic choices within a family (e.g., vectorized loads, shared-memory buffering, specific tiling dimensions).

Each path from root to a leaf composes a full optimization plan. Branches at any decision point represent mutually exclusive alternatives.

## Action Node Schema

Each tree node can carry an `action` field with the following structure:

```json
{
  "title": "Use vectorized 128-bit loads for input tensor A",
  "description": "Replace scalar global loads with ...",
  "difficulty_1_to_5": 2,
  "score_0_to_1": 0.65,
  "expected_vs_baseline_factor": null,
  "rationale": "Memory bandwidth is the bottleneck..."
}
```

- **title / description** — Free-form natural language. The LLM invents these based on the kernel spec, target GPU, evaluation results, and existing tree state.
- **difficulty_1_to_5** — Estimated implementation complexity (1 = trivial, 5 = very hard).
- **score_0_to_1** — LLM's confidence that this action will yield improvement.
- **expected_vs_baseline_factor** — Predicted performance ratio if the action succeeds.
- **rationale** — Why this action is expected to help.

Actions must be "small, single-iteration implementable changes" — not sweeping rewrites.

## Two Phases of Action Generation

### Phase 1: Initialization

When a new kernel task begins, the LLM performs a deep 7-step analysis of the kernel specification:

1. **Problem classification** — e.g., matrix-multiply-like, reduction, scan, attention/softmax, elementwise fusion.
2. **Canonical math & dependencies** — Rewrite the computation in canonical form; identify independent dimensions.
3. **Data layout & access patterns** — Tensor shapes, strides, reuse opportunities, staging potential.
4. **Bottleneck hypotheses by regime** — At least 3 runtime regimes with likely bottlenecks (bandwidth/latency/compute/sync).
5. **Kernel design space (knobs)** — Tunable dimensions: mapping, tiling, memory movement, compute strategy, numerics.
6. **High-level kernel skeleton** — Phases, register vs shared memory placement, sync points.
7. **Candidate kernel families** — 2-3 pruned families with strengths/weaknesses.

This analysis is encoded into the JSON tree (not emitted as free text). The LLM must produce **at least 3 OPEN action nodes** as starting points.

**Key files:** `world_model.py:build_world_model_prompts()` (lines 837-922).

### Phase 2: Refinement (after each evaluation round)

After each evaluation, the LLM emits a small **edit script** to evolve the tree. Supported operations:

| Operation | Purpose |
|-----------|---------|
| `update_node` | Revise scores, ratings, confidence, notes on existing nodes |
| `insert_node` | Add a new child action under a solved node (continuation step) |
| `split_node` | Split a node into alternative branches |
| `delete_node` | Remove an OPEN leaf that is wrong, redundant, or dominated |

New nodes are **capped at 3 per edit script** to prevent runaway tree growth.

**Key files:** `world_model.py:build_decision_tree_edit_prompt()` (lines 925-1077).

#### Edit Script Example

The following is a realistic edit script that might be emitted after a round where the LLM applied a "use shared memory for tile reuse" action, got a PASSED eval with 1.12x vs baseline (prediction was 1.3x), and needs to update beliefs and propose a continuation:

```json
{
  "active_leaf_id": "node_trimul_r3_cont_1",
  "ops": [
    {
      "op": "update_node",
      "node_id": "node_shared_mem_tiling",
      "patch": {
        "overall_rating_0_to_10": 6.5,
        "confidence_0_to_1": 0.72,
        "notes": "CURRENT:\n  - does: Tiles 128x128 output block in shared memory with 32x32 inner tiles; loads A and B tiles cooperatively per warp.\n  - bottleneck: memory bandwidth (global loads still dominate; shared mem reuse helps but occupancy dropped from 50% to 37.5% due to smem pressure)\nFOLLOW_THROUGH:\n  - aligned_with_intent: yes\nUPDATE_BELIEF:\n  - Shared memory tiling works but gains are modest (1.12x vs predicted 1.3x). Occupancy regression partially offsets reuse benefit.\n  - next bet: Reduce shared memory footprint by double-buffering with smaller tiles (64x64) to recover occupancy.\nPERF_GAP:\n  - expected: 1.30x vs_base, observed: 1.12x. Hypothesis: smem allocation (2x 128x128xfp16 = 64KB) exceeds L1 budget, causing occupancy drop.",
        "last_updated_round": 3,
        "action": {
          "score_0_to_1": 0.55,
          "difficulty_1_to_5": 2,
          "expected_vs_baseline_factor": 1.12
        },
        "impacts": {
          "memory_bandwidth": {
            "rating_0_to_10": 7.0,
            "risk": "Global load volume reduced ~40% but occupancy regression limits effective throughput",
            "notes": "Shared mem reuse confirmed effective; bottleneck shifted to occupancy"
          },
          "register_pressure": {
            "rating_0_to_10": 5.0,
            "risk": "moderate — tiling loop uses 42 registers per thread",
            "notes": "Not the primary limiter at current tile size"
          },
          "compute_intensity_and_hw_fit": {
            "rating_0_to_10": 6.0,
            "risk": "occupancy at 37.5% leaves SMs underutilized",
            "notes": "Tensor cores not yet engaged",
            "hw_notes": "H100 SM can hold 2048 threads; current config uses 768"
          }
        }
      }
    },
    {
      "op": "update_node",
      "node_id": "node_vectorized_global",
      "patch": {
        "overall_rating_0_to_10": 4.0,
        "confidence_0_to_1": 0.40,
        "action": {
          "score_0_to_1": 0.30
        },
        "notes": "Downscored: shared memory tiling already reduced global load volume by ~40%. Vectorized loads alone unlikely to yield significant additional gain on top of tiled access pattern. SELF_CHECK: mutually exclusive with sibling 'warp-shuffle reduction' which targets a different bottleneck (reduction latency, not load bandwidth)."
      }
    },
    {
      "op": "insert_node",
      "parent_id": "node_shared_mem_tiling",
      "parent_solution_id": "sol_trimul_r3_a1b2c3",
      "node": {
        "node_id": "node_trimul_r3_cont_1",
        "decision": "Tile size vs occupancy tradeoff",
        "choice": "Reduce tile to 64x64 with double-buffering",
        "overall_rating_0_to_10": 7.0,
        "confidence_0_to_1": 0.55,
        "notes": "Continuation from shared_mem_tiling. Halving tile size to 64x64 reduces smem from 64KB to 16KB, potentially recovering occupancy to ~75%. Double-buffering overlaps next tile load with current compute.\nSELF_CHECK: cannot combine with sibling 'increase tile to 256x256' — opposite direction on the tile-size knob.",
        "impacts": {
          "memory_bandwidth": {
            "rating_0_to_10": 7.5,
            "risk": "Smaller tile means more global load rounds, but overlap hides latency",
            "notes": "Double-buffering should keep memory pipeline saturated"
          },
          "register_pressure": {
            "rating_0_to_10": 6.0,
            "risk": "Double-buffer pointers add ~8 registers",
            "notes": "Should remain within budget"
          },
          "compute_intensity_and_hw_fit": {
            "rating_0_to_10": 7.5,
            "risk": "low — occupancy recovery is the primary goal",
            "notes": "75% occupancy target should improve SM utilization significantly",
            "hw_notes": "H100 L1 cache is 256KB shared; 16KB smem per block allows 4+ concurrent blocks per SM"
          }
        },
        "action": {
          "title": "Reduce tile to 64x64 with async double-buffering to recover occupancy",
          "description": "Replace 128x128 shared memory tiles with 64x64 tiles (16KB smem per block). Use cp.async double-buffering: while computing on tile[i], prefetch tile[i+1]. This should recover occupancy from 37.5% to ~75% while preserving most of the reuse benefit.",
          "difficulty_1_to_5": 3,
          "score_0_to_1": 0.70,
          "expected_vs_baseline_factor": 1.35,
          "rationale": "Occupancy regression was the main reason the 1.3x prediction missed. Fixing it with smaller tiles + overlap should unlock the latent bandwidth gain."
        },
        "solution_ref": {
          "solution_id": null,
          "parent_solution_id": "sol_trimul_r3_a1b2c3"
        },
        "last_updated_round": 3
      }
    }
  ]
}
```

**What's happening in this edit script:**

1. **`update_node` on the solved action** — Revises ratings/confidence based on the actual 1.12x result (down from predicted 1.3x). Notes the PERF_GAP and updated bottleneck hypothesis (occupancy, not bandwidth, is now the limiter). Downgrades `score_0_to_1` from its original value to 0.55.

2. **`update_node` on a sibling action** — Downscores an alternative action ("vectorized global loads") that is now less relevant given that shared memory tiling already addressed load bandwidth.

3. **`insert_node` as continuation** — Creates a new OPEN action child under the solved node, proposing the next step in the optimization chain. This satisfies the "continuation rule" — a solved node must have at least one OPEN child to keep the search moving. The new action targets the specific bottleneck revealed by the evaluation (occupancy regression).

The system then deterministically applies these ops via `_apply_decision_tree_ops()`, validates the result (root invariants preserved, solutions not dropped, continuation exists), and if valid, stores the updated tree.

### Within-Round Reflection

The edit prompt requires structured reflection before updating the tree:

- **CURRENT** — What the solution does; dominant bottleneck (bandwidth/latency/compute/sync).
- **FOLLOW_THROUGH** — Whether the generated code matches the action's intent (yes/no + explanation).
- **UPDATE_BELIEF** — What changed after evaluation; what to try next and why.
- **PERF_GAP** — Expected vs observed performance when predictions were wrong.

This reflection updates node scores and notes within the session but is **not persisted across tasks**.

## Action Selection (Deterministic)

`choose_next_action_node_id()` picks the next action to execute **without an LLM call**. The algorithm:

1. **Frontier filter** — Select nodes that:
   - Have an `action.title` (proposed optimization).
   - Have NO attached `solution_id` (not yet attempted).
   - Whose parent already has a solution, or is root.
   This enforces sequential chaining — step 2 cannot execute until step 1 succeeds.

2. **Difficulty gating** — Only consider actions with `difficulty_1_to_5 <= max_allowed` (default: 4). The ceiling relaxes once the best solution exceeds a configurable performance threshold.

3. **Deterministic ranking** — Sort by:
   - `score_0_to_1` (descending) — higher-confidence actions first
   - `difficulty_1_to_5` (ascending) — easier actions first
   - `overall_rating_0_to_10` (descending) — better-rated plan paths first
   - `node_id` (lexicographic) — stable tiebreaker

   Pick the top candidate.

**Key files:** `world_model_manager.py:choose_next_action_node_id()` (lines 1020-1202).

## Execution Cycle

Once an action node is selected:

1. **Attempt 1** — The LLM generates code implementing the action on top of the parent node's base code.
2. **Attempt 2+** — Debug/improve loop: same action, informed by evaluation results and failure logs.
3. **Cycle exit** — When no improvement is observed for N consecutive rounds (stagnation), or max rounds are hit.
4. **Next cycle** — Pick the next action from the frontier and repeat.

The inner loop prompts are action-scoped: the LLM is told to "implement ONLY the chosen action; keep everything else as close as possible to the base implementation."

**Key files:** `kernel_generator_world_model.py` (lines 410-600), `world_model_prompts.py`.

## Information Sources for Action Node Fields

The action node schema is populated through **three distinct channels**, not just the LLM's pretrained knowledge:

### Channel 1: LLM Priors (initialization only)

At init time, the LLM sees only the kernel specification, target GPU, and language. No evaluation data exists yet. All initial values — `score_0_to_1`, `difficulty_1_to_5`, `overall_rating_0_to_10`, `confidence_0_to_1`, impacts, and rationale — are filled from the LLM's pretrained knowledge conditioned on the spec. This is the "weights only" phase.

### Channel 2: LLM Reasoning over Episode Evidence (refinement)

After each evaluation round, the edit prompt feeds the LLM a rich evidence bundle:

| Signal | Source | Contents |
|--------|--------|----------|
| **eval_result** | Benchmark harness | `status` (passed/failed), `latency_ms`, `speedup_factor`, `mean_vs_baseline_factor`, `metrics.score` |
| **prediction** | Previous action node | `expected_vs_baseline_factor`, `confidence`, `rationale` — enables comparison of predicted vs actual outcome |
| **current_code_excerpt** | Generated kernel source | The actual code produced, so the LLM can reason about what was implemented |
| **trace_logs** | Task evaluator | Compilation errors, runtime errors, test failures, benchmark error messages (up to 8KB) |
| **perf_summary** | Computed from EvalResult | Formatted lines like `last_attempt_mean_latency_ms: 0.1234`, `base_vs_baseline: 1.05x` |
| **current_tree_path** | World model state | Root-to-active-leaf path showing the chain of decisions that led here |
| **open_frontier_nodes** | World model state | All executable candidate actions with their current scores/ratings |
| **WM status summary** | World model state | Best node, exploration stats, under-explored branches |

The LLM uses this evidence to revise scores, ratings, confidence, difficulty estimates, and bottleneck hypotheses on existing nodes, and to propose new actions informed by what worked or failed.

### Channel 3: Deterministic (non-LLM) Updates

Some state is written **without the LLM**:

- **`computed_signals`** — After each eval round, `merge_computed_signals()` deterministically writes `round_index`, `status`, `latency_ms`, `reference_latency_ms`, and `speedup_factor` into the world model JSON. The LLM can read these but doesn't produce them.
- **`solution_ref.eval`** — When a solution passes, the system attaches the eval result directly onto the node without asking the LLM.

### What's NOT Available

The system does **not** feed hardware profiler data (ncu/Nsight counters, occupancy, memory throughput, warp stall reasons) into these prompts. The bottleneck analysis in action nodes is the LLM's **inference** from timing numbers + code structure, not from actual profiler measurements. The `trace_logs` contain error messages and benchmark results but not profiler counters.

**Key files:** `world_model.py:build_decision_tree_edit_prompt()` (lines 925-1077), `world_model.py:merge_computed_signals()` (lines 1427-1478), `task_base.py:EvalResult` (lines 18-31).

## Cross-Session Persistence

The action space and decision tree are **ephemeral per kernel task**. Each new task starts fresh from the LLM's pretrained priors. The world model JSON is saved to disk (`.ksearch/<task>/world_model/world_model.json`) only for resumption within the same task.

There is no mechanism to transfer learned optimization patterns, successful strategies, or anti-patterns from one kernel task to future tasks. This is identified as a critical limitation in the [SkillForge-Search proposal](skillforge-ksearch-proposal.md).
