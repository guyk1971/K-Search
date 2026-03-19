# SkillForge-Search: Persistent Skill-Augmented World Models for LLM Kernel Optimization

## A Research Proposal for Integrating Evolutionary Skill Document Evolution with Structured Search via Co-Evolving World Models

**Principal Investigator:** [Name]
**Date:** March 2026
**Status:** Proposal for Internal Review

---

## 1. Executive Summary

We propose **SkillForge-Search** — a framework that integrates evolutionary skill document evolution (SkillForge) with structured tree-search guided by a co-evolving world model (K-Search), creating a system where persistent cross-task knowledge and within-task search reinforce each other bidirectionally.

**The core insight.** K-Search (Cao et al., 2026) demonstrated that LLMs can serve as effective world models for kernel optimization, using intrinsic knowledge to guide search through an optimization space. However, this world model operates entirely from the LLM's pretrained weights — it has no persistent cross-task memory. Every new kernel optimization starts from scratch. SkillForge addresses persistent knowledge accumulation but lacks a structured search mechanism for within-task optimization. SkillForge-Search unifies these: the SKILL.md becomes the world model's persistent, validated, cross-task memory, and K-Search's structured tree provides the highest-quality signal source for evolving that memory.

**Why this matters now.** K-Search achieves 2.10× average improvement over evolutionary baselines on complex FlashInfer kernels, but its world model resets between tasks. SkillsBench (Li et al., 2026) shows that human-curated Skills raise agent pass rates by 16.2pp while self-generated Skills provide zero benefit. SkillForge-Search asks: **can a skill document that co-evolves with a structured search process outperform both standalone search (K-Search) and standalone skill evolution (SkillForge)?**

**Three nested loops define the architecture:**

1. **Inner loop (Local Refinement):** Given a high-level optimization intent, repeatedly sample concrete implementations until stagnation. Unchanged from K-Search.
2. **Middle loop (Skill-Augmented Search):** K-Search's action selection, local refinement, and world model update phases — each augmented with SKILL.md context. A per-task scratchpad captures structured observations during search.
3. **Outer loop (Skill Evolution):** After each K-Search episode, reflect on the scratchpad and search tree to propose SKILL.md edits. Validate edits against held-out tasks. Maintain a Pareto frontier of SKILL.md variants.

**Key research questions:**

1. Does skill-augmented search converge faster and reach higher final scores than skill-free search (K-Search baseline)?
2. Does tree-structured insight extraction produce higher-quality SKILL.md entries than flat episode-based reflection (SkillForge baseline)?
3. Does value calibration error analysis — systematically comparing the world model's priority estimates against actual outcomes — reveal actionable knowledge that improves future search?
4. Does the evolved SKILL.md transfer across models, kernel types, and hardware targets?

---

## 2. Motivation and Problem Statement

### 2.1 The Persistent Knowledge Gap in LLM-Guided Search

K-Search's world model is powerful but ephemeral. During MLA Paged Decode optimization (Figure 2 in Cao et al., 2026), the world model learns that split-K parallelism is ineffective as a standalone strategy but highly effective atop a strong fusion kernel. It learns that register-resident rescaling should precede occupancy tuning. It learns that prescaling sm_scale upon Q loading yields the final optimum.

All of this knowledge is lost when the system starts the next kernel. If the system later optimizes GQA Paged Decode — which shares many structural similarities — it must rediscover these insights from scratch.

This is not a minor inefficiency. K-Search's 120-iteration budget means every wasted evaluation matters. The MoE kernel case is illustrative: OpenEvolve achieves only 3.09 final score because it cannot accumulate strategic knowledge across iterations. K-Search achieves 44.1 by maintaining a within-task search tree — but even K-Search's world model starts cold each time.

### 2.2 The Search Quality Gap in Skill Evolution

Conversely, SkillForge's evolutionary loop operates at the episode level: attempt a task, receive profiler feedback, reflect, propose SKILL.md edits. This flat structure misses the rich causal reasoning that K-Search's search tree captures. When K-Search prunes a branch, it encodes a judgment: "this strategy is not worth pursuing given what we've observed." When it inserts a child node under a successful parent, it encodes a compositional insight: "this refinement is promising in the context of the parent strategy."

SkillForge's episode-level reflection cannot capture these relational, compositional, and counterfactual insights because it never observes the branching structure of the optimization process.

### 2.3 The Opportunity: Bidirectional Reinforcement

The synthesis is natural:

- **SKILL.md → K-Search:** The skill document provides informed priors for the world model's value function, initial frontier construction, and profiler interpretation — reducing wasted evaluations and accelerating convergence.
- **K-Search → SKILL.md:** The search tree provides structured, causally rich signal for skill evolution — enabling relational insights ("X works only in the context of Y"), ordering constraints ("A must precede B"), and value calibration ("the model consistently overestimates the value of occupancy improvements for compute-bound kernels").

The bidirectional flow creates a virtuous cycle: better skills lead to more efficient search, and more efficient search produces richer signal for skill improvement.

---

## 3. Related Work

### 3.1 K-Search: Co-Evolving World Model for Kernel Optimization (Cao et al., 2026)

K-Search formulates kernel generation as a planning problem over a structured search tree. The LLM serves as an intrinsic world model that maintains a search state S_t, selects actions from a frontier based on priority scores V(a | S_t), executes local refinement to instantiate code, and then updates the search tree via insert/update/prune operations based on observed outcomes.

Key results: 2.10× average improvement over OpenEvolve across FlashInfer kernels, up to 14.3× on MoE, and state-of-the-art on GPUMode TriMul (1030 μs on H100). The system's advantage comes from decoupling high-level planning from low-level implementation and maintaining persistent within-task state via the search tree.

**Critical limitation for our purposes:** The world model has no cross-task memory. Its "co-evolution" is limited to in-context learning within a single optimization episode. Each new task starts from the LLM's pretrained priors alone.

### 3.2 SkillForge: Evolutionary Skill Document Evolution (Our Prior Proposal)

SkillForge evolves a single SKILL.md through iterative evolutionary search with rich profiler feedback. Key design elements include a multi-agent architecture (Executor/Proposer/Skill-Builder), Pareto-based candidate selection (adopted from GEPA), four-layer feedback engineering (compilation → correctness → performance → profiler decomposition), validation gate before commit, and production assimilation mode.

**Critical limitation addressed here:** SkillForge's within-task optimization is a simple generate-evaluate loop. It does not perform structured search over the optimization space — missing the compositional, relational, and counterfactual insights that a search tree provides.

### 3.3 GEPA, EvoSkill, SkillRL, RLEF, SkillsBench

These are covered in detail in the SkillForge proposal. Key points relevant to this extension:

- **GEPA** (Agrawal et al., 2025): Pareto-based selection and reflective mutation — adopted for the outer skill evolution loop.
- **EvoSkill** (Alzubi et al., 2026): Multi-agent separation, git-based variant management, validation discipline — adopted for SKILL.md management.
- **SkillRL** (Xia et al., 2026): Validates that skill abstraction improves agent performance but requires weight training. Cold-start insight informs our warm-up phase.
- **RLEF** (Gehring et al., ICML 2025): Complementary weight-based approach. SkillForge-Search externalizes knowledge into a readable document rather than internalizing into weights.
- **SkillsBench** (Li et al., 2026): Curated Skills +16.2pp, self-generated ~0pp. The definitive gap we aim to close.

### 3.4 Positioning SkillForge-Search

| Dimension | K-Search | SkillForge | **SkillForge-Search** |
|-----------|----------|------------|----------------------|
| Within-task search | Tree-structured world model | Flat generate-evaluate | **Skill-augmented tree search** |
| Cross-task memory | None (in-context only) | Evolved SKILL.md | **SKILL.md as persistent world model memory** |
| Knowledge extraction | Not formalized | Episode-level reflection | **Tree-structured reflection + value calibration** |
| World model priors | LLM weights only | N/A | **LLM weights + validated SKILL.md** |
| Feedback signal | Correctness + latency | 4-layer profiler decomposition | **4-layer profiler decomposition** |
| Insight granularity | Per-task observations | Per-episode observations | **Relational, compositional, counterfactual** |

**Novel contributions of SkillForge-Search:**

1. **Externalized, persistent world model memory.** The SKILL.md serves as validated cross-task memory for K-Search's world model, making the value function V(a | S_t, SKILL.md) informed by accumulated experience rather than pretrained weights alone.
2. **Tree-structured insight extraction.** The K-Search tree provides causally rich signal — branching structure, pruning decisions, parent-child relationships — that flat episode logs cannot capture.
3. **Value calibration as a learning signal.** Systematic analysis of V(a) predictions vs. actual outcomes reveals where the SKILL.md's knowledge is incomplete or incorrect, providing targeted improvement signal.
4. **Skill-conditioned search efficiency.** A measurable metric: search convergence speed (score achieved per evaluation budget) as a function of SKILL.md quality, closing the loop between skill quality and practical search performance.

---

## 4. Proposed Approach

### 4.1 Design Principles

**Principle 1: Bidirectional value flow.** The SKILL.md improves search efficiency; search improves SKILL.md quality. Both directions must be measurable.

**Principle 2: Structured observation over flat logging.** The per-task scratchpad captures structured observations (value calibration errors, ordering constraints, compositional insights) — not just "what happened."

**Principle 3: Validate before commit.** Unchanged from SkillForge. No SKILL.md edit is committed without empirical validation on held-out tasks.

**Principle 4: Incremental integration.** The architecture is designed so that each component can be evaluated independently: K-Search alone, K-Search + SKILL.md read-only, K-Search + scratchpad logging, and full bidirectional SkillForge-Search. This enables clean ablation.

### 4.2 The Three Nested Loops

#### Loop 1: Local Refinement (Inner — Unchanged from K-Search)

Given an action a_t = (x_parent, δ), repeatedly sample implementations x ~ π_code(· | a_t) and evaluate them o = E(x) until a stagnation condition is met (K consecutive attempts without improvement). This isolates implementation noise from strategic reasoning.

#### Loop 2: Skill-Augmented Search (Middle — K-Search + SKILL.md)

This is the primary integration point. The three K-Search phases each receive SKILL.md augmentation:

**Phase 1: Skill-Augmented Action Selection.**

The value function becomes:

    V(a | S_t, SKILL.md)

rather than V(a | S_t). Concretely, when the world model estimates priority scores for frontier actions, the SKILL.md provides:

- **Informed priors on strategy effectiveness:** "For memory-bound kernels with low L2 hit rates, coalescing improvements typically yield 2–3× more gain than compute optimizations." This biases V toward directions validated across prior tasks.
- **Known ordering constraints:** "Register-resident rescaling should precede occupancy tuning." This affects both V estimates and the insert logic for new child nodes.
- **Anti-patterns with context:** "Split-K parallelism is ineffective as a standalone strategy but highly effective atop a strong fusion kernel." This prevents premature pruning of strategies that require specific prerequisites.

**Phase 2: Skill-Augmented Local Refinement.**

The code generation policy becomes:

    x ~ π_code(· | a_t, SKILL.md)

The Executor agent reads the SKILL.md during code generation, applying known optimization techniques and avoiding documented pitfalls. This is the standard SkillForge executor behavior, now operating within K-Search's local refinement loop.

**Phase 3: Skill-Augmented World Model Update + Scratchpad Logging.**

After local refinement concludes, the world model update phase receives both the execution trajectory and the SKILL.md for context. This helps it:

- Interpret profiler output more accurately using the SKILL.md's profiler interpretation section.
- Make better insert/update/prune decisions based on accumulated domain knowledge.
- Identify when observations *contradict* the SKILL.md — these contradictions are high-value learning signals.

**The Scratchpad.** During the world model update phase, structured observations are logged to a per-task scratchpad:

```
{
  "task_id": "mla_paged_decode",
  "search_round": 42,
  "observation_type": "value_calibration_error",
  "details": {
    "action": "split_k_decoding",
    "parent_context": "root (no parent optimization)",
    "estimated_V": 0.7,
    "actual_outcome": "pruned after poor performance",
    "profiler_evidence": {
      "primary_bottleneck": "synchronization overhead",
      "warp_stall_dominant": "barrier"
    },
    "insight": "split_k is ineffective as standalone; requires fusion kernel as prerequisite",
    "generalizability": "likely applies to all paged attention variants"
  }
}
```

Observation types include:

- **Value calibration errors:** V estimate vs. actual outcome, with profiler evidence explaining the gap.
- **Ordering discoveries:** Strategy A required strategy B as prerequisite.
- **Compositional insights:** Refinement C emerged as a natural follow-on to parent D.
- **Pruning justifications:** Why a branch was abandoned, with evidence.
- **Novel patterns:** Profiler signatures or optimization techniques not currently in SKILL.md.
- **SKILL.md contradictions:** Cases where following SKILL.md advice led to worse outcomes.

#### Loop 3: Skill Evolution (Outer — SkillForge + Tree-Structured Reflection)

After each K-Search episode (one complete kernel optimization) concludes:

**Step 1: Tree-Structured Reflection.** The Proposer agent receives:
- The final search tree (with all closed nodes, pruned branches, and frontier state).
- The per-task scratchpad.
- The profiler traces from the best-performing kernel.
- The current SKILL.md.

The Proposer extracts candidate SKILL.md edits using the tree structure for richer attribution than flat episode logs:

- **From successful branches:** What strategies worked, in what order, with what parent context? What profiler signatures indicated they would work?
- **From pruned branches:** What looked promising but failed? Why? What distinguishing profiler signature separated the pruned branch from the successful one?
- **From value calibration errors:** Where were V estimates systematically wrong? What SKILL.md entries (or missing entries) contributed to the miscalibration?
- **From compositional patterns:** What parent-child relationships in the tree reveal that certain optimizations are only effective in specific contexts?

**Step 2: Validation Gate.** Candidate SKILL.md edits are validated against held-out tasks. Each edit must demonstrate improvement on the validation set before being admitted to the Pareto frontier. Per-task deltas are tracked to detect regressions.

**Step 3: Pareto Frontier Update.** Following GEPA's Pareto-based illumination strategy, the frontier maintains the k best SKILL.md variants. Variants are sampled proportional to the number of validation tasks where they are best. Each variant is stored as a git branch diverging only in its SKILL.md content.

### 4.3 Feedback Engineering

Unchanged from SkillForge. The four-layer feedback function μ_f provides:

- **Layer 1 (Compilation):** Binary + full compiler error messages.
- **Layer 2 (Correctness):** Pass/fail per test case, max absolute/relative error.
- **Layer 3 (Performance):** Throughput as % of cuBLAS/cuDNN/FlashInfer baseline.
- **Layer 4 (Profiler decomposition):** GPU Speed of Light, Memory Workload Analysis, Compute Workload Analysis, Occupancy Analysis, Warp State Statistics.

K-Search currently uses correctness + latency (Layers 1–3). SkillForge-Search adds Layer 4, providing the Proposer with structured diagnostic information for richer reflection and the world model with better signal for value estimation.

### 4.4 Value Calibration Analysis: A Novel Learning Signal

This is the mechanism we consider most novel. Every time the world model assigns V(a | S_t, SKILL.md) and then observes the actual outcome, there is an implicit prediction error. Over many episodes, systematic analysis of these errors reveals where the SKILL.md's knowledge is incomplete, wrong, or insufficiently contextualized.

**Formal definition.** For each action a evaluated during search, record:

    calibration_error(a) = outcome(a) - V(a | S_t, SKILL.md)

where outcome(a) is the normalized performance score of the best program found during local refinement of action a (0 if pruned without finding a correct program, J(x_best)/J(x*) otherwise, where x* is the global best).

**Aggregation.** After N episodes, aggregate calibration errors by:

- **SKILL.md entry relevance:** For each SKILL.md entry, what is the average calibration error when that entry is relevant to the action? If entry E says "use shared memory for reductions with >256 threads" and actions involving shared-memory reductions consistently have large positive calibration errors (outcomes better than predicted), the entry may be overly conservative. If calibration errors are consistently negative, the entry may be overly optimistic or misleading.
- **Kernel category:** Calibration errors may vary by kernel type (memory-bound vs. compute-bound, attention vs. MoE). Category-specific patterns suggest the SKILL.md needs context-conditional entries.
- **Search depth:** If calibration errors are large at tree depth 0 (initial strategies) but small at depth 3+ (refinements), the SKILL.md is weak on high-level strategy selection but good on local optimization tactics.

**Conversion to SKILL.md edits.** The value calibration analysis produces targeted edit proposals:

- Large systematic positive errors → "Entry X is overly conservative; add context about when the technique actually works well."
- Large systematic negative errors → "Entry Y is overly optimistic; add caveats about failure modes."
- Category-specific patterns → "Entry Z applies to memory-bound kernels but not compute-bound; add conditional guidance."

This is analogous to TD-learning in the original SKIRL proposal, but grounded in K-Search's structured search rather than flat episode observations, making the prediction errors more attributable and actionable.

### 4.5 Skill-Conditioned Search Efficiency

A practically important metric: how much does the SKILL.md improve K-Search's budget efficiency?

**Definition.** For a given kernel task and evaluation budget B:

    efficiency(SKILL.md, task, B) = J_best(B) / B

where J_best(B) is the best score achieved within B evaluations.

**Expected dynamics:**

- At episode 0 (empty SKILL.md): efficiency = K-Search baseline.
- After N episodes of skill evolution: efficiency should increase, meaning the same budget yields higher scores or the same score is reached with fewer evaluations.
- Diminishing returns are expected: the SKILL.md captures the most impactful insights first.

This metric directly measures the practical value of skill evolution for search — answering the question "is the overhead of maintaining and evolving a SKILL.md worth it?" with a concrete number.

### 4.6 Production Assimilation Mode

Unchanged from SkillForge. Beyond the controlled self-play loop, real-world CUDA optimization sessions can log lessons-learned to a buffer. These are periodically processed through the same validation gate and committed to the SKILL.md if they improve held-out performance.

The K-Search integration adds a new dimension: production sessions that use K-Search for optimization generate search trees and scratchpads, providing the same rich signal as controlled episodes. Production assimilation thus gets the same tree-structured reflection benefits.

### 4.7 Seed SKILL.md

The initial SKILL.md is deliberately sparse, providing foundational principles and profiler interpretation guidance:

```markdown
# CUDA Kernel Optimization Skill

## Core Principles
- Maximize memory coalescing for global memory accesses
- Minimize thread divergence within warps
- Use shared memory to reduce global memory traffic
- Choose grid/block dimensions for high occupancy

## Profiler Interpretation
- Start with "GPU Speed of Light" to identify compute-bound vs. memory-bound
- Check "Memory Workload Analysis" for coalescing and cache behavior
- Check "Warp State Statistics" for stall reasons

## Search Strategy Priors
(to be discovered through search episodes)

## Optimization Ordering Constraints
(to be discovered through search episodes)

## Anti-Patterns and Context-Conditional Guidance
(to be discovered through search episodes)

## Worked Examples with Profiler Evidence
(to be populated from search scratchpads)
```

Note the addition of "Search Strategy Priors" and "Optimization Ordering Constraints" sections — these specifically target the knowledge that K-Search's tree structure can provide.

---

## 5. Implementation Plan

### 5.1 Codebase Strategy: Building on K-Search

The K-Search codebase (https://github.com/caoshiyi/K-Search) provides a clean, modular foundation. The architecture separates concerns clearly:

- `kernel_generators/world_model.py` — World model data structures and JSON schema
- `kernel_generators/world_model_manager.py` — World model lifecycle (init/refine/select)
- `kernel_generators/world_model_prompts.py` — Prompt templates for world model operations
- `kernel_generators/kernel_generator_world_model.py` — Main search loop
- `tasks/task_base.py` — Task protocol, Solution, and EvalResult types

**Integration points:**

1. **SKILL.md injection into prompts:** Modify `world_model_prompts.py` and `kernel_generator_prompts.py` to include the current SKILL.md in the context for action selection, local refinement, and world model update phases.
2. **Scratchpad logging:** Add a structured logging module that records observations during the world model update phase (`world_model_manager.py`'s refine step).
3. **Profiler feedback integration:** Extend `task_base.py`'s EvalResult to include structured profiler output (Layer 4 feedback). Implement ncu profiling for FlashInfer-Bench tasks.
4. **Outer evolution loop:** New module wrapping the K-Search main loop, managing SKILL.md variants, Pareto frontier, and cross-episode reflection.

**Why K-Search as the base (not starting from scratch):**

- K-Search's search tree, stagnation detection, and world model update logic are non-trivial to reimplement and already well-tested.
- The task abstraction (FlashInfer-Bench, GPUMode) is ready to use.
- W&B integration provides logging infrastructure.
- Baseline comparisons (OpenEvolve, ShinkaEvolve) are already configured.
- We want to directly measure the delta from adding skill augmentation to K-Search.

### 5.2 Phased Implementation

#### Phase 1: Infrastructure and Baseline Reproduction (Weeks 1–2)

**Goal:** Reproduce K-Search results; establish baseline measurements.

**Tasks:**
- Clone K-Search repository and set up development environment.
- Reproduce MLA decode and GQA decode results to validate our setup.
- Instrument the codebase with detailed logging: per-round V estimates, actual outcomes, tree operations (insert/update/prune), and profiler output where available.
- Establish baseline metrics: final score, convergence speed (score at 25%, 50%, 75% of budget), and per-round value calibration.

**Deliverable:** Reproduced K-Search baselines with instrumented logging.

#### Phase 2: Profiler Feedback Integration (Weeks 3–4)

**Goal:** Add Layer 4 profiler decomposition to the evaluation pipeline.

**Tasks:**
- Implement ncu profiling integration for FlashInfer-Bench tasks. Parse profiler output into structured JSON matching SkillForge's μ_f specification.
- For tasks where ncu overhead is prohibitive, implement a tiered strategy: full profiling for the best kernel after each local refinement phase, lightweight metrics for intermediate evaluations.
- Verify that profiler output is correctly structured and informative by manually inspecting 5–10 episodes.
- Create the four-layer feedback function μ_f and integrate it into the EvalResult type.

**Deliverable:** Working profiler integration; μ_f producing structured feedback for FlashInfer kernels.

#### Phase 3: SKILL.md Read Path — Skill-Augmented Search (Weeks 5–7)

**Goal:** Inject SKILL.md into K-Search's search loop (read-only; no evolution yet).

**Tasks:**
- Modify `world_model_prompts.py` to include the SKILL.md in the context for:
  - Frontier initialization (initial action proposals).
  - Priority score estimation V(a | S_t, SKILL.md).
  - World model update reasoning (insert/update/prune decisions).
- Modify `kernel_generator_prompts.py` to include the SKILL.md in the context for code generation (local refinement).
- Implement a warm-up calibration: verify the agent reads and responds to SKILL.md content before measuring search performance.
- Run controlled experiments: K-Search with seed SKILL.md vs. K-Search with a manually enriched SKILL.md (containing insights from Phase 1 baseline logs) vs. K-Search baseline (no SKILL.md).
- Measure: does even a hand-written SKILL.md improve search efficiency?

**Deliverable:** Skill-augmented K-Search running; initial measurement of SKILL.md impact on search efficiency.

#### Phase 4: Scratchpad Logging and Tree-Structured Reflection (Weeks 8–10)

**Goal:** Implement the write path — structured observation capture and cross-episode reflection.

**Tasks:**
- Implement the per-task scratchpad with typed observation entries (value calibration errors, ordering discoveries, compositional insights, pruning justifications, novel patterns, SKILL.md contradictions).
- Hook scratchpad logging into the world model update phase: after each insert/update/prune decision, log the structured observation.
- Implement the Proposer agent for tree-structured reflection:
  - Input: final search tree + scratchpad + profiler traces + current SKILL.md.
  - Output: candidate SKILL.md edits with evidence and attribution from the tree structure.
- Implement the Skill-Builder agent for materializing edits with coherence checking.
- Run 10–15 K-Search episodes with scratchpad logging. Manually review scratchpad quality and Proposer output.

**Deliverable:** Full scratchpad logging; tree-structured reflection producing candidate SKILL.md edits.

#### Phase 5: Validation Gate and Pareto Frontier (Weeks 11–13)

**Goal:** Complete the outer evolution loop with validated skill commitment.

**Tasks:**
- Implement the validation gate: evaluate candidate SKILL.md variants against held-out kernel tasks.
- Implement Pareto frontier maintenance with weighted sampling (following GEPA's strategy).
- Implement git-based SKILL.md variant management (following EvoSkill's approach).
- Implement lineage tracking and periodic crossover (merge) between complementary SKILL.md lineages.
- Run the first complete SkillForge-Search loop: 30–50 K-Search episodes with automated SKILL.md evolution.
- Track SKILL.md growth, validation performance, and search efficiency trajectory.

**Deliverable:** Complete SkillForge-Search loop running autonomously; initial evolution trajectories.

#### Phase 6: Value Calibration Analysis (Weeks 14–15)

**Goal:** Implement and evaluate value calibration as a learning signal.

**Tasks:**
- Implement systematic calibration error aggregation across episodes.
- Implement calibration-driven SKILL.md edit proposals: identify entries associated with systematic miscalibration and propose targeted corrections.
- Compare SKILL.md evolution with vs. without calibration analysis as an additional signal source.
- Analyze calibration patterns: by kernel category, search depth, SKILL.md entry, and evolution epoch.

**Deliverable:** Value calibration analysis working; ablation results on its contribution.

#### Phase 7: Full Evaluation and Ablation (Weeks 16–20)

**Goal:** Large-scale evaluation across all research questions.

**Tasks:**
- Expand to all FlashInfer-Bench kernels (MLA decode, MLA prefill, GQA decode, MoE) plus GPUMode TriMul.
- Run complete ablation study:
  - (A) K-Search baseline (no SKILL.md)
  - (B) K-Search + static seed SKILL.md (read-only)
  - (C) K-Search + evolved SKILL.md (full SkillForge-Search)
  - (D) K-Search + evolved SKILL.md without value calibration
  - (E) K-Search + evolved SKILL.md without tree-structured reflection (flat episode reflection only)
  - (F) SkillForge baseline (flat generate-evaluate, no K-Search tree)
- Run model transfer experiments: evolve SKILL.md with Gemini-3-Pro, evaluate with GPT-5.2 and Claude.
- Expert review of evolved SKILL.md artifacts.
- Measure search efficiency trajectory over evolution epochs.

**Deliverable:** Full experimental results; expert-validated SKILL.md; paper draft.

#### Phase 8: Production Assimilation and Extensions (Weeks 21–24, if time permits)

**Goal:** Validate production assimilation and explore extensions.

**Tasks:**
- Accumulate lessons-learned from real CUDA optimization sessions.
- Run assimilation experiments.
- Explore: can the SKILL.md evolve to include *search strategy priors* — not just domain knowledge but meta-knowledge about how to search? (e.g., "For MoE kernels, prioritize routing optimization before computation optimization in the initial frontier.")
- Prototype adaptation for a second domain (database query optimization with EXPLAIN ANALYZE).

**Deliverable:** Production assimilation results; search strategy prior analysis; preliminary cross-domain evidence.

---

## 6. Experimental Design

### 6.1 Research Questions and Metrics

**RQ1: Does skill-augmented search outperform skill-free search?**
- Primary metric: Final best score (J*) across FlashInfer kernels, comparing conditions A vs. B vs. C (see Phase 7 ablation).
- Secondary metric: Convergence speed — score achieved at 25%, 50%, 75% of the 120-iteration budget.
- Protocol: 3 runs per condition per kernel; report mean and min-max range (following K-Search evaluation protocol).

**RQ2: Does tree-structured reflection produce better SKILL.md entries than flat reflection?**
- Metric: Validation performance of SKILL.md evolved via tree-structured reflection (C) vs. flat episode reflection (F).
- Analysis: Qualitative comparison of extracted insights — do tree-derived entries contain relational/compositional knowledge that flat-derived entries miss?

**RQ3: Does value calibration analysis improve skill evolution?**
- Metric: SKILL.md quality trajectory (validation performance over evolution epochs) comparing condition C (with calibration) vs. D (without calibration).
- Analysis: What fraction of committed SKILL.md edits originate from calibration analysis vs. standard reflection?

**RQ4: Does the evolved SKILL.md transfer across models?**
- Metric: Performance of Model B using a SKILL.md evolved with Model A, compared to Model B with no SKILL.md and with a SKILL.md evolved with Model B.
- Models: Evolve with Gemini-3-Pro (K-Search's default), evaluate transfer to GPT-5.2 and Claude Opus/Sonnet.

**RQ5: Does search efficiency improve over evolution epochs?**
- Metric: Score achieved per evaluation budget as a function of SKILL.md evolution epoch.
- Expected trajectory: monotonic improvement with diminishing returns.

### 6.2 Task Curriculum

| Tier | Tasks | Budget | Purpose |
|------|-------|--------|---------|
| Core (FlashInfer) | MLA Paged Decode, MLA Paged Prefill, GQA Paged Decode, MoE FP8 | 120 iters each | Main evaluation (matches K-Search setup) |
| Extended (GPUMode) | TriMul | 300 iters | Cross-benchmark validation |
| Validation | Subset of above held out from training | — | SKILL.md variant selection |
| Test | Subset never seen during evolution or validation | — | Final evaluation only |

### 6.3 Baselines

1. **K-Search (no SKILL.md):** Reproduced baseline from Cao et al.
2. **K-Search + seed SKILL.md:** Static, hand-written SKILL.md (read-only, no evolution).
3. **K-Search + evolved SKILL.md (SkillForge-Search full):** The complete system.
4. **SkillForge (flat):** Episode-level reflection without K-Search tree structure.
5. **OpenEvolve:** Evolutionary baseline from K-Search paper.
6. **ShinkaEvolve:** Population-based evolutionary baseline from K-Search paper.

---

## 7. Risks and Mitigations

| Risk | L | I | Mitigation |
|------|---|---|------------|
| SKILL.md injection increases prompt length, reducing K-Search performance | M | M | Measure prompt overhead; keep SKILL.md concise (SkillsBench: 2–3 focused sections); test with truncated variants |
| Skill-augmented world model becomes overconfident, reducing exploration | M | H | Monitor frontier diversity; compare exploration breadth with/without SKILL.md; add SKILL.md uncertainty markers |
| Scratchpad observations are too noisy for useful reflection | M | M | Structured observation types enforce quality; validation gate filters bad edits; manual review in Phase 4 |
| Value calibration analysis produces spurious correlations | L | M | Require minimum episode count before aggregation; statistical significance tests; cross-validation |
| K-Search codebase changes upstream, breaking our integration | L | L | Pin to specific commit; modular integration via separate wrapper modules |
| Profiler overhead makes 120-iteration budget impractical | M | M | Tiered profiling: full ncu only on best kernels after local refinement; lightweight metrics for intermediates |
| SKILL.md evolves to overfit specific kernel types | M | H | Diverse curriculum; three-way data split; track per-kernel-type validation performance separately |
| Cold-start: model ignores SKILL.md in prompts | M | M | Warm-up calibration phase; verify SKILL.md utilization before measuring evolution |
| LLM cost at scale (3 agents × many episodes × large contexts) | M | M | Sonnet for Executor, Opus for Proposer; batch efficiently; SKILL.md keeps contexts smaller over time (fewer dead ends) |

---

## 8. Expected Contributions

### Scientific Contributions

1. **Externalized persistent world model memory for LLM-guided search.** First demonstration that a validated, evolving knowledge document can serve as cross-task memory for a co-evolving world model, improving search efficiency over time.

2. **Tree-structured insight extraction for skill evolution.** Showing that the branching structure of a search tree (insert/update/prune decisions with parent-child relationships) provides causally richer signal for knowledge extraction than flat episode trajectories.

3. **Value calibration as a learning signal for procedural knowledge.** A novel mechanism: systematic analysis of V(a) predictions vs. actual outcomes reveals actionable gaps in the SKILL.md, enabling targeted improvement.

4. **Skill-conditioned search efficiency.** Empirical characterization of how knowledge accumulation affects search budget utilization — answering "how much faster does search converge as the skill document improves?"

5. **Synthesis of evolutionary skill evolution with structured search.** Bridging two distinct paradigms — population-based skill document evolution (GEPA/EvoSkill-inspired) and tree-structured planning with co-evolving world models (K-Search) — into a unified framework with bidirectional value flow.

### Practical Contributions

1. A **SkillForge-Search framework** built on the K-Search codebase, applicable to any kernel optimization task with rich evaluation feedback.
2. **Expert-validated SKILL.md for CUDA kernel optimization** containing cross-task strategic knowledge, ordering constraints, and context-conditional guidance — directly usable by practitioners.
3. **Search efficiency improvements** that reduce the evaluation budget needed for high-quality kernel generation — directly relevant to the practical deployment of LLM-guided kernel optimization systems.
4. **Design principles for integrating persistent knowledge with structured search** in LLM agent systems beyond kernel optimization.

### Relationship to Prior Work

SkillForge-Search synthesizes three lines of work:

- **From K-Search:** The co-evolving world model, search tree structure, local refinement with stagnation detection, and the task evaluation infrastructure.
- **From SkillForge:** The evolutionary SKILL.md framework, multi-agent architecture (Executor/Proposer/Skill-Builder), Pareto-based selection, validation gate, four-layer feedback engineering, and production assimilation mode.
- **From GEPA/EvoSkill:** Pareto-based candidate selection, reflective mutation, git-based variant management, and failure-driven iteration triggers.

The novel integration creates bidirectional value flow between persistent knowledge and within-task search — a pattern that, to our knowledge, has not been explored in the LLM agent literature.

### Publication Framing

"K-Search (Cao et al., 2026) showed that LLMs are effective world models for kernel optimization, achieving 2.10× improvement over evolutionary baselines. But this world model has no cross-task memory — it starts cold every time. SkillForge-Search gives the world model a persistent, validated memory in the form of an evolving SKILL.md. The skill document improves search; the search improves the skill document. We demonstrate that this bidirectional integration achieves [X]× improvement over K-Search alone, with search efficiency improving [Y]% over [N] evolution epochs, and that the evolved SKILL.md transfers meaningfully across models."

---

## 9. Resource Requirements

| Resource | Specification | Purpose |
|----------|--------------|---------|
| GPU compute | 2–4 NVIDIA H100/B200 | Kernel evaluation, profiling, FlashInfer-Bench |
| LLM API access | Gemini-3-Pro, GPT-5.2, Claude Opus + Sonnet | Search (Gemini/GPT), reflection (Opus), execution (Sonnet) |
| K-Search codebase | github.com/caoshiyi/K-Search (MIT-compatible) | Foundation for implementation |
| CUDA toolkit | CUDA 12.8 + Nsight Compute | Compilation, profiling environment |
| FlashInfer-Bench | flashinfer-bench + flashinfer-trace dataset | Task evaluation infrastructure |
| W&B | Weights & Biases | Experiment tracking, logging |
| Engineer time | 1 FTE, 20–24 weeks | Implementation, experimentation, analysis |
| Domain expert | Part-time CUDA/kernel specialist | SKILL.md review and validation |

---

## 10. References

1. Cao, S., Mao, Z., Gonzalez, J.E., & Stoica, I. (2026). K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model. arXiv:2602.19128.
2. Li, X. et al. (2026). SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks. arXiv:2602.12670.
3. Alzubi, S. et al. (2026). EvoSkill: Automated Skill Discovery for Multi-Agent Systems. arXiv:2603.02766.
4. Agrawal, L. A. et al. (2025). GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning. arXiv:2507.19457.
5. Xia, P. et al. (2026). SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning. arXiv:2602.08234.
6. Gehring, J. et al. (2025). RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning. ICML 2025.
7. Shinn, N. et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. NeurIPS 2023.
8. Wang, G. et al. (2023). Voyager: An Open-Ended Embodied Agent with Large Language Models. arXiv:2305.16291.
9. Madaan, A. et al. (2023). Self-Refine: Iterative Refinement with Self-Feedback. NeurIPS 2023.
10. Novikov, A. et al. (2025). AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery. Google DeepMind.
11. Superintelligence (2025). OpenEvolve: Teaching LLMs to Discover Algorithms Through Quality-Diversity Search.
12. Lange, R. T., Imajuku, Y., & Cetin, E. (2025). ShinkaEvolve: Towards Open-Ended and Sample-Efficient Program Evolution. arXiv:2509.19349.
13. Ouyang, A. et al. (2025). KernelBench: Can LLMs Write Efficient GPU Kernels? arXiv:2502.10517.
14. Xing, S. et al. (2026). FlashInfer-Bench: Building the Virtuous Cycle for AI-Driven LLM Systems. arXiv:2601.00227.
15. Khattab, O. et al. (2023). DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines. ICLR 2024.
16. Robeyns, J. et al. (2025). SICA: A Self-Improving Coding Agent. ICLR 2025 Workshop.
17. Yin, X. et al. (2024). Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement. ACL 2025.
18. Zelikman, E. et al. (2024). Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation. COLM 2024.
19. Mouret, J.-B. & Clune, J. (2015). Illuminating Search Spaces by Mapping Elites. arXiv:1504.04909.
20. Jaderberg, M. et al. (2017). Population-Based Training of Neural Networks. arXiv:1711.09846.
21. Ng, A. et al. (1999). Policy Invariance Under Reward Transformations. ICML 1999.
22. Hao, S. et al. (2023). Reasoning with Language Model is Planning with World Model. EMNLP 2023.
23. Anthropic. (2025). Equipping Agents for the Real World with Agent Skills.
24. Alderson, M. (2025). Self-Improving CLAUDE.md Files.

---

*This document is intended for internal review. Please direct feedback to [contact information].*
