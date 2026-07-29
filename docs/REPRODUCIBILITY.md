# STRIVE Reproducibility Protocol

## Scope

This is an evaluation study of execution-enabled reasoning agents. Every agent receives the same frozen
task set, ReAct-style controller, action budget, sandbox constraints, and output budget. API providers
are part of the experimental environment, so endpoint version, date, rate policy, retries, and failures
must be retained in the run manifest.

## Phases

### 1. Generation

Run `01_generate_trajectories.ipynb`. It creates a stateful trajectory where each model turn depends on
the preceding visible sandbox observation. The six model lanes run concurrently across problems, while
steps within a trajectory remain sequential. Each saved trajectory records prompts, visible and provider
output channels, parsed actions, code, stdout/stderr, timing, token estimates, and final answer.

The sandbox checks an AST allow-list before launching a resource-limited subprocess. It blocks filesystem,
network, process, and reflective primitives. It is appropriate for trusted benchmark-generated mathematics
code, not hostile code.

### 2. Recovery

Merge batches with Notebook 02. Recovery must be targeted: Notebook 03 retries only rows marked
`api_error`, `worker_exception`, `trajectory_timeout`, `format_error_limit`, or
`answer_only_violation_limit` according to the selected protocol. The recovery manifest must retain both
the original and replacement attempt history.

### 3. Evaluation and repair

The first-pass notebook computes deterministic equivalence, grounding, PRM features, token fields, and
semantic redundancy. It loads Math-Shepherd and Qwen2.5-Math PRMs sequentially, offloading one before
loading the next. Deterministically inconclusive final answers and selected near-boundary steps are sent
to the external Step 3.5 Flash evaluator through six independently paced NVIDIA lanes in Notebook 04.

### 4. Final metrics and analysis

Notebook 05 merges original and repaired metric bundles by `(agent, problem_id)`; repaired decisions
replace stale values rather than adding a second sample. Notebook 06 creates bootstrap intervals, paired
comparisons, metric sensitivity, budget curves, latency distributions, and subject/difficulty tables
without APIs or GPUs.

## Metric reporting rules

- Correctness `C` is final adjudicated correctness: deterministic equivalence where possible, then the
  external judge only for unresolved cases.
- Grounding `G` is an independent deterministic provenance score. It requires the declared final answer
  to be equivalent to a successful sandbox result or its deterministic transformation.
- Verified success is `V = C * G`.
- Step quality `Q` is a calibrated composite of two PRMs, execution signals, redundancy, and selectively
  invoked critic decisions. It is not a claim of human-ground-truth process correctness.
- Process token utility `PTU` is anchored at the first solution-evidence point. Report visible useful,
  pre-solution waste, post-solution waste, answer-reporting, API input/output, tool-output, and latency
  components separately.
- Harmful redundancy `R_harmful` excludes useful verification. A high-similarity step is harmful only
  when it repeats prior reasoning without new progress or verification value.
- `E_optional = V * PTU * Q * (1 - R_harmful)` is sensitivity-only and not a headline leaderboard.

## Acceptance checks

Do not publish a final table unless the release contains the final manifest, all source checksums, and the
per-agent completion/format-adherence/trace-eligibility table. Infrastructure-invalid trajectories should
be reported in operational coverage; trace-derived fields are `NA` when no usable trace exists.
