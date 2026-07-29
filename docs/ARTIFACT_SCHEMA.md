# Artifact Schema

## Generation archive

Every generation ZIP has this flat layout:

```text
generation_manifest.json
generation_health.csv
problems.jsonl
trajectories.jsonl
```

`generation_manifest.json` includes the protocol configuration, model definitions, selected-task count,
data schema version, and SHA-256 checksums for each data file. `trajectories.jsonl` uses one object per
`(agent_name, problem_id)` attempt. Important fields include `steps`, `final_answer`, `finished`,
`stop_reason`, latency components, visible/provider outputs, and sandbox observations.

## Final metrics archive

The authoritative metrics artifact includes at least:

```text
manifest.json
metrics.jsonl
trajectories.jsonl
problems.jsonl
prm_scores.json
summary.csv
```

One `metrics.jsonl` row corresponds to one agent-task attempt. Nested namespaces use `C_G_V`, `Q`, `T`,
`R`, and `L` for correctness/grounding, process quality, token accounting, redundancy, and latency.
The final merge is keyed by `(agent, problem_id)`; a repaired row supersedes an original row for the same
key.

## Evidence fields

Grounding fields include the final answer span, candidate answer source, symbolic-equivalence outcome,
grounding level (`G0` through `G3`), and evidence pointer. A correct intermediate tool output does not
ground a different declared final answer.
