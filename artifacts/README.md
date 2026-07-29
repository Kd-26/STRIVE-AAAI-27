# Required Release Artifacts

For a full paper reproduction, publish these immutable ZIPs alongside a checksum manifest:

1. Three generation batch archives, each with `trajectories.jsonl`, `problems.jsonl`,
   `generation_health.csv`, and `generation_manifest.json`.
2. Any targeted trajectory-recovery archive and its merge audit.
3. The evaluator-repair archive containing judge and critic decisions plus checkpoints.
4. The final metrics-pool archive containing authoritative `metrics.jsonl`, trajectories, PRM outputs,
   coverage tables, and the merge manifest.
5. The final advanced-analysis archive containing CSV tables and paper figures.

Do not concatenate overlapping artifacts as independent samples. The final metrics notebook de-duplicates
by `(agent, problem_id)` and uses repaired rows as authoritative.

The extraction and final analysis notebooks accept either a ZIP or an extracted directory. Validate every
generation archive with:

```bash
python scripts/validate_generation_artifact.py /path/to/archive.zip
```
