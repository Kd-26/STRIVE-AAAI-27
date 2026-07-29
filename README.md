# STRIVE Reproducibility Repository

This repository accompanies **STRIVE: Auditing Execution Enabled Reasoning Agents Beyond Accuracy**.
It reproduces the paper's execution-enabled reasoning-agent evaluation: multi-step generation with a
Python tool, deterministic provenance checks, selective external adjudication, two locally loaded
process reward models (PRMs), core metrics, and all paper-facing analysis.

## What this repository does

STRIVE evaluates fixed hosted language models; it does **not** train or fine-tune the agents. The
word "generation" below means generating agent trajectories under a frozen controller and problem
set. The locally loaded Math-Shepherd and Qwen PRMs are pretrained evaluators, not models trained by
this repository.

The paper protocol contains 300 frozen tasks (200 MATH-500 and 100 text-only OlympiadBench math
problems) evaluated by six agents, for 1,800 attempted trajectories. The five trace-comparable agents
are GLM-5.2, GPT-5 Nano, MiniMax-M3, Ministral-14B, and Nemotron-3-Nano. GPT-OSS-20B remains in
operational coverage/correctness/latency tables, but is excluded from trace-derived comparisons because
its provider-reasoning channel was not reliably mapped to visible controller actions.

## Repository map

| Path | Purpose |
| --- | --- |
| `notebooks/01_generate_trajectories.ipynb` | Frozen task selection, six model lanes, ReAct controller, Python sandbox, checkpoints, and first-pass metrics. |
| `notebooks/02_merge_generation_archives.ipynb` | Merges generation batches and verifies archive identity. |
| `notebooks/03_rerun_invalid_trajectories.ipynb` | Resumes only failed or protocol-invalid trajectories. |
| `notebooks/04_repair_judge_and_critic.ipynb` | Applies selective correctness and step-critic repair calls with six NVIDIA lanes. |
| `notebooks/05_compute_final_metrics.ipynb` | Builds the authoritative de-duplicated metrics pool from original and repaired bundles. |
| `notebooks/06_advanced_analysis.ipynb` | Recreates paper tables, confidence intervals, sensitivity plots, and supplementary analysis offline. |
| `configs/paper_v11.example.json` | Human-readable frozen protocol and endpoint configuration. |
| `data/sample/batch01_generation.zip` | A real generation-batch archive for schema and integrity smoke tests. |
| `scripts/validate_generation_artifact.py` | Validates a generation ZIP or extracted directory, including manifest checksums. |
| `scripts/validate_notebooks.py` | Checks notebook JSON structure, required headings, and absence of credentials. |
| `docs/REPRODUCIBILITY.md` | End-to-end procedures, expected artifacts, and reporting rules. |

## Quick start

```bash
git clone <YOUR-REPOSITORY-URL> strive-reproducibility
cd strive-reproducibility
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set only your own keys in `.env`; never paste keys into notebooks or commit `.env`.

```bash
set -a
source .env
set +a
python scripts/validate_notebooks.py
python scripts/validate_generation_artifact.py data/sample/batch01_generation.zip
jupyter lab
```

For a low-cost plumbing check, set `paper_sample_limit` to `5` in Notebook 01. The five-task smoke set
is disjoint from the paper task set and must not enter paper tables. For the full experiment, restore
`paper_sample_limit = None`; Notebook 01 makes three stratified 100-problem batches.

## Execution order

1. Run Notebook 01 for smoke testing or generation. It writes append-only checkpoints and a generation
   ZIP after each batch.
2. Use Notebook 02 to merge batch ZIPs. Use Notebook 03 only when the merged health table identifies
   retryable infrastructure or protocol failures.
3. Run Notebook 04 only for selective judge and critic calls. This consumes NVIDIA NIM evaluator API
   capacity but does **not** regenerate trajectories or reload PRMs.
4. Run Notebook 05 using the original metrics bundle and repaired evaluator bundle. It de-duplicates
   records by `(agent, problem_id)` and makes repaired records authoritative.
5. Run Notebook 06 on the final metrics bundle. This phase is offline: no generation APIs, sandbox
   executions, PRM loading, or judge calls are needed.

The configuration cells intentionally use `/path/to/your/...` placeholders. Point them to uploaded ZIPs
or extracted artifact directories before execution.

## Requirements and compute

Generation requires access to the listed hosted model endpoints. Metric computation requires a CUDA GPU
with approximately 16 GB VRAM for four-bit PRM loading; PRMs are loaded and offloaded one at a time.
The Python executor is an AST-restricted subprocess intended for trusted benchmark-generated math code.
It is not a hardened isolation boundary for adversarial code.

The exact dependencies, model IDs, PRM revision, controller limits, rate limits, timeout policy, and
metric weights are recorded in `configs/paper_v11.example.json` and copied into every run manifest.

## Data and artifact policy

The included sample archive is a single generation batch only. Full trajectory bundles and the final
metrics bundle should be distributed as a versioned release or archival deposit rather than committed to
Git. See `artifacts/README.md` for the required release files and checksums.

The supplied legacy `Experimentation_STRIVE.ipynb` is intentionally not included. It is an older
HotpotQA/Wikipedia experiment, uses the retired `S/T/N/R/IG` formulation, does not train a model, and
contained embedded credentials. It must not be cited as the code used for this paper.

## Reproducibility checks

Before reporting results, verify that every artifact has its expected count and checksum, and report:

- attempted, completed, format-adherent, and infrastructure-failed trajectories per agent;
- symbolic, judge-fallback, and unresolved correctness cases;
- grounding levels and evidence-pointer coverage;
- trace-eligibility policy, including GPT-OSS's operational-only treatment;
- exact endpoint model IDs, dates, controller budgets, and API failure policy.

See `docs/REPRODUCIBILITY.md` for the full protocol and `docs/ARTIFACT_SCHEMA.md` for JSON fields.

## License

Choose and add a public license before releasing this repository. No license is implied by this draft
submission package.
