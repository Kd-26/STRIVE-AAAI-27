#!/usr/bin/env python3
"""Adapt the reviewed advanced-analysis notebook to the final paper artifact."""

import ast
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "outputs" / "strive-advanced-metric-analysis.ipynb"
OUTPUT = ROOT / "outputs" / "strive-final-advanced-analysis.ipynb"


def source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def replace_cell(notebook, index, text):
    notebook["cells"][index]["source"] = dedent(text).strip() + "\n"


notebook = json.loads(TEMPLATE.read_text(encoding="utf-8"))

# The template displays many intermediate dataframes; make that explicit for
# notebook and headless validation environments.
template_imports = source(notebook["cells"][3])
template_imports += "\ntry:\n    from IPython.display import display\nexcept Exception:\n    display = print\n"
notebook["cells"][3]["source"] = template_imports

replace_cell(
    notebook,
    0,
    """
    # STRIVE Final Advanced Analysis From the Paper Artifact

    This notebook loads the merged paper artifact ZIP and reproduces the advanced tables and
    figures without trajectory generation, sandbox execution, PRM loading, or API calls.

    The artifact already contains the authoritative repaired metrics. GPT-OSS remains in the
    operational coverage/correctness/latency tables, but is excluded from trace-derived
    analyses (`G`, `V`, `Q`, `PTU`, redundancy, token budgets, and Pareto trace comparisons)
    because its provider reasoning was not reliably mapped to visible protocol actions.
    """,
)

replace_cell(
    notebook,
    5,
    r'''
    RESULT_INPUT = (
        "/path/to/your/project/"
        "i-got-these-reviews-from-a/paper_artifact_build/"
        "strive_paper_artifact_20260715T160237Z.zip"
    )
    RUN_LABEL = "strive_v11_final_paper_artifact"
    OUTPUT_DIR = Path(
        "/kaggle/working/strive_final_advanced_analysis"
        if Path("/kaggle/working").exists()
        else Path.cwd() / "strive_final_advanced_analysis"
    ).resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    BOOTSTRAP_SAMPLES = 5000
    TOKEN_BUDGETS = [512, 1024, 2048, 4096, 8192]
    REDUNDANCY_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    INFRASTRUCTURE_STOPS = {"api_error", "worker_exception", "trajectory_timeout"}
    TRACE_EXCLUDED_AGENTS = {"gpt-oss-20b"}
    ''',
)

replace_cell(
    notebook,
    6,
    r'''
    def read_jsonl(path):
        with Path(path).open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


    def safe_extract_zip(path, target):
        target.mkdir(parents=True, exist_ok=True)
        root = target.resolve()
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                destination = (target / member.filename).resolve()
                if destination != root and root not in destination.parents:
                    raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
            archive.extractall(target)


    def resolve_bundle(value):
        source = Path(value).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        if source.is_file():
            if source.suffix.lower() != ".zip":
                raise ValueError("RESULT_INPUT must be a ZIP or directory")
            target = OUTPUT_DIR / "loaded_bundle" / source.stem
            if target.exists():
                shutil.rmtree(target)
            safe_extract_zip(source, target)
            source = target
        if (source / "manifest.json").is_file():
            return source
        manifests = list(source.rglob("manifest.json"))
        if len(manifests) != 1:
            raise RuntimeError(f"Expected one manifest.json under {source}, found {len(manifests)}")
        return manifests[0].parent


    def first_existing(root, *relative_paths):
        for relative in relative_paths:
            candidate = root / relative
            if candidate.exists():
                return candidate
        return None


    def load_bundle(value):
        root = resolve_bundle(value)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        metrics_path = first_existing(root, "metrics.jsonl", "raw_data/metrics.jsonl")
        trajectories_path = first_existing(root, "trajectories.jsonl", "raw_data/trajectories.jsonl")
        problem_path = first_existing(root, "problems.jsonl", "raw_data/problems.jsonl")
        prm_path = first_existing(root, "prm_scores.json", "raw_data/prm_scores.json")
        if metrics_path is None or trajectories_path is None:
            raise RuntimeError("Artifact is missing authoritative metrics or trajectories")
        return {
            "root": root,
            "manifest": manifest,
            "metrics_raw": read_jsonl(metrics_path),
            "trajectories_raw": read_jsonl(trajectories_path),
            "problems_raw": read_jsonl(problem_path) if problem_path else [],
            "prm_scores": json.loads(prm_path.read_text()) if prm_path else {},
        }


    if not RESULT_INPUT:
        raise RuntimeError("Set RESULT_INPUT to the uploaded paper artifact ZIP before continuing.")
    bundle = load_bundle(RESULT_INPUT)
    metrics_raw_all = bundle["metrics_raw"]
    trajectories_raw = bundle["trajectories_raw"]
    prm_scores_all = bundle["prm_scores"]
    manifest = bundle["manifest"]
    trace_agents = sorted({
        row.get("agent") for row in metrics_raw_all
        if row.get("agent") not in TRACE_EXCLUDED_AGENTS
    })
    metrics_raw = [row for row in metrics_raw_all if row.get("agent") in trace_agents]
    prm_scores = {agent: value for agent, value in prm_scores_all.items() if agent in trace_agents}
    print("Loaded all metric rows:", len(metrics_raw_all))
    print("Loaded trace-comparable rows:", len(metrics_raw))
    print("Loaded trajectory rows:", len(trajectories_raw))
    print("Operational agents:", sorted({row.get("agent") for row in metrics_raw_all}))
    print("Trace-comparable agents:", trace_agents)
    print("Run:", manifest.get("run_name", manifest.get("artifact_type")))
    ''',
)

replace_cell(
    notebook,
    8,
    r'''
    metrics_df = pd.json_normalize(metrics_raw, sep=".")
    operational_df = pd.json_normalize(metrics_raw_all, sep=".")
    trajectories_df = pd.json_normalize(trajectories_raw, sep=".")
    trajectories_by_key = {
        (row["agent_name"], row["problem_id"]): row for row in trajectories_raw
    }
    if len(trajectories_by_key) != len(trajectories_raw):
        raise RuntimeError("Duplicate (agent_name, problem_id) trajectory records detected")

    COLUMN_MAP = {
        "C": "C_G_V.C_final", "G": "C_G_V.G", "V": "C_G_V.V",
        "G_level": "C_G_V.G_level", "correctness_source": "C_G_V.correctness_source",
        "judge_used": "C_G_V.judge_used", "needs_judge": "C_G_V.needs_judge",
        "Q": "Q.Q_step", "PTU": "T.PTU", "billable": "T.T_billable",
        "useful": "T.T_useful", "pre_waste": "T.T_pre_waste",
        "post_waste": "T.T_post_waste", "answer_reporting": "T.T_answer_reporting",
        "hidden": "T.T_hidden_reasoning", "solution_step": "T.solution_evidence_step",
        "R_harmful": "R.R_harmful", "R_useful": "R.R_useful_verification",
        "R_semantic": "R.R_semantic_raw", "latency": "L.L_trajectory_sec",
        "service": "L.L_provider_service_sec", "pace": "L.L_rate_limit_wait_sec",
        "retry_wait": "L.L_retry_wait_sec", "retry_count": "L.retry_count",
    }
    for frame in (metrics_df, operational_df):
        for short, long_name in COLUMN_MAP.items():
            frame[short] = frame[long_name] if long_name in frame else np.nan
        for required in ("agent", "problem_id", "dataset", "subject", "finished", "stop_reason"):
            if required not in frame:
                frame[required] = np.nan
        frame["infrastructure_failure"] = frame["stop_reason"].isin(INFRASTRUCTURE_STOPS)
        frame["completed"] = frame["finished"].fillna(False).astype(bool)
        frame["level"] = [
            trajectories_by_key[(row.agent, row.problem_id)].get("level", "unknown")
            for row in frame[["agent", "problem_id"]].itertuples(index=False)
        ]
    core_metrics = ["C", "G", "V", "Q", "PTU", "R_harmful"]

    # Preserve the authoritative final tables generated from the two-bundle merge.
    final_metrics_dir = bundle["root"] / "final_metrics"
    final_core_all = pd.read_csv(final_metrics_dir / "core_metrics_all_models.csv") if (final_metrics_dir / "core_metrics_all_models.csv").exists() else pd.DataFrame()
    final_core_trace = pd.read_csv(final_metrics_dir / "core_metrics_trace_comparable.csv") if (final_metrics_dir / "core_metrics_trace_comparable.csv").exists() else pd.DataFrame()
    final_core_all.to_csv(OUTPUT_DIR / "final_core_metrics_all_models.csv", index=False)
    final_core_trace.to_csv(OUTPUT_DIR / "final_core_metrics_trace_comparable.csv", index=False)
    display(final_core_all.round(4))
    ''',
)

replace_cell(
    notebook,
    10,
    r'''
    coverage_rows = []
    for agent, group in operational_df.groupby("agent"):
        row = {
            "agent": agent,
            "n": len(group),
            "unique_problems": group["problem_id"].nunique(),
            "completion_rate": group["completed"].mean(),
            "infrastructure_failure_rate": group["infrastructure_failure"].mean(),
            "judge_fallbacks": int(group["judge_used"].fillna(False).sum()),
            "unresolved_judgments": int(group["needs_judge"].fillna(False).sum()),
            "trace_metric_eligible": agent in trace_agents,
        }
        for metric in core_metrics:
            # Coverage is always counted within this agent's operational rows.
            # Trace eligibility only controls whether trace-derived fields are
            # reported, not the denominator used for per-agent missingness.
            analysis_frame = group
            if agent not in trace_agents and metric in {"Q", "PTU", "R_harmful"}:
                row[f"{metric}_nan"] = len(group)
                row[f"{metric}_zero"] = np.nan
            else:
                row[f"{metric}_nan"] = int(analysis_frame[metric].isna().sum())
                row[f"{metric}_zero"] = int((analysis_frame[metric] == 0).sum())
        coverage_rows.append(row)
    coverage = pd.DataFrame(coverage_rows).sort_values("agent")
    display(coverage)
    coverage.to_csv(OUTPUT_DIR / "coverage_and_missingness.csv", index=False)

    # Operational sensitivity retains all six models for C/G/V; trace scores are shown
    # only for the five eligible models in the other advanced sections.
    sensitivity = pd.concat([
        frame.groupby("agent")[["C", "G", "V"]].mean().assign(view=name).reset_index()
        for name, frame in {
            "all_attempted": operational_df,
            "completed_only": operational_df[operational_df["completed"]],
            "infrastructure_clean": operational_df[~operational_df["infrastructure_failure"]],
        }.items()
    ], ignore_index=True)
    display(sensitivity)
    sensitivity.to_csv(OUTPUT_DIR / "clean_run_sensitivity.csv", index=False)
    g = sns.catplot(
        data=sensitivity.melt(id_vars=["view", "agent"], value_vars=["C", "G", "V"]),
        x="agent", y="value", hue="variable", col="view", kind="bar",
        height=4.2, aspect=1.15, sharex=False,
    )
    g.set_xticklabels(rotation=30)
    g.set(ylim=(0, 1))
    g.fig.savefig(OUTPUT_DIR / "clean_run_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.show()
    ''',
)

replace_cell(
    notebook,
    32,
    r'''
    analysis_manifest = {
        "source_run": manifest.get("run_name"),
        "source_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "token_budgets": TOKEN_BUDGETS,
        "redundancy_thresholds": REDUNDANCY_THRESHOLDS,
        "operational_agents": sorted(set(operational_df["agent"])),
        "trace_comparable_agents": trace_agents,
        "trace_excluded_agents": sorted(TRACE_EXCLUDED_AGENTS),
        "notes": {
            "bundle_rule": "The merged paper artifact is already deduplicated by (agent, problem_id).",
            "budget_curve": "Visible API output-token prefix budget; hidden reasoning availability varies by provider.",
            "post_solution": "Current solution point may be terminal for trajectories without traceable tool evidence.",
            "paired_tests": "Paired by problem_id; McNemar p-values use Holm correction.",
            "gpt_oss": "Retained operationally; excluded from trace-derived analyses due to unmapped provider reasoning.",
        },
    }
    (OUTPUT_DIR / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2), encoding="utf-8"
    )
    archive = Path(shutil.make_archive(str(OUTPUT_DIR), "zip", root_dir=OUTPUT_DIR))
    print("Advanced analysis directory:", OUTPUT_DIR)
    print("Downloadable archive:", archive)
    try:
        from IPython.display import FileLink, display as ipy_display
        ipy_display(FileLink(str(archive)))
    except Exception:
        pass
    ''',
)

# Validate normal Python cells before writing the notebook.
for index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue
    text = source(cell)
    text = "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith(("%", "!"))
    )
    ast.parse(text, filename=f"cell_{index}")

OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {OUTPUT}")
