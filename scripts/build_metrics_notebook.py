#!/usr/bin/env python3
"""Build the final STRIVE metrics notebook from the original and repaired bundles."""

import ast
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "strive-final-metrics-from-two-bundles.ipynb"


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip() + "\n",
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip() + "\n",
    }


cells = [
    md(
        """
        # STRIVE: Final Metrics From the Two Bundle Pool

        This notebook produces the final paper-facing tables from:

        1. the original five-ZIP metrics bundle; and
        2. the repaired Step 3.5 Flash judge/critic bundle.

        The bundles contain overlapping snapshots of the same 1,800 `(agent, problem_id)`
        attempts. They are **not concatenated as independent samples**. The repaired record
        is authoritative whenever a key exists in both bundles; the original record is used
        only as a fallback and audit reference.

        GPT-OSS is retained for operational correctness, coverage, format, cost, and latency
        results. It is excluded only from trace-derived `G`, `V`, `Q`, `PTU`, redundancy, and
        `E` comparisons because its provider reasoning was not reliably mapped to visible
        controller actions.
        """
    ),
    md("## 1. Configuration"),
    code(
        r'''
        import hashlib
        import io
        import json
        import math
        import shutil
        import zipfile
        from collections import Counter
        from datetime import datetime, timezone
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        try:
            from IPython.display import FileLink, display
        except Exception:
            FileLink = None


        ORIGINAL_BUNDLE = (
            "/path/to/your/artifacts/"
            "paper_math200_olympiad100_v11_five_zip_metrics_20260715T035532Z.zip"
        )
        REPAIRED_BUNDLE = (
            "/path/to/your/artifacts/"
            "paper_math200_olympiad100_v11_step35_judge_critic_repaired_20260715T142545Z.zip"
        )

        EXPECTED_MODELS = [
            "ministral-14b", "glm-5.2", "minimax-m3",
            "nemotron-3-nano", "gpt-oss-20b", "gpt-5-nano",
        ]
        EXPECTED_ATTEMPTS_PER_MODEL = 300
        TRACE_EXCLUDED_AGENTS = {"gpt-oss-20b"}
        BOOTSTRAP_REPLICATES = 2000
        RANDOM_SEED = 42

        WORK_ROOT = Path(
            "/kaggle/working" if Path("/kaggle/working").exists() else Path.cwd()
        ).resolve()
        OUTPUT_ROOT = WORK_ROOT / "strive_final_metrics_pool"
        FIGURE_ROOT = OUTPUT_ROOT / "figures"
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

        print("Original bundle:", ORIGINAL_BUNDLE)
        print("Repaired bundle:", REPAIRED_BUNDLE)
        print("Output root:", OUTPUT_ROOT)
        '''
    ),
    md("## 2. Load and verify both bundles"),
    code(
        r'''
        REQUIRED_FILES = {"manifest.json", "metrics.jsonl", "trajectories.jsonl", "problems.jsonl"}
        OPTIONAL_FILES = {
            "summary.csv", "paper_primary_metrics.csv", "paper_trace_metrics.csv",
            "evaluator_coverage.csv", "generation_health.csv", "merge_audit.json",
        }


        def parse_jsonl(payload, label):
            rows = []
            for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid JSON in {label}, line {line_number}") from exc
                if not isinstance(value, dict):
                    raise RuntimeError(f"Expected object in {label}, line {line_number}")
                rows.append(value)
            return rows


        def unique_name(names, basename, required=True):
            matches = [name for name in names if Path(name).name == basename]
            if len(matches) == 1:
                return matches[0]
            if not matches and not required:
                return None
            raise RuntimeError(f"Expected one {basename}; found {len(matches)}")


        def find_extracted_file(root, basename, required=True):
            matches = [path for path in root.rglob(basename) if path.is_file()]
            if len(matches) == 1:
                return matches[0]
            if not matches and not required:
                return None
            raise RuntimeError(f"Expected one {basename} under {root}; found {len(matches)}")


        def load_metrics_bundle(value):
            path = Path(value).expanduser().resolve()
            payloads = {}
            if path.is_dir():
                root = find_extracted_file(path, "metrics.jsonl").parent
                for name in REQUIRED_FILES | OPTIONAL_FILES:
                    member = root / name
                    if member.is_file():
                        payloads[name] = member.read_bytes()
                source = str(root)
            elif path.is_file() and path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as archive:
                    for name in REQUIRED_FILES | OPTIONAL_FILES:
                        member = unique_name(archive.namelist(), name, required=name in REQUIRED_FILES)
                        if member:
                            payloads[name] = archive.read(member)
                source = str(path)
            else:
                raise FileNotFoundError(f"Expected ZIP or extracted directory: {path}")

            manifest = json.loads(payloads["manifest.json"])
            for name, expected in manifest.get("sha256", {}).items():
                basename = Path(name).name
                if basename in payloads:
                    actual = hashlib.sha256(payloads[basename]).hexdigest()
                    if actual != expected:
                        raise RuntimeError(f"Checksum mismatch for {basename} in {source}")

            result = {
                "source": source,
                "manifest": manifest,
                "payloads": payloads,
                "metrics": parse_jsonl(payloads["metrics.jsonl"], f"{source}/metrics.jsonl"),
                "trajectories": parse_jsonl(payloads["trajectories.jsonl"], f"{source}/trajectories.jsonl"),
                "problems": parse_jsonl(payloads["problems.jsonl"], f"{source}/problems.jsonl"),
            }
            for name in OPTIONAL_FILES:
                if name not in payloads:
                    continue
                if name.endswith(".csv"):
                    result[name] = pd.read_csv(io.BytesIO(payloads[name]))
                elif name.endswith(".json"):
                    result[name] = json.loads(payloads[name])
            return result


        original = load_metrics_bundle(ORIGINAL_BUNDLE)
        repaired = load_metrics_bundle(REPAIRED_BUNDLE)
        print("Original rows:", len(original["metrics"]), "source:", original["source"])
        print("Repaired rows:", len(repaired["metrics"]), "source:", repaired["source"])
        print("Repaired evaluator:", repaired["manifest"].get("judge", {}))
        '''
    ),
    md("## 3. Deduplicate the overlapping sample pool"),
    code(
        r'''
        def metric_key(row):
            key = (str(row.get("agent", "")).strip(), str(row.get("problem_id", "")).strip())
            if not all(key):
                raise RuntimeError(f"Metric row missing key: {row}")
            return key


        def trajectory_key(row):
            key = (str(row.get("agent_name", "")).strip(), str(row.get("problem_id", "")).strip())
            if not all(key):
                raise RuntimeError(f"Trajectory row missing key: {row}")
            return key


        def index_unique(rows, key_fn, label):
            indexed = {}
            for row in rows:
                key = key_fn(row)
                if key in indexed:
                    raise RuntimeError(f"Duplicate {label} key: {key}")
                indexed[key] = row
            return indexed


        def digest(row):
            return hashlib.sha256(
                json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
            ).hexdigest()


        original_metrics = index_unique(original["metrics"], metric_key, "original metric")
        repaired_metrics = index_unique(repaired["metrics"], metric_key, "repaired metric")
        original_trajectories = index_unique(original["trajectories"], trajectory_key, "original trajectory")
        repaired_trajectories = index_unique(repaired["trajectories"], trajectory_key, "repaired trajectory")

        all_keys = sorted(set(original_metrics) | set(repaired_metrics))
        final_metrics_records = []
        source_rows = []
        for key in all_keys:
            if key in repaired_metrics:
                selected = repaired_metrics[key]
                selected_source = "repaired"
            else:
                selected = original_metrics[key]
                selected_source = "original_fallback"
            source_rows.append({
                "agent": key[0], "problem_id": key[1],
                "selected_source": selected_source,
                "present_original": key in original_metrics,
                "present_repaired": key in repaired_metrics,
                "overlap_changed": (
                    key in original_metrics and key in repaired_metrics
                    and digest(original_metrics[key]) != digest(repaired_metrics[key])
                ),
            })
            final_metrics_records.append(selected)

        source_selection = pd.DataFrame(source_rows)
        final_metrics = index_unique(final_metrics_records, metric_key, "final metric")

        overlap = set(original_metrics) & set(repaired_metrics)
        changed = source_selection["overlap_changed"].sum()
        print("Union of unique keys:", len(all_keys))
        print("Overlap keys:", len(overlap))
        print("Changed repaired rows on overlap:", int(changed))
        print("Selected source counts:")
        display(source_selection["selected_source"].value_counts().rename_axis("source").to_frame("rows"))

        if len(all_keys) != len(EXPECTED_MODELS) * EXPECTED_ATTEMPTS_PER_MODEL:
            print("WARNING: the union does not contain the expected 1,800 unique attempts.")
        '''
    ),
    md("## 4. Apply the frozen final filtering policy"),
    code(
        r'''
        repaired_primary = repaired.get("paper_primary_metrics.csv", pd.DataFrame())
        eligibility_by_agent = {}
        exclusion_by_agent = {}
        primary_by_agent = {}
        if not repaired_primary.empty and "agent" in repaired_primary:
            for row in repaired_primary.to_dict("records"):
                agent = row["agent"]
                primary_by_agent[agent] = row
                eligibility_by_agent[agent] = bool(row.get("trace_metric_eligible", True))
                exclusion_by_agent[agent] = str(row.get("exclusion_reason", "") or "")

        # Explicitly preserve the predeclared GPT-OSS protocol exclusion even if an older
        # bundle lacks paper_primary_metrics.csv.
        for agent in EXPECTED_MODELS:
            if agent in TRACE_EXCLUDED_AGENTS:
                eligibility_by_agent[agent] = False
                exclusion_by_agent[agent] = (
                    "provider reasoning is not reliably mapped to visible protocol actions"
                )


        def number(value, default=np.nan):
            if value is None or value == "":
                return default
            try:
                value = float(value)
                return value if math.isfinite(value) else default
            except (TypeError, ValueError):
                return default


        def flat_record(row, source):
            cg = row.get("C_G_V") or {}
            q = row.get("Q") or {}
            token = row.get("T") or {}
            redundancy = row.get("R") or {}
            latency = row.get("L") or {}
            valid = bool(row.get("valid_trajectory", False))
            c = number(cg.get("C_final"))
            g = number(cg.get("G"))
            v = number(cg.get("V"))
            # Invalid attempts are retained in operational metrics with zero outcomes.
            c_oper = 0.0 if not valid else (0.0 if np.isnan(c) else c)
            g_oper = 0.0 if not valid else (0.0 if np.isnan(g) else g)
            v_oper = 0.0 if not valid else (0.0 if np.isnan(v) else v)
            trace_ok = bool(eligibility_by_agent.get(row.get("agent"), True))
            return {
                "agent": row.get("agent"), "problem_id": row.get("problem_id"),
                "dataset": row.get("dataset"), "subject": row.get("subject"),
                "source": source, "valid_trajectory": valid,
                "stop_reason": row.get("stop_reason"), "finished": row.get("finished"),
                "C_final": c, "G": g, "V": v,
                "C_operational_row": c_oper, "G_operational_row": g_oper,
                "V_operational_row": v_oper,
                "Q": number(q.get("Q_step")) if trace_ok and valid else np.nan,
                "PTU": number(token.get("PTU")) if trace_ok and valid else np.nan,
                "R_harmful": number(redundancy.get("R_harmful")) if trace_ok and valid else np.nan,
                "E_optional_row": number(row.get("E_optional")) if trace_ok and valid else np.nan,
                "T_billable": number(token.get("T_billable"), 0.0),
                "T_api_input": number(token.get("T_api_input"), 0.0),
                "T_api_output": number(token.get("T_api_output"), 0.0),
                "T_hidden_reasoning": number(token.get("T_hidden_reasoning"), 0.0),
                "T_useful": number(token.get("T_useful"), 0.0),
                "T_pre_waste": number(token.get("T_pre_waste"), 0.0),
                "T_post_waste": number(token.get("T_post_waste"), 0.0),
                "T_answer_reporting": number(token.get("T_answer_reporting"), 0.0),
                "latency_sec": number(latency.get("L_trajectory_sec"), number(row.get("elapsed_sec"), np.nan)),
                "trace_metric_eligible": trace_ok,
                "trace_exclusion_reason": "" if trace_ok else exclusion_by_agent.get(row.get("agent"), ""),
                "correctness_source": cg.get("correctness_source"),
                "G_level": cg.get("G_level"),
            }


        source_by_key = {
            (row["agent"], row["problem_id"]): row["selected_source"]
            for row in source_rows
        }
        final_frame = pd.DataFrame([
            flat_record(row, source_by_key[metric_key(row)])
            for row in final_metrics_records
        ])
        operational_frame = final_frame.copy()
        trace_frame = final_frame[final_frame["trace_metric_eligible"]].copy()

        print("Final unique attempts:", len(final_frame))
        print("Operational models:", sorted(final_frame["agent"].dropna().unique()))
        print("Trace-comparable models:", sorted(trace_frame["agent"].dropna().unique()))
        display(final_frame.groupby("agent").agg(
            attempted=("problem_id", "size"),
            valid=("valid_trajectory", "sum"),
            trace_eligible=("trace_metric_eligible", "first"),
        ).reset_index())
        '''
    ),
    md("## 5. Final core metrics"),
    code(
        r'''
        def mean_or_nan(values):
            values = pd.Series(values, dtype="float64").dropna()
            return float(values.mean()) if len(values) else np.nan


        def recorded_primary_value(agent, field, fallback=np.nan):
            value = primary_by_agent.get(agent, {}).get(field, fallback)
            return number(value, fallback)


        summary_rows = []
        for agent, group in operational_frame.groupby("agent", sort=True):
            valid = group[group["valid_trajectory"]]
            trace = group[group["trace_metric_eligible"] & group["valid_trajectory"]]
            summary_rows.append({
                "agent": agent,
                "attempted_n": int(len(group)),
                "valid_n": int(len(valid)),
                "coverage": float(len(valid) / max(len(group), 1)),
                "format_adherence": recorded_primary_value(
                    agent, "format_adherence", len(valid) / max(len(group), 1)
                ),
                "infrastructure_failure_rate": float(
                    group["stop_reason"].isin({"api_error", "worker_exception", "trajectory_timeout"}).mean()
                ),
                "C_operational": float(group["C_operational_row"].mean()),
                "C_valid": mean_or_nan(valid["C_final"]),
                "G_operational": float(group["G_operational_row"].mean()),
                "G_valid": mean_or_nan(valid["G"]),
                "V_operational": float(group["V_operational_row"].mean()),
                "V_valid": mean_or_nan(valid["V"]),
                "Q": mean_or_nan(trace["Q"]),
                "PTU": mean_or_nan(trace["PTU"]),
                "R_harmful": mean_or_nan(trace["R_harmful"]),
                "E_optional": mean_or_nan(trace["E_optional_row"]),
                "total_billable_tokens": float(group["T_billable"].sum()),
                "avg_billable_tokens_per_attempt": float(group["T_billable"].mean()),
                "tokens_per_correct_solve": float(
                    group["T_billable"].sum() / max(group["C_operational_row"].sum(), 1e-12)
                ),
                "tokens_per_verified_solve": float(
                    group["T_billable"].sum() / max(group["V_operational_row"].sum(), 1e-12)
                ),
                "avg_latency_sec": mean_or_nan(group["latency_sec"]),
                "p95_latency_sec": float(group["latency_sec"].quantile(0.95)),
                "trace_metric_eligible": bool(eligibility_by_agent.get(agent, True)),
                "trace_exclusion_reason": exclusion_by_agent.get(agent, ""),
            })

        core_summary = pd.DataFrame(summary_rows)
        paper_core = core_summary[core_summary["trace_metric_eligible"]].copy()
        print("Operational summary, including GPT-OSS:")
        display(core_summary.round(4))
        print("Trace-comparable summary for headline multidimensional results:")
        display(paper_core.round(4))
        '''
    ),
    md("## 6. Source and evaluator audit"),
    code(
        r'''
        repair_audit = source_selection.groupby("agent", as_index=False).agg(
            unique_keys=("problem_id", "size"),
            overlap_keys=("present_original", "sum"),
            repaired_present=("present_repaired", "sum"),
            repaired_changed=("overlap_changed", "sum"),
        )
        repair_audit["repaired_changed"] = repair_audit["repaired_changed"].astype(int)

        evaluator_coverage = repaired.get("evaluator_coverage.csv", pd.DataFrame())
        if not evaluator_coverage.empty:
            display(evaluator_coverage)
        display(repair_audit)
        '''
    ),
    md("## 7. Advanced analyses"),
    code(
        r'''
        rng = np.random.default_rng(RANDOM_SEED)
        bootstrap_rows = []
        bootstrap_metrics = ["C_operational_row", "V_operational_row", "Q", "PTU", "R_harmful"]
        for agent, group in operational_frame.groupby("agent", sort=True):
            eligible = bool(eligibility_by_agent.get(agent, True))
            for metric_name in bootstrap_metrics:
                if metric_name in {"Q", "PTU", "R_harmful"} and not eligible:
                    continue
                values = group[metric_name].dropna().to_numpy(dtype=float)
                if not len(values):
                    continue
                samples = rng.choice(values, size=(BOOTSTRAP_REPLICATES, len(values)), replace=True).mean(axis=1)
                bootstrap_rows.append({
                    "agent": agent, "metric": metric_name,
                    "estimate": float(values.mean()),
                    "ci_low": float(np.quantile(samples, 0.025)),
                    "ci_high": float(np.quantile(samples, 0.975)),
                    "n": int(len(values)),
                })
        bootstrap_ci = pd.DataFrame(bootstrap_rows)

        latency_percentiles = (
            operational_frame.groupby("agent")["latency_sec"]
            .quantile([0.50, 0.90, 0.95, 0.99])
            .unstack()
            .rename(columns={0.50: "p50", 0.90: "p90", 0.95: "p95", 0.99: "p99"})
            .reset_index()
        )

        subject_verified = (
            trace_frame.groupby(["subject", "agent"], dropna=False)
            .agg(
                n=("problem_id", "size"),
                V=("V", "mean"), C=("C_final", "mean"), Q=("Q", "mean"),
            )
            .reset_index()
        )

        grounding_levels = (
            operational_frame[operational_frame["trace_metric_eligible"]]
            .groupby(["agent", "G_level"], dropna=False)
            .size().rename("n").reset_index()
        )
        grounding_levels["rate"] = grounding_levels["n"] / grounding_levels.groupby("agent")["n"].transform("sum")

        step_label_rates = pd.DataFrame()
        if "Q" in trace_frame:
            step_label_rates = trace_frame.groupby("agent")["Q"].agg(["count", "mean"]).reset_index()

        # A point is Pareto-efficient when no other trace-comparable model is both cheaper
        # and at least as successful, with one strict improvement.
        pareto = paper_core[["agent", "avg_billable_tokens_per_attempt", "V_operational"]].copy()
        pareto["pareto_efficient"] = True
        for i, left in pareto.iterrows():
            for j, right in pareto.iterrows():
                if i == j:
                    continue
                dominates = (
                    right["avg_billable_tokens_per_attempt"] <= left["avg_billable_tokens_per_attempt"]
                    and right["V_operational"] >= left["V_operational"]
                    and (
                        right["avg_billable_tokens_per_attempt"] < left["avg_billable_tokens_per_attempt"]
                        or right["V_operational"] > left["V_operational"]
                    )
                )
                if dominates:
                    pareto.loc[i, "pareto_efficient"] = False
                    break

        print("Bootstrap confidence intervals:")
        display(bootstrap_ci.round(4))
        print("Latency percentiles:")
        display(latency_percentiles.round(3))
        '''
    ),
    md("## 8. Plots"),
    code(
        r'''
        sns.set_theme(style="whitegrid")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        plot_all = core_summary.set_index("agent")
        plot_all[["C_operational", "V_operational"]].plot.bar(ax=axes[0], ylim=(0, 1), title="Operational correctness and verified success")
        plot_paper = paper_core.set_index("agent")
        plot_paper[["G_operational", "Q", "PTU"]].plot.bar(ax=axes[1], ylim=(0, 1), title="Trace-comparable metrics")
        axes[0].set_ylabel("rate"); axes[1].set_ylabel("score")
        fig.tight_layout()
        core_plot = FIGURE_ROOT / "final_core_metrics.png"
        fig.savefig(core_plot, dpi=180, bbox_inches="tight")
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 6))
        for agent, group in operational_frame.groupby("agent"):
            values = np.sort(group["latency_sec"].dropna().to_numpy())
            if len(values):
                ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=agent)
        ax.set(title="Trajectory latency distributions", xlabel="latency (seconds)", ylabel="empirical CDF")
        ax.legend()
        fig.tight_layout()
        latency_plot = FIGURE_ROOT / "latency_ecdf.png"
        fig.savefig(latency_plot, dpi=180, bbox_inches="tight")
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(paper_core["avg_billable_tokens_per_attempt"], paper_core["V_operational"])
        for row in paper_core.to_dict("records"):
            ax.annotate(row["agent"], (row["avg_billable_tokens_per_attempt"], row["V_operational"]))
        ax.set(title="Cost versus verified success: trace-comparable models", xlabel="average billable tokens per attempt", ylabel="V")
        fig.tight_layout()
        pareto_plot = FIGURE_ROOT / "cost_verified_pareto.png"
        fig.savefig(pareto_plot, dpi=180, bbox_inches="tight")
        plt.show()
        '''
    ),
    md("## 9. Export final portable metrics"),
    code(
        r'''
        def write_jsonl(path, rows):
            with Path(path).open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")


        final_frame.to_csv(OUTPUT_ROOT / "final_metrics_flat.csv", index=False)
        core_summary.to_csv(OUTPUT_ROOT / "core_metrics_all_models.csv", index=False)
        paper_core.to_csv(OUTPUT_ROOT / "core_metrics_trace_comparable.csv", index=False)
        source_selection.to_csv(OUTPUT_ROOT / "source_selection_audit.csv", index=False)
        repair_audit.to_csv(OUTPUT_ROOT / "repair_overlap_audit.csv", index=False)
        bootstrap_ci.to_csv(OUTPUT_ROOT / "bootstrap_ci.csv", index=False)
        latency_percentiles.to_csv(OUTPUT_ROOT / "latency_percentiles.csv", index=False)
        subject_verified.to_csv(OUTPUT_ROOT / "subject_verified.csv", index=False)
        grounding_levels.to_csv(OUTPUT_ROOT / "grounding_levels.csv", index=False)
        pareto.to_csv(OUTPUT_ROOT / "pareto_summary.csv", index=False)
        write_jsonl(OUTPUT_ROOT / "final_metrics.jsonl", final_metrics_records)

        final_manifest = {
            "artifact_type": "strive_final_metrics_two_bundle_pool",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sources": {"original": ORIGINAL_BUNDLE, "repaired": REPAIRED_BUNDLE},
            "deduplication_key": ["agent", "problem_id"],
            "selection_rule": "repaired record wins on overlap; original is fallback only",
            "unique_attempts": int(len(final_frame)),
            "expected_attempts": int(len(EXPECTED_MODELS) * EXPECTED_ATTEMPTS_PER_MODEL),
            "operational_models": sorted(final_frame["agent"].dropna().unique()),
            "trace_comparable_models": sorted(paper_core["agent"].dropna().unique()),
            "trace_excluded_models": sorted(TRACE_EXCLUDED_AGENTS),
            "trace_exclusion_policy": exclusion_by_agent,
            "judge": repaired["manifest"].get("judge", {}),
            "files": sorted(path.name for path in OUTPUT_ROOT.iterdir() if path.is_file()),
        }
        (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(final_manifest, indent=2), encoding="utf-8")
        export_zip = Path(shutil.make_archive(str(OUTPUT_ROOT), "zip", root_dir=OUTPUT_ROOT))
        print("Final portable metrics ZIP:", export_zip)
        if FileLink is not None:
            display(FileLink(str(export_zip)))
        '''
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

for index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue
    source_text = "".join(cell["source"])
    source_text = "\n".join(
        line for line in source_text.splitlines()
        if not line.lstrip().startswith(("%", "!"))
    )
    ast.parse(source_text, filename=f"cell_{index}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {OUTPUT}")
