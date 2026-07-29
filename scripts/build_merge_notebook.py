import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "strive-merge-generation-archives.ipynb"


def markdown(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip() + "\n",
    }


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


cells = [
    markdown(
        """
        # STRIVE Canonical Generation Merge

        This notebook combines the three 100-problem generation batches and all later
        rerun snapshots into one canonical 300-problem generation bundle.

        **Important merge policy:** archives are ordered by their manifest timestamp.
        A later archive is treated as a newer version of the same `(agent_name,
        problem_id)` record. Identical records are not duplicated; changed records are
        replaced by the newer version. The notebook never selects a trajectory because
        its answer is correct, so the merge cannot create best-of-N or cherry-picking
        bias.

        The two rerun ZIPs are full 1,800-record snapshots. They must not be concatenated
        directly with the three 600-record batches. This notebook resolves those overlaps
        and exports exactly one record per model-problem pair.
        """
    ),
    markdown("## 1. Configuration"),
    code(
        r'''
        from collections import Counter, defaultdict
        from datetime import datetime, timezone
        from pathlib import Path
        import csv
        import hashlib
        import json
        import os
        import re
        import shutil
        import zipfile

        try:
            import pandas as pd
            from IPython.display import FileLink, display
        except ImportError as exc:
            raise RuntimeError("This notebook requires pandas and IPython.") from exc


        ZIP_INPUTS = [
            "/path/to/your/artifacts/paper_math200_olympiad100_v11_batch01_generation_20260713T020241Z.zip",
            "/path/to/your/artifacts/paper_math200_olympiad100_v11_batch02_generation_20260713T123900Z.zip",
            "/path/to/your/artifacts/paper_math200_olympiad100_v11_batch03_generation_20260713T163714Z.zip",
            "/path/to/your/artifacts/paper_math200_olympiad100_v11_glm_minimax_infrastructure_rerun_merged_generation_20260713T222059Z.zip",
            "/path/to/your/artifacts/paper_math200_olympiad100_v11_glm_minimax_infrastructure_rerun_merged_generation_20260714T130853Z.zip",
        ]

        EXPECTED_MODELS = [
            "ministral-14b",
            "glm-5.2",
            "minimax-m3",
            "nemotron-3-nano",
            "gpt-oss-20b",
            "gpt-5-nano",
        ]
        EXPECTED_PROBLEMS = 300
        EXPECTED_TRAJECTORIES = EXPECTED_PROBLEMS * len(EXPECTED_MODELS)
        VERIFY_SOURCE_CHECKSUMS = True
        FAIL_ON_INFRASTRUCTURE_ERRORS = False
        INFRASTRUCTURE_STOPS = {"api_error", "worker_exception", "trajectory_timeout"}
        PROTOCOL_STOPS = {"format_error_limit", "answer_only_violation_limit"}

        WORK_ROOT = Path(
            "/kaggle/working" if Path("/kaggle/working").exists() else Path.cwd()
        ).resolve()
        EXPORT_ROOT = WORK_ROOT / "strive_canonical_merge"
        EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

        if not ZIP_INPUTS:
            raise RuntimeError("Add the generation ZIP paths to ZIP_INPUTS.")
        for value in ZIP_INPUTS:
            if not Path(value).expanduser().exists():
                raise FileNotFoundError(Path(value).expanduser())
        print("Configured source archives:", len(ZIP_INPUTS))
        '''
    ),
    markdown("## 2. Safe archive readers and validation helpers"),
    code(
        r'''
        def canonical_json_bytes(value):
            return json.dumps(
                value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")


        def record_digest(value):
            return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


        def parse_jsonl_bytes(payload, label):
            records = []
            for line_number, raw_line in enumerate(payload.decode("utf-8").splitlines(), 1):
                if not raw_line.strip():
                    continue
                try:
                    value = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid JSON in {label}, line {line_number}") from exc
                if not isinstance(value, dict):
                    raise RuntimeError(f"Expected object in {label}, line {line_number}")
                records.append(value)
            return records


        def find_member(archive, basename, required=True):
            matches = [name for name in archive.namelist() if Path(name).name == basename]
            if len(matches) == 1:
                return matches[0]
            if not matches and not required:
                return None
            raise RuntimeError(
                f"Expected exactly one {basename} in archive; found {len(matches)}"
            )


        def parse_created_at(value, fallback_path):
            if value:
                text = str(value).strip().replace("Z", "+00:00")
                try:
                    parsed = datetime.fromisoformat(text)
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    return parsed.astimezone(timezone.utc)
                except ValueError:
                    pass
            match = re.search(r"(20\d{6}T\d{6}Z)", fallback_path.name)
            if match:
                return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(
                    tzinfo=timezone.utc
                )
            return datetime.fromtimestamp(fallback_path.stat().st_mtime, tz=timezone.utc)


        def trajectory_key(record):
            agent = str(record.get("agent_name", "")).strip()
            problem_id = str(record.get("problem_id", "")).strip()
            if not agent or not problem_id:
                raise RuntimeError("Trajectory is missing agent_name or problem_id")
            return agent, problem_id


        def problem_key(record):
            problem_id = str(record.get("id", record.get("problem_id", ""))).strip()
            if not problem_id:
                raise RuntimeError("Problem is missing id/problem_id")
            return problem_id


        def load_generation_zip(value):
            path = Path(value).expanduser().resolve()
            with zipfile.ZipFile(path) as archive:
                manifest_member = find_member(archive, "generation_manifest.json")
                manifest = json.loads(archive.read(manifest_member))
                trajectory_member = find_member(archive, "trajectories.jsonl")
                problem_member = find_member(archive, "problems.jsonl")

                if VERIFY_SOURCE_CHECKSUMS:
                    for name, expected in manifest.get("sha256", {}).items():
                        member = find_member(archive, Path(name).name, required=False)
                        if member is None:
                            raise RuntimeError(f"{path.name}: manifest member missing: {name}")
                        actual = hashlib.sha256(archive.read(member)).hexdigest()
                        if actual != expected:
                            raise RuntimeError(f"{path.name}: checksum mismatch for {name}")

                trajectories = parse_jsonl_bytes(
                    archive.read(trajectory_member), f"{path.name}/trajectories.jsonl"
                )
                problems = parse_jsonl_bytes(
                    archive.read(problem_member), f"{path.name}/problems.jsonl"
                )
                audit_member = find_member(archive, "recovery_audit.json", required=False)
                recovery_audit = (
                    json.loads(archive.read(audit_member)) if audit_member else None
                )

            keys = [trajectory_key(record) for record in trajectories]
            duplicate_keys = [key for key, count in Counter(keys).items() if count > 1]
            if duplicate_keys:
                raise RuntimeError(
                    f"{path.name}: duplicate trajectory keys inside archive: "
                    f"{duplicate_keys[:10]}"
                )

            return {
                "path": path,
                "name": path.name,
                "manifest": manifest,
                "created_at": parse_created_at(manifest.get("created_at"), path),
                "trajectories": trajectories,
                "problems": problems,
                "recovery_audit": recovery_audit,
            }
        '''
    ),
    markdown("## 3. Load archives and build the canonical versioned record set"),
    code(
        r'''
        sources = sorted(
            [load_generation_zip(value) for value in ZIP_INPUTS],
            key=lambda item: (item["created_at"], item["name"]),
        )

        source_rows = []
        for source in sources:
            stops = Counter(
                str(item.get("stop_reason", "")) for item in source["trajectories"]
            )
            source_rows.append({
                "archive": source["name"],
                "created_at": source["created_at"].isoformat(),
                "trajectories": len(source["trajectories"]),
                "unique_problems": len({trajectory_key(x)[1] for x in source["trajectories"]}),
                "problem_rows": len(source["problems"]),
                "explicit_final": stops.get("explicit_final_answer", 0),
                "infrastructure_failures": sum(stops.get(reason, 0) for reason in INFRASTRUCTURE_STOPS),
                "protocol_failures": sum(stops.get(reason, 0) for reason in PROTOCOL_STOPS),
            })
        source_summary = pd.DataFrame(source_rows)
        display(source_summary)

        canonical_trajectories = {}
        selected_source = {}
        problems_by_id = {}
        problem_digests = {}
        merge_events = []

        for source in sources:
            for problem in source["problems"]:
                key = problem_key(problem)
                digest = record_digest(problem)
                if key in problem_digests and problem_digests[key] != digest:
                    raise RuntimeError(
                        f"Conflicting definitions for problem {key} in {source['name']}"
                    )
                problems_by_id.setdefault(key, problem)
                problem_digests.setdefault(key, digest)

            for record in source["trajectories"]:
                key = trajectory_key(record)
                new_digest = record_digest(record)
                if key not in canonical_trajectories:
                    canonical_trajectories[key] = record
                    selected_source[key] = source["name"]
                    merge_events.append({
                        "agent": key[0], "problem_id": key[1], "event": "added",
                        "from_archive": None, "to_archive": source["name"],
                        "old_stop": None, "new_stop": record.get("stop_reason"),
                    })
                    continue

                old_record = canonical_trajectories[key]
                old_digest = record_digest(old_record)
                if new_digest == old_digest:
                    continue

                # Versioning rule: later snapshot replaces the earlier version. This is
                # intentionally independent of correctness and final-answer content.
                old_source = selected_source[key]
                canonical_trajectories[key] = record
                selected_source[key] = source["name"]
                merge_events.append({
                    "agent": key[0], "problem_id": key[1], "event": "replaced",
                    "from_archive": old_source, "to_archive": source["name"],
                    "old_stop": old_record.get("stop_reason"),
                    "new_stop": record.get("stop_reason"),
                })

        model_order = {name: index for index, name in enumerate(EXPECTED_MODELS)}
        problem_order = {problem_id: index for index, problem_id in enumerate(problems_by_id)}
        canonical_records = sorted(
            canonical_trajectories.values(),
            key=lambda item: (
                problem_order.get(str(item.get("problem_id")), 10**9),
                model_order.get(str(item.get("agent_name")), 10**9),
                str(item.get("problem_id")),
                str(item.get("agent_name")),
            ),
        )
        canonical_problems = list(problems_by_id.values())

        print("Canonical trajectory rows:", len(canonical_records))
        print("Canonical problem rows:", len(canonical_problems))
        print("Changed-record replacements:", sum(x["event"] == "replaced" for x in merge_events))
        '''
    ),
    markdown("## 4. Coverage, failure audit, and paper-run gates"),
    code(
        r'''
        keys = [trajectory_key(record) for record in canonical_records]
        if len(keys) != len(set(keys)):
            raise RuntimeError("Canonical records still contain duplicate model-problem keys")

        actual_models = sorted({agent for agent, _ in keys})
        actual_problem_ids = {problem_id for _, problem_id in keys}
        expected_grid = {
            (agent, problem_id)
            for agent in EXPECTED_MODELS
            for problem_id in actual_problem_ids
        }
        missing_grid_keys = sorted(expected_grid - set(keys))
        unexpected_grid_keys = sorted(set(keys) - expected_grid)

        if actual_models != sorted(EXPECTED_MODELS):
            raise RuntimeError(
                f"Model mismatch. Expected {sorted(EXPECTED_MODELS)}, got {actual_models}"
            )
        if len(actual_problem_ids) != EXPECTED_PROBLEMS:
            raise RuntimeError(
                f"Expected {EXPECTED_PROBLEMS} problems, got {len(actual_problem_ids)}"
            )
        if len(canonical_records) != EXPECTED_TRAJECTORIES:
            raise RuntimeError(
                f"Expected {EXPECTED_TRAJECTORIES} trajectories, got {len(canonical_records)}"
            )
        if missing_grid_keys or unexpected_grid_keys:
            raise RuntimeError(
                f"Incomplete model-problem grid: missing={missing_grid_keys[:10]}, "
                f"unexpected={unexpected_grid_keys[:10]}"
            )

        health_rows = []
        unresolved_infrastructure = []
        unresolved_protocol = []
        for agent in EXPECTED_MODELS:
            rows = [item for item in canonical_records if item["agent_name"] == agent]
            stops = Counter(str(item.get("stop_reason", "")) for item in rows)
            infra = [item for item in rows if item.get("stop_reason") in INFRASTRUCTURE_STOPS]
            protocol = [item for item in rows if item.get("stop_reason") in PROTOCOL_STOPS]
            unresolved_infrastructure.extend({
                "agent": agent,
                "problem_id": item["problem_id"],
                "stop_reason": item.get("stop_reason"),
                "selected_source": selected_source[(agent, item["problem_id"])],
            } for item in infra)
            unresolved_protocol.extend({
                "agent": agent,
                "problem_id": item["problem_id"],
                "stop_reason": item.get("stop_reason"),
                "selected_source": selected_source[(agent, item["problem_id"])],
            } for item in protocol)
            health_rows.append({
                "agent": agent,
                "records": len(rows),
                "finished": sum(bool(item.get("finished")) for item in rows),
                "completion_rate": sum(bool(item.get("finished")) for item in rows) / len(rows),
                "explicit_final_answer": stops.get("explicit_final_answer", 0),
                "infrastructure_failures": len(infra),
                "protocol_failures": len(protocol),
                "stop_reasons": json.dumps(dict(sorted(stops.items())), sort_keys=True),
            })

        generation_health = pd.DataFrame(health_rows)
        display(generation_health)
        print("Unresolved infrastructure failures:", len(unresolved_infrastructure))
        print("Unresolved format/protocol failures:", len(unresolved_protocol))
        if unresolved_infrastructure:
            display(pd.DataFrame(unresolved_infrastructure))
            if FAIL_ON_INFRASTRUCTURE_ERRORS:
                raise RuntimeError(
                    "Infrastructure failures remain. Rerun them before final paper scoring."
                )
        '''
    ),
    markdown("## 5. Export one portable canonical generation ZIP"),
    code(
        r'''
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = f"paper_math200_olympiad100_v11_canonical_generation_{timestamp}"
        export_dir = EXPORT_ROOT / run_name
        export_dir.mkdir(parents=True, exist_ok=False)


        def write_jsonl(path, records):
            with path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


        trajectories_path = export_dir / "trajectories.jsonl"
        problems_path = export_dir / "problems.jsonl"
        health_path = export_dir / "generation_health.csv"
        audit_path = export_dir / "merge_audit.json"
        manifest_path = export_dir / "generation_manifest.json"

        write_jsonl(trajectories_path, canonical_records)
        write_jsonl(problems_path, canonical_problems)
        generation_health.to_csv(health_path, index=False)

        merge_audit = {
            "merge_policy": "chronological version replacement; never best-of-answer selection",
            "sources": source_rows,
            "source_order": [source["name"] for source in sources],
            "merge_events": merge_events,
            "selected_source_counts": dict(Counter(selected_source.values())),
            "canonical_trajectory_count": len(canonical_records),
            "canonical_problem_count": len(canonical_problems),
            "unresolved_infrastructure": unresolved_infrastructure,
            "unresolved_protocol": unresolved_protocol,
        }
        audit_path.write_text(json.dumps(merge_audit, indent=2), encoding="utf-8")

        data_files = [
            "trajectories.jsonl",
            "problems.jsonl",
            "generation_health.csv",
            "merge_audit.json",
        ]
        latest_manifest = sources[-1]["manifest"]
        manifest = {
            "artifact_type": "canonical_merged_generation",
            "raw_data_schema_version": latest_manifest.get("raw_data_schema_version", 1),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_name": run_name,
            "merge_policy": merge_audit["merge_policy"],
            "source_archives": [source["name"] for source in sources],
            "models": latest_manifest.get("models", []),
            "trajectory_count": len(canonical_records),
            "problem_count": len(canonical_problems),
            "data_files": data_files,
            "sha256": {
                name: hashlib.sha256((export_dir / name).read_bytes()).hexdigest()
                for name in data_files
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        canonical_zip = Path(shutil.make_archive(
            str(export_dir), "zip", root_dir=export_dir
        ))
        print("Canonical generation directory:", export_dir)
        print("Canonical generation ZIP:", canonical_zip)
        display(FileLink(str(canonical_zip)))
        '''
    ),
    markdown(
        """
        ## 6. Exact next steps

        ### Core metrics

        Open `strive-metrics-from-trajectories.ipynb` and set:

        ```python
        TRAJECTORY_INPUTS = [
            "/path/to/paper_math200_olympiad100_v11_canonical_generation_...zip"
        ]
        RUN_NAME = "paper_math200_olympiad100_v11_metrics_300"
        ```

        Then run the notebook from the configuration cell through the expensive metric
        cell and export cell. Use only the canonical ZIP; do not list the five source ZIPs
        again.

        For a final paper run, set `FAIL_ON_INFRASTRUCTURE_ERRORS = True`. With the current
        archives, this gate is expected to fail until the remaining infrastructure-error
        trajectories are rerun. Keeping it `False` produces operational scores that count
        those failures as failed attempts.

        ### Advanced analysis

        After the core metrics notebook exports its metrics ZIP, open
        `strive-advanced-metric-analysis.ipynb` and set:

        ```python
        RESULT_INPUT = "/path/to/paper_math200_olympiad100_v11_metrics_300_...zip"
        RUN_LABEL = "paper_math200_olympiad100_v11_300"
        ```

        Run all advanced-analysis cells. This stage uses the saved trajectories, PRM
        scores, and metrics; it does not regenerate model trajectories or reload PRMs.
        """
    ),
]

for index, cell in enumerate(cells):
    cell["id"] = f"strive-merge-{index:02d}"


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
