#!/usr/bin/env python3
"""Build the STRIVE judge/critic-only recovery notebook."""

import ast
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "strive-judge-critic-repair-from-metrics.ipynb"


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
        # STRIVE: Judge and Critic Repair From Saved Metrics

        This notebook repairs the failed NVIDIA NIM evaluator stage in an existing STRIVE
        metrics ZIP. It does **not** regenerate trajectories, execute the Python sandbox,
        load either PRM, recompute embeddings, or rerun deterministic provenance tracing.

        It performs only the work affected by the missing evaluator credential:

        1. adjudicate symbolic-inconclusive final answers with Step 3.5 Flash;
        2. rerun the already-selected near-boundary step critic calls with Step 3.5 Flash;
        3. recompute `C_final`, `V`, `Q`, token utility, and optional `E` from cached evidence;
        4. export a repaired portable metrics ZIP and paper-facing audit tables.

        Every evaluator response is checkpointed. Interrupted runs resume without repeating
        successful calls. Failed calls remain unresolved and are never silently scored as wrong.
        """
    ),
    md("## 1. Install the lightweight repair dependencies"),
    code(
        r'''
        %pip install -q "openai>=1.68,<3" pandas numpy matplotlib seaborn scipy tqdm
        '''
    ),
    md("## 2. Imports and repair configuration"),
    code(
        r'''
        import copy
        import hashlib
        import json
        import math
        import os
        import re
        import shutil
        import threading
        import time
        import zipfile
        from collections import Counter, defaultdict
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from datetime import datetime, timezone
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from openai import OpenAI
        from tqdm.auto import tqdm

        try:
            from IPython.display import FileLink, display
        except Exception:
            FileLink = None


        # Accepts either the literal ZIP or Kaggle's automatically extracted directory.
        METRICS_INPUT = (
            "/path/to/your/artifacts/"
            "paper_math200_olympiad100_v11_five_zip_metrics_20260715T035532Z.zip"
        )

        RUN_NAME = "paper_math200_olympiad100_v11_step35_judge_critic_repaired"
        WORK_ROOT = Path(
            "/kaggle/working" if Path("/kaggle/working").exists() else Path.cwd()
        ).resolve()
        OUTPUT_ROOT = WORK_ROOT / "strive_judge_critic_repair"
        CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

        # Use the same Step 3.5 Flash evaluator for correctness and step criticism.
        # Keeping one model makes the evaluator protocol consistent across both phases.
        CORRECTNESS_MODEL = "stepfun-ai/step-3.5-flash"
        CRITIC_MODEL = "stepfun-ai/step-3.5-flash"
        # Six independent accounts run as six fixed parallel lanes.
        JUDGE_RPM_PER_KEY = 40
        REQUIRED_EVALUATOR_KEYS = 6
        MAX_API_ATTEMPTS = 5
        REQUEST_TIMEOUT_SEC = 120
        RETRY_BASE_SEC = 3.0
        RATE_LIMIT_COOLDOWN_SEC = 60.0
        # Step 3.5 Flash is a reasoning model. A 256-token cap can truncate its
        # reasoning before it emits the requested JSON object.
        EVALUATOR_TEMPERATURE = 1.0
        EVALUATOR_TOP_P = 0.9
        EVALUATOR_MAX_TOKENS = 16384
        CRITIC_BOUNDARY = 0.08
        LOW_CONFIDENCE_AUDIT_THRESHOLD = 0.70

        # Predeclared trace-comparability rule used for the paper-facing table.
        TRACE_MIN_FORMAT_ADHERENCE = 0.95
        TRACE_MAX_UNMAPPED_REASONING_STEP_RATE = 0.05

        # Keys remain hidden in notebook output. Kaggle secrets with these names are also read.
        NVIDIA_KEYS = {
            "NVIDIA_API_KEY_1": "",
            "NVIDIA_API_KEY_2": "",
            "NVIDIA_API_KEY_3": "",
            "NVIDIA_API_KEY_4": "",
            "NVIDIA_API_KEY_5": "",
            "NVIDIA_API_KEY_6": "",
        }


        def load_nvidia_keys():
            if Path("/kaggle/working").exists():
                try:
                    from kaggle_secrets import UserSecretsClient
                    secrets = UserSecretsClient()
                    for name in NVIDIA_KEYS:
                        if not os.environ.get(name):
                            try:
                                value = secrets.get_secret(name)
                                if value:
                                    os.environ[name] = value
                            except Exception:
                                pass
                except Exception:
                    pass
            for name, value in NVIDIA_KEYS.items():
                if str(value).strip():
                    os.environ[name] = str(value).strip()
            configured = [
                (name, os.environ[name].strip())
                for name in NVIDIA_KEYS
                if os.environ.get(name, "").strip()
            ]
            if len({value for _, value in configured}) != len(configured):
                raise RuntimeError("Each NVIDIA evaluator lane must use a distinct API key.")
            print(f"Configured NVIDIA evaluator keys: {len(configured)}/6; values are hidden")
            return configured


        CONFIGURED_NVIDIA_KEYS = load_nvidia_keys()
        print("Input:", METRICS_INPUT)
        print("Output root:", OUTPUT_ROOT)
        '''
    ),
    md("## 3. Load and checksum-verify the existing metrics bundle"),
    code(
        r'''
        REQUIRED_FILES = {
            "manifest.json", "metrics.jsonl", "trajectories.jsonl",
            "problems.jsonl", "prm_scores.json",
        }


        def parse_jsonl_bytes(payload, label):
            rows = []
            for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid JSON in {label}, line {line_number}") from exc
                if not isinstance(row, dict):
                    raise RuntimeError(f"Expected a JSON object in {label}, line {line_number}")
                rows.append(row)
            return rows


        def unique_member(archive, basename, required=True):
            matches = [name for name in archive.namelist() if Path(name).name == basename]
            if len(matches) == 1:
                return matches[0]
            if not matches and not required:
                return None
            raise RuntimeError(f"Expected one {basename}; found {len(matches)}")


        def find_bundle_directory(path):
            metric_files = [candidate for candidate in path.rglob("metrics.jsonl") if candidate.is_file()]
            valid_roots = [
                candidate.parent for candidate in metric_files
                if all((candidate.parent / name).is_file() for name in REQUIRED_FILES)
            ]
            if len(valid_roots) != 1:
                raise RuntimeError(
                    f"Expected one extracted metrics bundle under {path}; found {len(valid_roots)}"
                )
            return valid_roots[0]


        def load_metrics_bundle(value):
            path = Path(value).expanduser().resolve()
            payloads = {}
            if path.is_dir():
                root = find_bundle_directory(path)
                for name in REQUIRED_FILES | {
                    "summary.csv", "invalid_attempts.csv", "generation_health.csv", "merge_audit.json"
                }:
                    candidate = root / name
                    if candidate.is_file():
                        payloads[name] = candidate.read_bytes()
                source_label = str(root)
            elif path.is_file() and path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as archive:
                    for name in REQUIRED_FILES | {
                        "summary.csv", "invalid_attempts.csv", "generation_health.csv", "merge_audit.json"
                    }:
                        member = unique_member(archive, name, required=name in REQUIRED_FILES)
                        if member:
                            payloads[name] = archive.read(member)
                source_label = str(path)
            else:
                raise FileNotFoundError(f"Expected a metrics ZIP or extracted directory: {path}")

            manifest = json.loads(payloads["manifest.json"])
            for name, expected in manifest.get("sha256", {}).items():
                basename = Path(name).name
                if basename in payloads:
                    actual = hashlib.sha256(payloads[basename]).hexdigest()
                    if actual != expected:
                        raise RuntimeError(f"Checksum mismatch for {basename}")

            return {
                "source": source_label,
                "payloads": payloads,
                "manifest": manifest,
                "metrics": parse_jsonl_bytes(payloads["metrics.jsonl"], "metrics.jsonl"),
                "trajectories": parse_jsonl_bytes(payloads["trajectories.jsonl"], "trajectories.jsonl"),
                "problems": parse_jsonl_bytes(payloads["problems.jsonl"], "problems.jsonl"),
                "prm_scores": json.loads(payloads["prm_scores.json"]),
            }


        bundle = load_metrics_bundle(METRICS_INPUT)
        metrics_before = bundle["metrics"]
        metrics = copy.deepcopy(metrics_before)
        trajectories = bundle["trajectories"]
        manifest_before = bundle["manifest"]

        metric_by_key = {(row["agent"], row["problem_id"]): row for row in metrics}
        metric_before_by_key = {(row["agent"], row["problem_id"]): row for row in metrics_before}
        trajectory_by_key = {
            (row["agent_name"], row["problem_id"]): row for row in trajectories
        }

        if len(metric_by_key) != len(metrics):
            raise RuntimeError("Duplicate (agent, problem_id) rows in metrics.jsonl")
        if set(metric_by_key) != set(trajectory_by_key):
            raise RuntimeError("Metrics and trajectories do not contain the same model-problem grid")

        unresolved_before = [
            row for row in metrics if row.get("valid_trajectory") and row["C_G_V"].get("needs_judge")
        ]
        critic_failures_before = [
            (row, index, detail)
            for row in metrics if row.get("valid_trajectory")
            for index, detail in enumerate(row.get("Q", {}).get("step_details", []))
            if detail.get("critic_record", {}).get("error")
        ]

        print("Loaded source:", bundle["source"])
        print("Metric rows:", len(metrics))
        print("Trajectory rows:", len(trajectories))
        print("Cached PRM models:", sorted({
            name for agent in bundle["prm_scores"].values()
            for entry in agent.values() for name in entry.get("by_model", {})
        }))
        print("Unresolved correctness decisions:", len(unresolved_before))
        print("Failed critic calls:", len(critic_failures_before))
        display(pd.Series(Counter(row["agent"] for row in unresolved_before), name="unresolved"))
        '''
    ),
    md(
        """
        ## 4. NVIDIA-only evaluators with key rotation and checkpoints

        Correctness adjudication and step criticism both use Step 3.5 Flash through the
        NVIDIA NIM endpoint and the same six fixed key lanes.
        """
    ),
    code(
        r'''
        def extract_json_object(text):
            text = str(text or "").strip()
            decoder = json.JSONDecoder()
            parsed_candidates = []
            for match in re.finditer(r"\{", text):
                try:
                    candidate, _ = decoder.raw_decode(text[match.start():])
                except Exception:
                    continue
                if isinstance(candidate, dict):
                    parsed_candidates.append(candidate)
            return parsed_candidates[-1] if parsed_candidates else None


        def recover_step35_object(text, schema_validator):
            """Recover the requested field when Step 3.5 emits JSON-like prose."""
            text = str(text or "")
            schema_name = getattr(schema_validator, "__name__", "")
            if schema_name == "correctness_schema":
                matches = re.findall(
                    r"(?:[\"']?correct[\"']?)\s*[:=]\s*(true|false)",
                    text,
                    flags=re.IGNORECASE,
                )
                if matches:
                    return {"correct": matches[-1].lower() == "true"}
            if schema_name == "critic_schema":
                labels = "progressive|neutral_useful|neutral_waste|redundant|regressive"
                matches = re.findall(
                    rf"(?:[\"']?label[\"']?)\s*[:=]\s*[\"']?({labels})[\"']?",
                    text,
                    flags=re.IGNORECASE,
                )
                if matches:
                    return {"label": matches[-1].lower()}
            return None


        def is_retryable_error(exc):
            text = str(exc).lower()
            return any(token in text for token in [
                "429", "rate limit", "resource exhausted", "timeout", "timed out",
                "500", "502", "503", "504", "internal server", "connection",
            ])


        class NvidiaJudgeLane:
            """One sequential request lane permanently bound to one NVIDIA account key."""

            def __init__(self, lane_id, slot, api_key, rpm_per_key):
                self.lane_id = int(lane_id)
                self.slot = slot
                self.client = OpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=api_key,
                    timeout=REQUEST_TIMEOUT_SEC,
                )
                self.interval = 60.0 / max(float(rpm_per_key), 1.0)
                self.last_request_started = 0.0

            def pace(self):
                wait = self.interval - (time.monotonic() - self.last_request_started)
                if wait > 0:
                    time.sleep(wait)
                self.last_request_started = time.monotonic()

            def call_json(self, messages, schema_validator, model_id, extra_body=None):
                errors = []
                for attempt in range(1, MAX_API_ATTEMPTS + 1):
                    self.pace()
                    started = time.perf_counter()
                    try:
                        request_kwargs = {
                            "model": model_id,
                            "messages": messages,
                            "temperature": EVALUATOR_TEMPERATURE,
                            "top_p": EVALUATOR_TOP_P,
                            "max_tokens": EVALUATOR_MAX_TOKENS,
                            "stream": False,
                        }
                        if extra_body is not None:
                            # Keep provider-specific request extensions optional.
                            request_kwargs["extra_body"] = extra_body
                        response = self.client.chat.completions.create(**request_kwargs)
                        message = response.choices[0].message
                        response_candidates = []
                        content = getattr(message, "content", None)
                        if content:
                            response_candidates.append(("content", str(content)))
                        # Depending on the NIM response shape, the structured answer may
                        # be exposed in content or in a reasoning field. Search both, but
                        # never treat either field as trajectory/tool evidence.
                        for field in ("reasoning_content", "reasoning"):
                            value = getattr(message, field, None)
                            if value:
                                response_candidates.append((f"{field}_fallback", str(value)))
                        parsed = None
                        raw = response_candidates[0][1] if response_candidates else "<empty response>"
                        response_channel = response_candidates[0][0] if response_candidates else "missing"
                        for candidate_channel, candidate_raw in response_candidates:
                            candidate = extract_json_object(candidate_raw)
                            if candidate is None:
                                candidate = recover_step35_object(candidate_raw, schema_validator)
                            if candidate is not None:
                                parsed = candidate
                                raw = candidate_raw
                                response_channel = candidate_channel
                                break
                        valid, normalized, schema_error = schema_validator(parsed)
                        if not valid:
                            raise ValueError(f"Invalid evaluator JSON: {schema_error}; raw={raw[:300]!r}")
                        return {
                            "status": "success",
                            "parsed": normalized,
                            "raw": raw,
                            "key_slot": self.slot,
                            "lane_id": self.lane_id,
                            "attempts": attempt,
                            "latency_sec": time.perf_counter() - started,
                            "response_channel": response_channel,
                            "error": "",
                        }
                    except Exception as exc:
                        error = str(exc)
                        errors.append({
                            "attempt": attempt, "key_slot": self.slot,
                            "lane_id": self.lane_id, "error": error,
                        })
                        if is_retryable_error(exc):
                            cooldown = RATE_LIMIT_COOLDOWN_SEC if (
                                "429" in error or "rate" in error.lower() or "resource" in error.lower()
                            ) else RETRY_BASE_SEC * (2 ** (attempt - 1))
                        else:
                            cooldown = RETRY_BASE_SEC * attempt
                        if attempt < MAX_API_ATTEMPTS:
                            time.sleep(cooldown)
                return {
                    "status": "error", "parsed": None, "raw": "", "key_slot": self.slot,
                    "lane_id": self.lane_id,
                    "attempts": MAX_API_ATTEMPTS, "latency_sec": None,
                    "error": errors[-1]["error"] if errors else "unknown_error",
                    "attempt_errors": errors,
                }


        def correctness_schema(value):
            if not isinstance(value, dict) or not isinstance(value.get("correct"), bool):
                return False, None, "expected boolean field 'correct'"
            normalized = {
                "correct": bool(value["correct"]),
                "confidence": float(value.get("confidence", 0.0)),
                "reason": str(value.get("reason", ""))[:500],
            }
            return True, normalized, ""


        CRITIC_LABELS = {
            "progressive": 1.0,
            "neutral_useful": 0.25,
            "neutral_waste": 0.0,
            "redundant": -0.5,
            "regressive": -1.0,
        }


        def critic_schema(value):
            if not isinstance(value, dict) or value.get("label") not in CRITIC_LABELS:
                return False, None, "expected one supported step label"
            normalized = {
                "label": value["label"],
                "confidence": float(value.get("confidence", 0.0)),
                "reason": str(value.get("reason", ""))[:500],
            }
            return True, normalized, ""


        def load_jsonl_cache(path):
            records = {}
            if Path(path).is_file():
                for row in parse_jsonl_bytes(Path(path).read_bytes(), Path(path).name):
                    records[row["task_id"]] = row
            return records


        def append_jsonl(path, row):
            with Path(path).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                handle.flush()


        if len(CONFIGURED_NVIDIA_KEYS) != REQUIRED_EVALUATOR_KEYS:
            raise RuntimeError(
                f"This fast evaluator cell requires exactly {REQUIRED_EVALUATOR_KEYS} distinct "
                f"NVIDIA keys; configured={len(CONFIGURED_NVIDIA_KEYS)}. Fill all six slots in Section 2."
            )
        JUDGE_LANES = [
            NvidiaJudgeLane(index + 1, slot, api_key, JUDGE_RPM_PER_KEY)
            for index, (slot, api_key) in enumerate(CONFIGURED_NVIDIA_KEYS)
        ]
        print(
            "Six NVIDIA lanes ready:",
            "correctness=", CORRECTNESS_MODEL,
            "critic=", CRITIC_MODEL,
            "rpm/key=", JUDGE_RPM_PER_KEY,
        )
        '''
    ),
    md("## 5. Build exactly the missing correctness and critic task queues"),
    code(
        r'''
        def format_step(step):
            parts = [str(step.get("reasoning", "")).strip()]
            if step.get("code"):
                parts.append("Code:\n" + str(step["code"]))
            if step.get("action_type") == "code":
                parts.append("Observation:\n" + str(step.get("observation", "")))
            if step.get("action_type") == "answer":
                parts.append("Final answer: " + str(step.get("final_answer", "")))
            return "\n".join(part for part in parts if part).strip()


        def correctness_messages(metric, trajectory):
            prompt = f"""Determine whether the predicted final answer is mathematically equivalent to the reference.
        Use the problem to interpret notation. Ignore reasoning style and provenance.

        Problem:
        {trajectory.get('problem_text', '')}

        Reference answer:
        {metric.get('gold_answer', trajectory.get('gold_answer', ''))}

        Predicted answer:
        {metric.get('final_answer', trajectory.get('final_answer', ''))}

        Return JSON only: {{"correct": true_or_false, "confidence": number_0_to_1, "reason": "brief"}}"""
            return [{
                "role": "user",
                "content": (
                    "Act as a neutral mathematical answer adjudicator. "
                    "Return only the requested JSON object.\n\n" + prompt
                ),
            }]


        def critic_messages(metric, trajectory, step_index):
            prefix = "\n\n".join(
                f"Step {index + 1}:\n{format_step(step)}"
                for index, step in enumerate(trajectory.get("steps", [])[:step_index + 1])
            )
            prompt = f"""Classify only the final step in this trajectory prefix.
        Labels: progressive, neutral_useful, neutral_waste, redundant, regressive.

        Problem:
        {trajectory.get('problem_text', '')}

        Prefix:
        {prefix}

        Return JSON only: {{"label": "one_label", "confidence": number_0_to_1, "reason": "brief"}}"""
            return [{
                "role": "user",
                "content": (
                    "Judge mathematical process steps independently of prose style. "
                    "Return only the requested JSON object.\n\n" + prompt
                ),
            }]


        correctness_tasks = []
        for metric in metrics:
            if not metric.get("valid_trajectory") or not metric["C_G_V"].get("needs_judge"):
                continue
            key = (metric["agent"], metric["problem_id"])
            correctness_tasks.append({
                "task_id": f"correctness|{key[0]}|{key[1]}",
                "task_type": "correctness", "agent": key[0], "problem_id": key[1],
                "judge_model": CORRECTNESS_MODEL,
                "prompt_hash": hashlib.sha256(
                    json.dumps(correctness_messages(metric, trajectory_by_key[key]), sort_keys=True).encode()
                ).hexdigest(),
            })


        config = manifest_before.get("config", {})
        W_PRM = float(config.get("w_prm", 0.45))
        W_CRITIC = float(config.get("w_critic", 0.25))
        W_TOOL_GAIN = float(config.get("w_tool_gain", 0.15))
        W_REDUNDANCY = float(config.get("w_redundancy", -0.25))
        W_ERROR = float(config.get("w_error", -0.50))
        PROGRESSIVE_THRESHOLD = float(config.get("progressive_threshold", 0.20))
        REGRESSIVE_THRESHOLD = float(config.get("regressive_threshold", -0.20))


        def cached_base_score(detail):
            return float(
                W_PRM * float(detail.get("prm_signal", 0.0))
                + W_TOOL_GAIN * float(detail.get("tool_gain", 0.0))
                + W_REDUNDANCY * float(detail.get("harmful_repeat", 0.0))
                + W_ERROR * float(detail.get("error_flag", 0.0))
            )


        critic_tasks = []
        for metric in metrics:
            if not metric.get("valid_trajectory"):
                continue
            key = (metric["agent"], metric["problem_id"])
            trajectory = trajectory_by_key[key]
            steps = trajectory.get("steps", [])
            for index, detail in enumerate(metric.get("Q", {}).get("step_details", [])):
                action_type = steps[index].get("action_type") if index < len(steps) else ""
                failed_before = bool(detail.get("critic_record", {}).get("error"))
                selected_by_policy = abs(cached_base_score(detail)) < CRITIC_BOUNDARY and action_type != "answer"
                if not detail.get("critic_used") and (failed_before or selected_by_policy):
                    critic_tasks.append({
                        "task_id": f"critic|{key[0]}|{key[1]}|{index + 1}",
                        "task_type": "critic", "agent": key[0], "problem_id": key[1],
                        "judge_model": CRITIC_MODEL,
                        "step_index": index,
                        "prompt_hash": hashlib.sha256(
                            json.dumps(critic_messages(metric, trajectory, index), sort_keys=True).encode()
                        ).hexdigest(),
                    })


        print("Correctness tasks:", len(correctness_tasks))
        print("Critic tasks:", len(critic_tasks))
        print("Expected from the supplied ZIP: correctness=134, critic=677")
        if len(correctness_tasks) != 134 or len(critic_tasks) != 677:
            print("NOTE: counts differ because a different or partially repaired input bundle was loaded.")
        '''
    ),
    md("## 6. Run/resume the Step 3.5 Flash repair calls"),
    code(
        r'''
        def run_task_on_lane(task, lane):
            key = (task["agent"], task["problem_id"])
            if task["task_type"] == "correctness":
                return lane.call_json(
                    correctness_messages(metric_by_key[key], trajectory_by_key[key]),
                    correctness_schema,
                    model_id=CORRECTNESS_MODEL,
                    extra_body=None,
                )
            return lane.call_json(
                critic_messages(metric_by_key[key], trajectory_by_key[key], task["step_index"]),
                critic_schema,
                model_id=CRITIC_MODEL,
                extra_body=None,
            )


        def cache_is_compatible(record, task):
            return bool(
                record.get("status") == "success"
                and record.get("prompt_hash") == task["prompt_hash"]
                and record.get("judge_model") == task["judge_model"]
            )


        legacy_checkpoint_paths = [
            CHECKPOINT_ROOT / "correctness_judge_repair.jsonl",
            CHECKPOINT_ROOT / "step_critic_repair.jsonl",
        ]
        historical_lane_paths = [
            CHECKPOINT_ROOT / f"evaluator_lane_{index + 1}.jsonl"
            for index in range(len(JUDGE_LANES))
        ]
        phase_lane_paths = [
            CHECKPOINT_ROOT / f"correctness_lane_{index + 1}.jsonl"
            for index in range(len(JUDGE_LANES))
        ] + [
            CHECKPOINT_ROOT / f"critic_lane_{index + 1}.jsonl"
            for index in range(len(JUDGE_LANES))
        ]
        repair_cache = {}
        for path in legacy_checkpoint_paths + historical_lane_paths + phase_lane_paths:
            repair_cache.update(load_jsonl_cache(path))
        progress_lock = threading.Lock()
        cache_lock = threading.Lock()

        def run_parallel_phase(tasks, phase_name, checkpoint_prefix):
            tasks = sorted(tasks, key=lambda task: task["task_id"])
            lane_task_lists = [[] for _ in JUDGE_LANES]
            cached_successes = 0
            for index, task in enumerate(tasks):
                if cache_is_compatible(repair_cache.get(task["task_id"], {}), task):
                    cached_successes += 1
                    continue
                lane_task_lists[index % len(JUDGE_LANES)].append(task)

            checkpoint_paths = [
                CHECKPOINT_ROOT / f"{checkpoint_prefix}_lane_{index + 1}.jsonl"
                for index in range(len(JUDGE_LANES))
            ]
            pending_count = sum(map(len, lane_task_lists))
            phase_plan = pd.DataFrame({
                "phase": phase_name,
                "lane_id": [lane.lane_id for lane in JUDGE_LANES],
                "key_slot": [lane.slot for lane in JUDGE_LANES],
                "pending_tasks": [len(items) for items in lane_task_lists],
                "cached_successes": cached_successes,
            })
            print(
                f"{phase_name}: total={len(tasks)}, cached_success={cached_successes}, "
                f"pending={pending_count}, aggregate_limit={JUDGE_RPM_PER_KEY * len(JUDGE_LANES)} RPM"
            )
            display(phase_plan)

            progress = tqdm(total=pending_count, desc=f"{phase_name}: six-key repair", unit="call")

            def run_lane(lane, assigned_tasks, checkpoint_path):
                stats = {
                    "phase": phase_name, "lane_id": lane.lane_id, "key_slot": lane.slot,
                    "assigned": len(assigned_tasks), "success": 0, "error": 0,
                }
                for task in assigned_tasks:
                    try:
                        result = run_task_on_lane(task, lane)
                    except Exception as exc:
                        result = {
                            "status": "error", "parsed": None, "raw": "",
                            "key_slot": lane.slot, "lane_id": lane.lane_id,
                            "attempts": 0, "latency_sec": None, "error": str(exc),
                        }
                    record = {
                        **task, **result,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    }
                    append_jsonl(checkpoint_path, record)
                    with cache_lock:
                        repair_cache[task["task_id"]] = record
                    stats["success" if result.get("status") == "success" else "error"] += 1
                    with progress_lock:
                        progress.update(1)
                        progress.set_postfix({"lane": lane.lane_id}, refresh=False)
                return stats

            lane_stats = []
            if pending_count:
                with ThreadPoolExecutor(max_workers=len(JUDGE_LANES)) as executor:
                    futures = [
                        executor.submit(run_lane, lane, assigned, checkpoint)
                        for lane, assigned, checkpoint in zip(
                            JUDGE_LANES, lane_task_lists, checkpoint_paths
                        )
                    ]
                    for future in as_completed(futures):
                        lane_stats.append(future.result())
            progress.close()
            runtime_summary = pd.DataFrame(lane_stats or phase_plan.to_dict("records"))
            display(runtime_summary.sort_values("lane_id"))
            return phase_plan, runtime_summary


        # Phase 1: finish correctness adjudication before starting critic calls.
        correctness_phase_plan, correctness_runtime = run_parallel_phase(
            correctness_tasks, "Step 3.5 Flash correctness", "correctness"
        )
        # Phase 2: use Step 3.5 Flash for all selected step-critic calls.
        critic_phase_plan, critic_runtime = run_parallel_phase(
            critic_tasks, "Step 3.5 Flash critic", "critic"
        )

        correctness_cache = {
            task["task_id"]: repair_cache.get(task["task_id"], {})
            for task in correctness_tasks
        }
        critic_cache = {
            task["task_id"]: repair_cache.get(task["task_id"], {})
            for task in critic_tasks
        }

        correctness_success = sum(
            correctness_cache.get(task["task_id"], {}).get("status") == "success"
            for task in correctness_tasks
        )
        critic_success = sum(
            critic_cache.get(task["task_id"], {}).get("status") == "success"
            for task in critic_tasks
        )
        print(f"Correctness repaired: {correctness_success}/{len(correctness_tasks)}")
        print(f"Critic repaired: {critic_success}/{len(critic_tasks)}")
        error_records = [
            record for record in list(correctness_cache.values()) + list(critic_cache.values())
            if record.get("status") != "success"
        ]
        if error_records:
            print("First evaluator errors (full details are in the JSONL checkpoints):")
            display(pd.DataFrame([
                {
                    "task_id": record.get("task_id"),
                    "judge_model": record.get("judge_model"),
                    "key_slot": record.get("key_slot"),
                    "error": str(record.get("error", ""))[:500],
                }
                for record in error_records[:12]
            ]))
        '''
    ),
    md("## 7. Apply repaired decisions and recompute only dependent metrics"),
    code(
        r'''
        CLASS_VALUE = {
            "progressive": 1.0,
            "neutral_useful": 0.5,
            "neutral_waste": 0.0,
            "redundant": -0.5,
            "regressive": -1.0,
        }


        def label_from_cached_signals(detail):
            base = cached_base_score(detail)
            critic_signal = float(detail.get("critic_signal", 0.0))
            score = base + W_CRITIC * critic_signal
            if float(detail.get("error_flag", 0.0)) or score <= REGRESSIVE_THRESHOLD:
                label = "regressive"
            elif float(detail.get("harmful_repeat", 0.0)):
                label = "redundant"
            elif score >= PROGRESSIVE_THRESHOLD:
                label = "progressive"
            elif bool(detail.get("useful_verification", False)):
                label = "neutral_useful"
            else:
                label = "neutral_waste"
            return float(score), label


        def refresh_quality(metric):
            details = metric.get("Q", {}).get("step_details", [])
            labels = []
            for detail in details:
                score, label = label_from_cached_signals(detail)
                detail["hybrid_score"] = score
                detail["label"] = label
                labels.append(label)
            signed = float(np.mean([CLASS_VALUE[label] for label in labels])) if labels else -1.0
            quality = metric["Q"]
            quality["step_labels"] = labels
            quality["signed_quality"] = signed
            quality["Q_step"] = float(np.clip((signed + 1.0) / 2.0, 0.0, 1.0))
            for label in CLASS_VALUE:
                quality[f"{label}_rate"] = labels.count(label) / max(len(labels), 1)


        def first_solution_index(metric, trajectory):
            if not metric["C_G_V"].get("C_final"):
                return None
            candidates = []
            evidence_step = metric["C_G_V"].get("grounding_evidence_step")
            if evidence_step is not None:
                candidates.append(max(0, int(evidence_step) - 1))
            for index, step in enumerate(trajectory.get("steps", [])):
                if step.get("action_type") == "answer":
                    candidates.append(index)
                    break
            return min(candidates) if candidates else None


        def refresh_token_utility(metric, trajectory):
            if not metric.get("valid_trajectory"):
                return
            labels = metric["Q"].get("step_labels", [])
            solution_index = first_solution_index(metric, trajectory)
            useful = pre_waste = post_waste = answer_reporting = visible_process = 0
            for index, (step, label) in enumerate(zip(trajectory.get("steps", []), labels)):
                output_tokens = max(
                    0,
                    int(step.get("output_tokens", 0) or 0)
                    - int(step.get("hidden_reasoning_tokens", 0) or 0),
                )
                if step.get("action_type") == "answer":
                    answer_reporting += output_tokens
                    continue
                visible_process += output_tokens
                if solution_index is not None and index > solution_index:
                    post_waste += output_tokens
                elif label in {"progressive", "neutral_useful"}:
                    useful += output_tokens
                else:
                    pre_waste += output_tokens
            denominator = max(useful + pre_waste + post_waste, 1)
            token_record = metric["T"]
            token_record.update({
                "PTU": useful / denominator,
                "T_visible_process": visible_process,
                "T_useful": useful,
                "T_pre_waste": pre_waste,
                "T_post_waste": post_waste,
                "T_answer_reporting": answer_reporting,
                "solution_evidence_step": None if solution_index is None else solution_index + 1,
                "PreWasteRate": pre_waste / denominator,
                "PostWasteRate": post_waste / denominator,
            })


        correctness_audit_rows = []
        for task in correctness_tasks:
            result = correctness_cache.get(task["task_id"], {})
            key = (task["agent"], task["problem_id"])
            metric = metric_by_key[key]
            before = metric_before_by_key[key]["C_G_V"]
            success = result.get("status") == "success"
            if success:
                parsed = result["parsed"]
                decision = int(parsed["correct"])
                cg = metric["C_G_V"]
                cg["C_judge"] = decision
                cg["C_final"] = decision
                cg["correctness_source"] = "judge_fallback"
                cg["judge_used"] = True
                cg["judge_record"] = {
                    "decision": decision,
                    "confidence": parsed["confidence"],
                    "reason": parsed["reason"],
                    "raw": result.get("raw", ""),
                    "error": "",
                    "key_slot": result.get("key_slot"),
                    "attempts": result.get("attempts"),
                    "latency_sec": result.get("latency_sec"),
                }
                cg["needs_judge"] = False
                if int(cg.get("G", 0)):
                    cg["G_level"] = "G3" if decision else "G2"
                cg["V"] = decision * int(cg.get("G", 0))
            correctness_audit_rows.append({
                "agent": key[0], "problem_id": key[1], "status": result.get("status", "missing"),
                "C_before": before.get("C_final"),
                "C_after": metric["C_G_V"].get("C_final"),
                "V_before": before.get("V"), "V_after": metric["C_G_V"].get("V"),
                "confidence": (result.get("parsed") or {}).get("confidence"),
                "low_confidence": (
                    result.get("status") == "success"
                    and float((result.get("parsed") or {}).get("confidence", 0.0))
                    < LOW_CONFIDENCE_AUDIT_THRESHOLD
                ),
                "error": result.get("error", ""), "key_slot": result.get("key_slot"),
            })


        critic_audit_rows = []
        affected_quality_keys = set()
        for task in critic_tasks:
            result = critic_cache.get(task["task_id"], {})
            key = (task["agent"], task["problem_id"])
            metric = metric_by_key[key]
            index = int(task["step_index"])
            detail = metric["Q"]["step_details"][index]
            label_before = detail.get("label")
            score_before = detail.get("hybrid_score")
            if result.get("status") == "success":
                parsed = result["parsed"]
                detail["critic_signal"] = float(CRITIC_LABELS[parsed["label"]])
                detail["critic_used"] = True
                detail["critic_record"] = {
                    "used": True, "signal": detail["critic_signal"],
                    "label": parsed["label"], "confidence": parsed["confidence"],
                    "reason": parsed["reason"], "raw": result.get("raw", ""),
                    "error": "", "key_slot": result.get("key_slot"),
                    "attempts": result.get("attempts"), "latency_sec": result.get("latency_sec"),
                }
                affected_quality_keys.add(key)
            score_after, label_after = label_from_cached_signals(detail)
            critic_audit_rows.append({
                "agent": key[0], "problem_id": key[1], "step_index": index + 1,
                "status": result.get("status", "missing"),
                "label_before": label_before, "label_after": label_after,
                "label_changed": label_before != label_after,
                "score_before": score_before, "score_after": score_after,
                "critic_label": (result.get("parsed") or {}).get("label"),
                "confidence": (result.get("parsed") or {}).get("confidence"),
                "error": result.get("error", ""), "key_slot": result.get("key_slot"),
            })

        for key in affected_quality_keys:
            refresh_quality(metric_by_key[key])

        # Correctness and critic changes can both alter the token decomposition.
        for key, metric in metric_by_key.items():
            refresh_token_utility(metric, trajectory_by_key[key])
            if metric.get("valid_trajectory"):
                metric["E_optional"] = float(np.clip(
                    metric["C_G_V"]["V"]
                    * metric["T"]["PTU"]
                    * metric["Q"]["Q_step"]
                    * (1.0 - metric["R"]["R_harmful"]),
                    0.0, 1.0,
                ))

        correctness_audit = pd.DataFrame(correctness_audit_rows)
        critic_audit = pd.DataFrame(critic_audit_rows)
        print("Remaining unresolved correctness cases:", sum(
            row.get("valid_trajectory") and row["C_G_V"].get("needs_judge") for row in metrics
        ))
        print("Successful critic calls now present:", sum(
            detail.get("critic_used", False)
            for row in metrics for detail in row.get("Q", {}).get("step_details", [])
        ))
        display(correctness_audit.groupby(["agent", "status"]).size().rename("count"))
        display(critic_audit.groupby(["agent", "status"]).size().rename("count"))
        '''
    ),
    md("## 8. Recompute summaries and evaluator coverage"),
    code(
        r'''
        INFRASTRUCTURE_STOPS = {"api_error", "worker_exception", "trajectory_timeout"}
        PROTOCOL_STOPS = {"format_error_limit", "answer_only_violation_limit"}


        def safe_mean(values):
            series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
            return float(series.mean()) if len(series) else np.nan


        def summarize(metric_rows):
            rows = []
            for agent, values in pd.DataFrame({"row": metric_rows}).assign(
                agent=[row["agent"] for row in metric_rows]
            ).groupby("agent")["row"]:
                values = list(values)
                valid = [row for row in values if row.get("valid_trajectory")]
                attempted = len(values)
                correct = sum(int(row["C_G_V"]["C_final"]) for row in values)
                grounded = sum(int(row["C_G_V"]["G"]) for row in values)
                verified = sum(int(row["C_G_V"]["V"]) for row in values)
                billable = sum(float(row["T"]["T_billable"]) for row in values)
                latency = sum(float(row["L"]["L_trajectory_sec"]) for row in values)
                protocol = sum(row.get("stop_reason") in PROTOCOL_STOPS for row in values)
                infrastructure = sum(row.get("stop_reason") in INFRASTRUCTURE_STOPS for row in values)
                critic_details = [
                    detail for row in valid for detail in row.get("Q", {}).get("step_details", [])
                ]
                rows.append({
                    "agent": agent, "attempted_n": attempted, "valid_n": len(valid),
                    "coverage": len(valid) / attempted,
                    "format_adherence": 1 - protocol / attempted,
                    "infrastructure_failure_rate": infrastructure / attempted,
                    "C_operational": correct / attempted,
                    "C_valid": safe_mean(row["C_G_V"]["C_final"] for row in valid),
                    "G_operational": grounded / attempted,
                    "G_valid": safe_mean(row["C_G_V"]["G"] for row in valid),
                    "V_operational": verified / attempted,
                    "V_valid": safe_mean(row["C_G_V"]["V"] for row in valid),
                    "Q": safe_mean(row["Q"]["Q_step"] for row in valid),
                    "PTU": safe_mean(row["T"]["PTU"] for row in valid),
                    "R_harmful": safe_mean(row["R"]["R_harmful"] for row in valid),
                    "E_optional": safe_mean(row.get("E_optional") for row in valid),
                    "total_billable_tokens": billable,
                    "avg_billable_tokens_per_attempt": billable / attempted,
                    "tokens_per_correct_solve": billable / correct if correct else np.nan,
                    "tokens_per_verified_solve": billable / verified if verified else np.nan,
                    "seconds_per_verified_solve": latency / verified if verified else np.nan,
                    "avg_latency_sec_all_attempts": latency / attempted,
                    "unresolved_judge_count": sum(row["C_G_V"].get("needs_judge", False) for row in valid),
                    "judge_fallback_count": sum(
                        row["C_G_V"].get("correctness_source") == "judge_fallback" for row in valid
                    ),
                    "critic_selected_count": sum(
                        bool(detail.get("critic_used")) or bool(detail.get("critic_record", {}).get("error"))
                        for detail in critic_details
                    ),
                    "critic_success_count": sum(bool(detail.get("critic_used")) for detail in critic_details),
                    "reasoning_metric_n": len(valid),
                })
            return pd.DataFrame(rows).sort_values(
                ["V_operational", "C_operational"], ascending=False
            ).reset_index(drop=True)


        summary_before = summarize(metrics_before)
        summary = summarize(metrics)
        display(summary)

        judge_attempts_by_agent = correctness_audit.groupby("agent").size().rename("judge_selected_count")
        judge_success_by_agent = (
            correctness_audit.assign(success=correctness_audit["status"].eq("success"))
            .groupby("agent")["success"].sum().rename("judge_success_count")
        )
        evaluator_coverage = summary[[
            "agent", "valid_n", "judge_fallback_count", "unresolved_judge_count",
            "critic_selected_count", "critic_success_count",
        ]].merge(judge_attempts_by_agent, on="agent", how="left").merge(
            judge_success_by_agent, on="agent", how="left"
        )
        evaluator_coverage[["judge_selected_count", "judge_success_count"]] = (
            evaluator_coverage[["judge_selected_count", "judge_success_count"]].fillna(0).astype(int)
        )
        evaluator_coverage["critic_success_rate"] = (
            evaluator_coverage["critic_success_count"]
            / evaluator_coverage["critic_selected_count"].replace(0, np.nan)
        )
        evaluator_coverage["judge_success_rate"] = (
            evaluator_coverage["judge_success_count"]
            / evaluator_coverage["judge_selected_count"].replace(0, np.nan)
        )
        evaluator_coverage["correctness_decision_coverage"] = 1 - (
            evaluator_coverage["unresolved_judge_count"]
            / evaluator_coverage["valid_n"].replace(0, np.nan)
        )
        display(evaluator_coverage)

        impact = summary_before[["agent", "C_operational", "V_operational", "Q", "PTU"]].merge(
            summary[["agent", "C_operational", "V_operational", "Q", "PTU"]],
            on="agent", suffixes=("_before", "_after"),
        )
        for metric_name in ["C_operational", "V_operational", "Q", "PTU"]:
            impact[f"delta_{metric_name}"] = (
                impact[f"{metric_name}_after"] - impact[f"{metric_name}_before"]
            )
        display(impact.sort_values("V_operational_after", ascending=False))
        '''
    ),
    md("## 9. Paper-facing uncertainty, observability, and eligibility tables"),
    code(
        r'''
        def wilson_interval(successes, n, z=1.959963984540054):
            if n <= 0:
                return np.nan, np.nan
            p = successes / n
            denominator = 1 + z * z / n
            centre = (p + z * z / (2 * n)) / denominator
            half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
            return max(0.0, centre - half), min(1.0, centre + half)


        bounds_rows = []
        for agent in sorted({row["agent"] for row in metrics}):
            before_rows = [row for row in metrics_before if row["agent"] == agent]
            after_rows = [row for row in metrics if row["agent"] == agent]
            unresolved_before_n = sum(row["C_G_V"].get("needs_judge", False) for row in before_rows)
            unresolved_after_n = sum(row["C_G_V"].get("needs_judge", False) for row in after_rows)
            c_before = sum(row["C_G_V"]["C_final"] for row in before_rows)
            c_after = sum(row["C_G_V"]["C_final"] for row in after_rows)
            bounds_rows.append({
                "agent": agent, "n": len(after_rows),
                "C_lower_before": c_before / len(before_rows),
                "C_upper_before": (c_before + unresolved_before_n) / len(before_rows),
                "unresolved_before": unresolved_before_n,
                "C_lower_after": c_after / len(after_rows),
                "C_upper_after": (c_after + unresolved_after_n) / len(after_rows),
                "unresolved_after": unresolved_after_n,
            })
        correctness_bounds = pd.DataFrame(bounds_rows)
        display(correctness_bounds)


        observability_rows = []
        for agent in sorted({row["agent_name"] for row in trajectories}):
            agent_trajectories = [row for row in trajectories if row["agent_name"] == agent]
            steps = [step for row in agent_trajectories for step in row.get("steps", [])]
            provider_reasoning_steps = [
                step for step in steps
                if str(step.get("provider_reasoning", "")).strip()
                or int(step.get("provider_reasoning_chars", 0) or 0) > 0
            ]
            unmapped_reasoning_steps = [
                step for step in provider_reasoning_steps
                if not str(step.get("raw_response", "")).strip()
                and int(step.get("visible_content_chars", 0) or 0) == 0
            ]
            actionable = [step for step in steps if step.get("action_type") in {"code", "answer"}]
            code_success = [
                step for step in steps
                if step.get("action_type") == "code" and step.get("code_success")
            ]
            valid_n = sum(row.get("finished") and row.get("stop_reason") == "explicit_final_answer" for row in agent_trajectories)
            protocol_failures = sum(row.get("stop_reason") in PROTOCOL_STOPS for row in agent_trajectories)
            observability_rows.append({
                "agent": agent, "attempted_n": len(agent_trajectories), "valid_n": valid_n,
                "total_steps": len(steps),
                "format_adherence": 1 - protocol_failures / max(len(agent_trajectories), 1),
                "provider_reasoning_step_rate": len(provider_reasoning_steps) / max(len(steps), 1),
                "unmapped_provider_reasoning_step_rate": len(unmapped_reasoning_steps) / max(len(steps), 1),
                "actionable_step_rate": len(actionable) / max(len(steps), 1),
                "successful_code_step_rate": len(code_success) / max(len(steps), 1),
                "hidden_reasoning_tokens": sum(
                    int(step.get("hidden_reasoning_tokens", 0) or 0) for step in steps
                ),
                "provider_reasoning_chars": sum(
                    int(step.get("provider_reasoning_chars", 0) or 0)
                    or len(str(step.get("provider_reasoning", ""))) for step in steps
                ),
            })
        trace_observability = pd.DataFrame(observability_rows)
        trace_observability["trace_metric_eligible"] = (
            (trace_observability["format_adherence"] >= TRACE_MIN_FORMAT_ADHERENCE)
            & (
                trace_observability["unmapped_provider_reasoning_step_rate"]
                <= TRACE_MAX_UNMAPPED_REASONING_STEP_RATE
            )
        )
        trace_observability["exclusion_reason"] = np.where(
            trace_observability["trace_metric_eligible"],
            "",
            "provider reasoning is not reliably mapped to visible protocol actions",
        )
        display(trace_observability)


        paper_primary = summary.copy()
        c_intervals, v_intervals = [], []
        for row in paper_primary.itertuples(index=False):
            c_intervals.append(wilson_interval(round(row.C_operational * row.attempted_n), row.attempted_n))
            v_intervals.append(wilson_interval(round(row.V_operational * row.attempted_n), row.attempted_n))
        paper_primary["C_ci_low"] = [value[0] for value in c_intervals]
        paper_primary["C_ci_high"] = [value[1] for value in c_intervals]
        paper_primary["V_ci_low"] = [value[0] for value in v_intervals]
        paper_primary["V_ci_high"] = [value[1] for value in v_intervals]
        paper_primary = paper_primary.merge(
            trace_observability[["agent", "trace_metric_eligible", "exclusion_reason"]], on="agent"
        )

        paper_trace = paper_primary[[
            "agent", "reasoning_metric_n", "Q", "PTU", "R_harmful",
            "trace_metric_eligible", "exclusion_reason",
        ]].copy()
        for column in ["Q", "PTU", "R_harmful"]:
            paper_trace.loc[~paper_trace["trace_metric_eligible"], column] = np.nan

        decision_sources = (
            pd.DataFrame({
                "agent": [row["agent"] for row in metrics],
                "correctness_source": [row["C_G_V"]["correctness_source"] for row in metrics],
            })
            .groupby(["agent", "correctness_source"]).size().rename("count").reset_index()
        )
        decision_sources["rate"] = decision_sources["count"] / decision_sources.groupby("agent")["count"].transform("sum")

        display(paper_primary[[
            "agent", "C_operational", "C_ci_low", "C_ci_high", "G_operational",
            "V_operational", "V_ci_low", "V_ci_high", "coverage", "format_adherence",
            "tokens_per_verified_solve", "avg_latency_sec_all_attempts", "trace_metric_eligible",
        ]])
        display(paper_trace)
        '''
    ),
    md("## 10. Updated core and repair-impact plots"),
    code(
        r'''
        FIGURE_ROOT = OUTPUT_ROOT / "figures"
        FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        summary.set_index("agent")[["C_operational", "G_operational", "V_operational"]].plot.bar(
            ax=axes[0], ylim=(0, 1), title="Operational outcomes after evaluator repair"
        )
        impact.set_index("agent")[["delta_C_operational", "delta_V_operational"]].plot.bar(
            ax=axes[1], title="Judge repair impact"
        )
        impact.set_index("agent")[["delta_Q", "delta_PTU"]].plot.bar(
            ax=axes[2], title="Critic repair impact"
        )
        for axis in axes:
            axis.tick_params(axis="x", rotation=30)
            axis.axhline(0, color="black", linewidth=0.7)
        fig.tight_layout()
        repair_plot = FIGURE_ROOT / "evaluator_repair_impact.png"
        fig.savefig(repair_plot, dpi=180, bbox_inches="tight")
        plt.show()

        plot_data = trace_observability.set_index("agent")[[
            "format_adherence", "actionable_step_rate", "unmapped_provider_reasoning_step_rate"
        ]]
        plot_data.plot.bar(figsize=(11, 5), ylim=(0, 1), title="Trace observability and protocol compatibility")
        plt.xticks(rotation=30)
        plt.tight_layout()
        observability_plot = FIGURE_ROOT / "trace_observability.png"
        plt.savefig(observability_plot, dpi=180, bbox_inches="tight")
        plt.show()
        '''
    ),
    md("## 11. Export the repaired portable metrics bundle"),
    code(
        r'''
        def write_jsonl(path, rows):
            with Path(path).open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")


        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        export_dir = OUTPUT_ROOT / f"{RUN_NAME}_{stamp}"
        export_dir.mkdir(parents=True, exist_ok=False)

        write_jsonl(export_dir / "metrics.jsonl", metrics)
        write_jsonl(export_dir / "trajectories.jsonl", trajectories)
        write_jsonl(export_dir / "problems.jsonl", bundle["problems"])
        (export_dir / "prm_scores.json").write_text(
            json.dumps(bundle["prm_scores"], ensure_ascii=True), encoding="utf-8"
        )
        summary.to_csv(export_dir / "summary.csv", index=False)
        evaluator_coverage.to_csv(export_dir / "evaluator_coverage.csv", index=False)
        impact.to_csv(export_dir / "evaluator_repair_impact.csv", index=False)
        correctness_audit.to_csv(export_dir / "correctness_judge_audit.csv", index=False)
        critic_audit.to_csv(export_dir / "step_critic_audit.csv", index=False)
        correctness_bounds.to_csv(export_dir / "correctness_uncertainty_bounds.csv", index=False)
        trace_observability.to_csv(export_dir / "trace_observability.csv", index=False)
        paper_primary.to_csv(export_dir / "paper_primary_metrics.csv", index=False)
        paper_trace.to_csv(export_dir / "paper_trace_metrics.csv", index=False)
        decision_sources.to_csv(export_dir / "correctness_decision_sources.csv", index=False)
        shutil.copy2(repair_plot, export_dir / repair_plot.name)
        shutil.copy2(observability_plot, export_dir / observability_plot.name)

        # Preserve original non-derived audit files when present.
        for name in ["invalid_attempts.csv", "generation_health.csv", "merge_audit.json"]:
            payload = bundle["payloads"].get(name)
            if payload is not None:
                (export_dir / name).write_bytes(payload)

        data_files = sorted(path.name for path in export_dir.iterdir() if path.is_file())
        repair_manifest = {
            "artifact_type": "strive_metrics_evaluator_repair",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_name": RUN_NAME,
            "source_bundle": bundle["source"],
            "source_manifest_sha256": hashlib.sha256(
                bundle["payloads"]["manifest.json"]
            ).hexdigest(),
            "config": manifest_before.get("config", {}),
            "models": manifest_before.get("models", []),
            "core_metrics": manifest_before.get(
                "core_metrics", ["C", "G", "V", "Q", "PTU", "R_harmful"]
            ),
            "optional_metric": manifest_before.get(
                "optional_metric", "E_optional = V * PTU * Q * (1 - R_harmful)"
            ),
            "raw_data_schema_version": manifest_before.get("raw_data_schema_version", 1),
            "evaluation_policy": manifest_before.get("evaluation_policy", {}),
            "reused_without_recomputation": [
                "trajectories", "sandbox outputs", "deterministic grounding",
                "two PRM score sets", "semantic redundancy features", "latency",
            ],
            "recomputed": [
                "judge fallback", "selective critic", "C_final", "V", "Q",
                "token utility", "E_optional", "dependent summaries",
            ],
            "judge": {
                "provider": "nvidia",
                "correctness_model": CORRECTNESS_MODEL,
                "critic_model": CRITIC_MODEL,
                "configured_key_count": len(CONFIGURED_NVIDIA_KEYS),
                "rpm_per_key": JUDGE_RPM_PER_KEY,
                "correctness_tasks": len(correctness_tasks),
                "correctness_successes": int((correctness_audit["status"] == "success").sum()),
                "critic_tasks": len(critic_tasks),
                "critic_successes": int((critic_audit["status"] == "success").sum()),
            },
            "trace_metric_policy": {
                "minimum_format_adherence": TRACE_MIN_FORMAT_ADHERENCE,
                "maximum_unmapped_provider_reasoning_step_rate": TRACE_MAX_UNMAPPED_REASONING_STEP_RATE,
                "operational_metrics_keep_all_models": True,
                "ineligible_trace_metrics_are_NA": True,
            },
            "data_files": data_files,
            "sha256": {
                name: hashlib.sha256((export_dir / name).read_bytes()).hexdigest()
                for name in data_files
            },
        }
        (export_dir / "manifest.json").write_text(
            json.dumps(repair_manifest, indent=2), encoding="utf-8"
        )

        export_zip = Path(shutil.make_archive(str(export_dir), "zip", root_dir=export_dir))
        print("Repaired portable metrics ZIP:", export_zip)
        if FileLink is not None:
            display(FileLink(str(export_zip)))
        '''
    ),
    md(
        """
        ## What to run

        Run all cells in this notebook once. Sections 1-5 load the saved evidence and build the
        repair queue. Section 6 is the only network-expensive section and is resumable. Sections
        7-11 are cheap recomputation and export.

        The generated repaired ZIP can be loaded into the existing advanced-analysis notebook.
        That later analysis reruns only tables, bootstrap resampling, statistical tests, and plots;
        it must not reload the PRMs or regenerate trajectories.

        For the paper, use `paper_primary_metrics.csv` for all six models. Use
        `paper_trace_metrics.csv` for `Q`, PTU, and harmful redundancy; models failing the declared
        trace-observability rule appear as `NA` with an explicit exclusion reason.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


for index, cell in enumerate(cells, 1):
    if cell["cell_type"] != "code" or cell["source"].lstrip().startswith("%"):
        continue
    ast.parse(cell["source"], filename=f"cell_{index}")

OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {OUTPUT}")
