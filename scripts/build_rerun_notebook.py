import copy
import json
import re
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "strive-maths-final-core-metrics-multimodel.ipynb"
OUTPUT = ROOT / "outputs" / "strive-rerun-all-invalid-trajectories.ipynb"


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


base = json.loads(BASE.read_text(encoding="utf-8"))
cells = [
    markdown(
        """
        # STRIVE Recovery: Rerun Every Invalid Trajectory

        This generation-only notebook loads the canonical 1,800-record bundle and reruns
        every record that is not publishable. A publishable trajectory must:

        - be marked `finished=True`;
        - not be timed out;
        - stop with `explicit_final_answer`;
        - contain a non-empty final answer;
        - contain at least one recorded step and one answer step.

        Successful replacements are checkpointed immediately. Failed recovery attempts do
        not overwrite the source record. The final gate passes only when all 1,800 unique
        model-problem trajectories satisfy the publishability predicate.

        **Methodology warning:** infrastructure retries repair missing observations.
        Repeatedly retrying model-format failures changes the operational benchmark. Keep
        the original canonical ZIP for operational reporting and describe the fully
        repaired ZIP as a conditional/recovery analysis dataset.
        """
    )
]

# Reuse the reviewed configuration, sandbox, API adapter, parser, and ReAct controller.
# The PRM and metric sections are intentionally excluded from this generation-only notebook.
for index in range(1, 17):
    cell = copy.deepcopy(base["cells"][index])
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    else:
        cell.pop("execution_count", None)
        cell.pop("outputs", None)
    source = "".join(cell.get("source", []))
    if index == 2:
        source = dedent(
            '''
            # Generation-only dependencies. No PRM is loaded in this notebook.
            %pip install -q "openai>=1.68,<3" "datasets>=3.2,<5" \
                "transformers>=4.46,<5" sympy scipy pandas matplotlib seaborn \
                tqdm tiktoken
            '''
        ).strip() + "\n"
    if index == 5:
        replacement = dedent(
            '''
            API_KEYS = {
                "NVIDIA_API_KEY_1": "",
                "NVIDIA_API_KEY_2": "",
                "NVIDIA_API_KEY_3": "",
                "NVIDIA_API_KEY_4": "",
                "NVIDIA_API_KEY_5": "",
                "NVIDIA_API_KEY_6": "",
                "OPENAI_API_KEY": "",
                "NVIDIA_CRITIC_API_KEY": "",
            }

            load_api_secrets(REQUIRED_SECRET_NAMES + OPTIONAL_SECRET_NAMES)
            '''
        ).strip()
        source, count = re.subn(
            r"API_KEYS\s*=\s*\{.*?\}\s*\n\s*load_api_secrets\(REQUIRED_SECRET_NAMES \+ OPTIONAL_SECRET_NAMES\)",
            replacement,
            source,
            count=1,
            flags=re.DOTALL,
        )
        if count != 1:
            raise RuntimeError("Could not sanitize the API_KEYS block in the base notebook")
    if index == 7:
        source = source.split("\npaper_problems, smoke_pool = load_paper_problem_sets(cfg)", 1)[0]
        source += (
            "\n\n# Recovery uses problems.jsonl from the canonical generation ZIP.\n"
            "# No Hugging Face dataset download is required here.\n"
        )
    cell["source"] = source
    cells.append(cell)

cells.extend(
    [
        markdown("## 7. Recovery input and strict selection policy"),
        code(
            r'''
            SOURCE_GENERATION_INPUT = (
                "/path/to/your/project/"
                "i-got-these-reviews-from-a/strive_canonical_merge/"
                "paper_math200_olympiad100_v11_canonical_generation_20260714T132605Z.zip"
            )
            from IPython.display import FileLink, display


            def show_download_link(path):
                display(FileLink(str(Path(path).resolve())))


            RECOVERY_RUN_NAME = "paper_math200_olympiad100_v11_all_invalid_recovery"

            # Every invalid source record receives at most this many new trajectory attempts.
            # Rerunning the recovery cell resumes from the append-only checkpoint.
            MAX_RECOVERY_ATTEMPTS_PER_TRAJECTORY = 3
            RERUN_PROTOCOL_FAILURES = True
            SHOW_LIVE_TRACE = False
            RUN_ENDPOINT_PREFLIGHT = False
            REQUIRE_ALL_PUBLISHABLE = True

            INFRASTRUCTURE_STOPS = {"api_error", "worker_exception", "trajectory_timeout"}
            PROTOCOL_STOPS = {"format_error_limit", "answer_only_violation_limit"}
            NVIDIA_RECOVERY_KEY_SLOTS = [f"NVIDIA_API_KEY_{index}" for index in range(1, 7)]

            recovery_cfg = replace(cfg)
            recovery_cfg.output_dir = str(Path(
                "/kaggle/working/strive_all_invalid_recovery"
                if Path("/kaggle/working").exists()
                else Path.cwd() / "strive_all_invalid_recovery"
            ).resolve())
            Path(recovery_cfg.output_dir).mkdir(parents=True, exist_ok=True)
            Path(recovery_cfg.output_dir, "checkpoints").mkdir(parents=True, exist_ok=True)

            # The router owns NVIDIA retries and rotates six distinct accounts.
            recovery_cfg.max_retries = 1
            recovery_cfg.request_timeout_sec = 180
            recovery_cfg.request_total_timeout_sec = 1200
            recovery_cfg.trajectory_timeout_sec = 1800
            recovery_cfg.rate_limit_cooldown_sec = 180.0
            recovery_cfg.resource_exhausted_cooldown_sec = 60.0
            recovery_cfg.rpm_utilization = 0.80
            recovery_cfg.live_trace = SHOW_LIVE_TRACE
            cfg.live_trace = SHOW_LIVE_TRACE

            print("Recovery output directory:", recovery_cfg.output_dir)
            '''
        ),
        code(
            r'''
            def _safe_extract_generation_zip(path, target):
                if target.exists():
                    shutil.rmtree(target)
                target.mkdir(parents=True, exist_ok=True)
                safe_root = target.resolve()
                import zipfile
                with zipfile.ZipFile(path) as archive:
                    for member in archive.infolist():
                        destination = (target / member.filename).resolve()
                        if destination != safe_root and safe_root not in destination.parents:
                            raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
                    archive.extractall(target)


            def load_generation_source(value):
                source = Path(value).expanduser().resolve()
                if not source.exists():
                    raise FileNotFoundError(source)
                if source.is_file():
                    if source.suffix.lower() != ".zip":
                        raise ValueError("SOURCE_GENERATION_INPUT must be a ZIP or directory")
                    target = Path(recovery_cfg.output_dir) / "loaded_source" / source.stem
                    _safe_extract_generation_zip(source, target)
                    source = target
                manifests = list(source.rglob("generation_manifest.json"))
                if len(manifests) != 1:
                    raise RuntimeError(
                        f"Expected one generation_manifest.json under {source}; found {len(manifests)}"
                    )
                root = manifests[0].parent
                manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
                for name, expected in manifest.get("sha256", {}).items():
                    candidate = root / name
                    if not candidate.exists():
                        raise FileNotFoundError(candidate)
                    actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
                    if actual != expected:
                        raise RuntimeError(f"Checksum mismatch: {candidate}")
                with (root / "trajectories.jsonl").open(encoding="utf-8") as handle:
                    trajectories = [
                        Trajectory.from_dict(json.loads(line)) for line in handle if line.strip()
                    ]
                with (root / "problems.jsonl").open(encoding="utf-8") as handle:
                    problems = [json.loads(line) for line in handle if line.strip()]
                return trajectories, problems, manifest


            def trajectory_validation_reasons(trajectory):
                reasons = []
                if not trajectory.finished:
                    reasons.append("not_finished")
                if trajectory.timed_out:
                    reasons.append("timed_out")
                if trajectory.stop_reason != "explicit_final_answer":
                    reasons.append(trajectory.stop_reason or "missing_stop_reason")
                if not str(trajectory.final_answer or "").strip():
                    reasons.append("empty_final_answer")
                if not trajectory.steps:
                    reasons.append("no_steps")
                if not any(
                    str(step.get("action_type", "")) == "answer"
                    and str(step.get("final_answer", step.get("action_content", "")) or "").strip()
                    for step in trajectory.steps
                ):
                    reasons.append("answer_step_missing")
                return sorted(set(reasons))


            def is_publishable_trajectory(trajectory):
                return not trajectory_validation_reasons(trajectory)


            source_trajectories, source_problems, source_manifest = load_generation_source(
                SOURCE_GENERATION_INPUT
            )
            source_by_key = {}
            for trajectory in source_trajectories:
                key = (trajectory.agent_name, trajectory.problem_id)
                if key in source_by_key:
                    raise RuntimeError(f"Duplicate source trajectory: {key}")
                source_by_key[key] = trajectory
            problems_by_id = {str(problem["id"]): problem for problem in source_problems}

            if len(source_by_key) != 1800:
                raise RuntimeError(f"Expected 1,800 source trajectories, found {len(source_by_key)}")
            if len(problems_by_id) != 300:
                raise RuntimeError(f"Expected 300 source problems, found {len(problems_by_id)}")

            required_reruns = []
            for key, trajectory in source_by_key.items():
                reasons = trajectory_validation_reasons(trajectory)
                if not reasons:
                    continue
                if not RERUN_PROTOCOL_FAILURES and trajectory.stop_reason in PROTOCOL_STOPS:
                    continue
                required_reruns.append({
                    "agent_name": key[0],
                    "problem_id": key[1],
                    "dataset": trajectory.dataset,
                    "subject": trajectory.subject,
                    "level": trajectory.level,
                    "original_stop_reason": trajectory.stop_reason,
                    "validation_reasons": "|".join(reasons),
                })
            required_reruns.sort(key=lambda item: (item["agent_name"], item["problem_id"]))
            required_reruns_df = pd.DataFrame(required_reruns)
            display(required_reruns_df.groupby(
                ["agent_name", "original_stop_reason"]
            ).size().rename("count").reset_index())
            print("Total trajectories selected for recovery:", len(required_reruns))

            selection_path = Path(recovery_cfg.output_dir) / "required_reruns.csv"
            required_reruns_df.to_csv(selection_path, index=False)
            print("Selection audit:", selection_path)
            '''
        ),
        markdown("## 8. Six-key NVIDIA router and resumable recovery engine"),
        code(
            r'''
            missing_keys = [
                name for name in NVIDIA_RECOVERY_KEY_SLOTS
                if not os.environ.get(name, "").strip()
            ]
            if missing_keys:
                raise RuntimeError(f"Configure all six NVIDIA recovery keys: {missing_keys}")
            fingerprints = [
                hashlib.sha256(os.environ[name].strip().encode()).hexdigest()
                for name in NVIDIA_RECOVERY_KEY_SLOTS
            ]
            if len(set(fingerprints)) != len(fingerprints):
                raise RuntimeError("The six NVIDIA recovery credentials must be distinct")
            if any(item["agent_name"] == "gpt-5-nano" for item in required_reruns):
                if not os.environ.get("OPENAI_API_KEY", "").strip():
                    raise RuntimeError("OPENAI_API_KEY is required for GPT-5 nano recovery")
            print("Recovery credentials validated; secret values remain hidden.")

            PER_KEY_RPM = 2
            NVIDIA_MODEL_RPM = {
                "ministral-14b": 3,
                "glm-5.2": 3,
                "minimax-m3": 3,
                "nemotron-3-nano": 6,
                "gpt-oss-20b": 3,
            }
            MAX_ROUTER_ATTEMPTS_PER_CALL = 12
            KEY_429_COOLDOWN_SEC = 180.0
            TRANSIENT_KEY_COOLDOWN_SEC = 60.0
            MINIMAX_DEGRADED_COOLDOWN_SEC = 300.0
            BETWEEN_RECOVERY_PASSES_SEC = 30.0


            class NvidiaRecoveryRouter:
                def __init__(self, key_slots):
                    self.key_slots = list(key_slots)
                    self.condition = threading.Condition()
                    self.busy = {slot: False for slot in self.key_slots}
                    self.cooldown_until = {slot: 0.0 for slot in self.key_slots}
                    self.cursor = 0
                    self.model_pacers = {
                        name: PacedRateLimiter(rpm, recovery_cfg.rpm_utilization)
                        for name, rpm in NVIDIA_MODEL_RPM.items()
                    }
                    self.model_cooldown_until = defaultdict(float)

                def acquire_key(self, deadline):
                    with self.condition:
                        while True:
                            now = time.monotonic()
                            for offset in range(len(self.key_slots)):
                                index = (self.cursor + offset) % len(self.key_slots)
                                slot = self.key_slots[index]
                                if not self.busy[slot] and self.cooldown_until[slot] <= now:
                                    self.busy[slot] = True
                                    self.cursor = (index + 1) % len(self.key_slots)
                                    return slot
                            remaining = deadline - now
                            if remaining <= 1:
                                raise TimeoutError("No NVIDIA recovery key available before deadline")
                            available_times = [
                                value for slot, value in self.cooldown_until.items()
                                if not self.busy[slot]
                            ]
                            wait = min(5.0, remaining - 1)
                            if available_times:
                                wait = min(wait, max(0.1, min(available_times) - now))
                            self.condition.wait(timeout=max(0.1, wait))

                def release_key(self, slot, cooldown=0.0):
                    with self.condition:
                        self.busy[slot] = False
                        self.cooldown_until[slot] = max(
                            self.cooldown_until[slot], time.monotonic() + float(cooldown)
                        )
                        self.condition.notify_all()

                @staticmethod
                def failure_blob(exc):
                    parts = [str(exc)]
                    if isinstance(exc, ModelCallError):
                        parts.append(json.dumps(exc.meta.get("retry_errors", []), sort_keys=True))
                    return " ".join(parts)

                def wait_for_model(self, model_name, deadline):
                    while self.model_cooldown_until[model_name] > time.monotonic():
                        remaining = deadline - time.monotonic()
                        if remaining <= 1:
                            raise TimeoutError(f"{model_name} recovery deadline exhausted")
                        wait = min(
                            30.0,
                            self.model_cooldown_until[model_name] - time.monotonic(),
                            remaining - 1,
                        )
                        time.sleep(max(0.1, wait))

                def call(self, base_call, spec, messages, active_cfg, **kwargs):
                    if spec.provider != "nvidia":
                        openai_cfg = replace(active_cfg)
                        openai_cfg.max_retries = 4
                        return base_call(spec, messages, openai_cfg, **kwargs)

                    provided_deadline = kwargs.get("deadline_monotonic")
                    deadline = min(
                        provided_deadline if provided_deadline is not None else float("inf"),
                        time.monotonic() + recovery_cfg.request_total_timeout_sec,
                    )
                    errors = []
                    router_wait = 0.0
                    for router_attempt in range(1, MAX_ROUTER_ATTEMPTS_PER_CALL + 1):
                        self.wait_for_model(spec.name, deadline)
                        pace_started = time.monotonic()
                        self.model_pacers[spec.name].acquire()
                        router_wait += time.monotonic() - pace_started
                        slot = self.acquire_key(deadline)
                        routed_spec = replace(spec, api_key_env=slot, rpm=PER_KEY_RPM)
                        single_cfg = replace(active_cfg)
                        single_cfg.max_retries = 1
                        single_cfg.request_timeout_sec = recovery_cfg.request_timeout_sec
                        single_cfg.request_total_timeout_sec = recovery_cfg.request_timeout_sec
                        try:
                            text, meta = base_call(
                                routed_spec, messages, single_cfg, **kwargs
                            )
                            self.release_key(slot)
                            meta = dict(meta)
                            meta["router_attempts"] = router_attempt
                            meta["router_key_slot"] = slot
                            meta["rate_limit_wait_sec"] = float(
                                meta.get("rate_limit_wait_sec", 0.0) + router_wait
                            )
                            meta["retry_errors"] = errors + list(meta.get("retry_errors", []))
                            return text, meta
                        except Exception as exc:
                            blob = self.failure_blob(exc)
                            lower = blob.lower()
                            is_429 = "429" in blob or "too many requests" in lower
                            is_degraded = "degraded function cannot be invoked" in lower
                            is_transient = any(token in lower for token in (
                                "timed out", "timeout", "500", "502", "503", "504",
                                "resourceexhausted", "internalserver", "workers busy",
                                "connection", "list index out of range", "nonetype",
                            ))
                            errors.append({
                                "router_attempt": router_attempt,
                                "key_slot": slot,
                                "type": type(exc).__name__,
                                "message": truncate_middle(blob, 700),
                                "rate_limit": is_429,
                                "degraded_backend": is_degraded,
                            })
                            if is_429:
                                self.release_key(slot, KEY_429_COOLDOWN_SEC)
                                continue
                            if is_degraded and spec.name == "minimax-m3":
                                self.release_key(slot)
                                self.model_cooldown_until[spec.name] = (
                                    time.monotonic() + MINIMAX_DEGRADED_COOLDOWN_SEC
                                )
                                continue
                            if is_transient:
                                self.release_key(slot, TRANSIENT_KEY_COOLDOWN_SEC)
                                continue
                            self.release_key(slot)
                            raise

                    failure_meta = {
                        "latency_sec": 0.0,
                        "service_latency_sec": 0.0,
                        "service_time_all_attempts_sec": 0.0,
                        "rate_limit_wait_sec": router_wait,
                        "retry_wait_sec": 0.0,
                        "request_attempts": len(errors),
                        "retry_errors": errors,
                        "finish_reason": "error",
                        "provider_truncated": False,
                        "provider_reasoning": "",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "hidden_reasoning_tokens": 0,
                    }
                    raise ModelCallError(
                        f"NVIDIA recovery router exhausted for {spec.name}", failure_meta
                    )


            class RecoveryAttemptCheckpoint:
                def __init__(self, root):
                    self.path = Path(root) / "checkpoints" / RECOVERY_RUN_NAME / "attempts.jsonl"
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    self.lock = threading.Lock()

                def load(self):
                    attempts = defaultdict(int)
                    successful = {}
                    if not self.path.exists():
                        return attempts, successful
                    for line_number, line in enumerate(
                        self.path.read_text(encoding="utf-8").splitlines(), 1
                    ):
                        if not line.strip():
                            continue
                        try:
                            entry = json.loads(line)
                            trajectory = Trajectory.from_dict(entry["trajectory"])
                        except Exception as exc:
                            tqdm.write(f"Skipping checkpoint line {line_number}: {exc}")
                            continue
                        key = (trajectory.agent_name, trajectory.problem_id)
                        attempts[key] = max(attempts[key], int(entry.get("attempt", 1)))
                        if is_publishable_trajectory(trajectory):
                            successful[key] = trajectory
                    return attempts, successful

                def append(self, trajectory, attempt):
                    entry = {
                        "attempt": int(attempt),
                        "saved_at": datetime.now(timezone.utc).isoformat(),
                        "validation_reasons": trajectory_validation_reasons(trajectory),
                        "trajectory": trajectory.to_dict(),
                    }
                    with self.lock, self.path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())


            recovery_router = NvidiaRecoveryRouter(NVIDIA_RECOVERY_KEY_SLOTS)
            recovery_checkpoint = RecoveryAttemptCheckpoint(recovery_cfg.output_dir)
            attempt_counts, successful_replacements = recovery_checkpoint.load()
            required_keys = {
                (item["agent_name"], item["problem_id"]) for item in required_reruns
            }
            successful_replacements = {
                key: trajectory for key, trajectory in successful_replacements.items()
                if key in required_keys
            }
            print("Recovered from checkpoint:", len(successful_replacements), "/", len(required_keys))
            '''
        ),
        code(
            r'''
            specs_by_name = {spec.name: spec for spec in MODEL_SPECS}
            missing_specs = sorted({key[0] for key in required_keys} - set(specs_by_name))
            if missing_specs:
                raise RuntimeError(f"Missing model specifications: {missing_specs}")

            if RUN_ENDPOINT_PREFLIGHT:
                pending_names = sorted({key[0] for key in required_keys - set(successful_replacements)})
                try:
                    display(preflight_models([specs_by_name[name] for name in pending_names]))
                except Exception as exc:
                    print("Preflight warning; normal recovery retries remain active:", exc)

            _base_call_model = call_model


            def _recovery_call_model(spec, messages, active_cfg, **kwargs):
                return recovery_router.call(
                    _base_call_model, spec, messages, active_cfg, **kwargs
                )


            success_lock = threading.Lock()
            total_attempts_this_run = 0
            success_progress = tqdm(
                total=len(required_keys),
                initial=len(successful_replacements),
                desc="publishable recovery trajectories",
                unit="traj",
                dynamic_ncols=True,
            )


            def current_pending_by_model():
                grouped = defaultdict(list)
                for key in sorted(required_keys):
                    if key in successful_replacements:
                        continue
                    if attempt_counts[key] >= MAX_RECOVERY_ATTEMPTS_PER_TRAJECTORY:
                        continue
                    grouped[key[0]].append(problems_by_id[key[1]])
                return grouped


            def run_recovery_lane(model_name, problems):
                spec = specs_by_name[model_name]
                lane_results = []
                for index, problem in enumerate(problems, 1):
                    key = (model_name, problem["id"])
                    with success_lock:
                        attempt_counts[key] += 1
                        attempt_number = attempt_counts[key]
                    tqdm.write(
                        f"[{model_name}] recovery {index}/{len(problems)} | "
                        f"{problem['id']} | attempt {attempt_number}/"
                        f"{MAX_RECOVERY_ATTEMPTS_PER_TRAJECTORY}"
                    )
                    try:
                        trajectory = run_react_trajectory(problem, spec, recovery_cfg)
                    except Exception:
                        trajectory = Trajectory(
                            problem_id=problem["id"], dataset=problem["dataset"],
                            subject=problem["subject"], level=problem["level"],
                            problem_text=problem["problem"], gold_answer=problem["gold_answer"],
                            agent_name=spec.name, provider=spec.provider, model_id=spec.model_id,
                            stop_reason="worker_exception",
                            steps=[asdict(StepRecord(step_num=0, error=traceback.format_exc()))],
                        )
                    recovery_checkpoint.append(trajectory, attempt_number)
                    reasons = trajectory_validation_reasons(trajectory)
                    accepted = not reasons
                    if accepted:
                        with success_lock:
                            if key not in successful_replacements:
                                successful_replacements[key] = trajectory
                                success_progress.update(1)
                    lane_results.append((key, accepted, trajectory.stop_reason, reasons))
                    success_progress.set_postfix(
                        model=model_name,
                        stop=trajectory.stop_reason,
                        accepted=accepted,
                        refresh=True,
                    )
                return lane_results


            call_model = _recovery_call_model
            try:
                for recovery_pass in range(1, MAX_RECOVERY_ATTEMPTS_PER_TRAJECTORY + 1):
                    pending_by_model = current_pending_by_model()
                    if not pending_by_model:
                        break
                    print(
                        f"Recovery pass {recovery_pass}: ",
                        {name: len(values) for name, values in sorted(pending_by_model.items())},
                    )
                    with ThreadPoolExecutor(max_workers=len(pending_by_model)) as pool:
                        futures = {
                            pool.submit(run_recovery_lane, name, problems): name
                            for name, problems in pending_by_model.items()
                        }
                        for future in as_completed(futures):
                            lane_results = future.result()
                            total_attempts_this_run += len(lane_results)
                            tqdm.write(
                                f"[{futures[future]}] pass complete: "
                                f"{sum(item[1] for item in lane_results)}/{len(lane_results)} accepted"
                            )
                    remaining = len(required_keys - set(successful_replacements))
                    print(f"Remaining invalid trajectories after pass {recovery_pass}: {remaining}")
                    if remaining and recovery_pass < MAX_RECOVERY_ATTEMPTS_PER_TRAJECTORY:
                        time.sleep(BETWEEN_RECOVERY_PASSES_SEC)
            finally:
                call_model = _base_call_model
                success_progress.close()

            unresolved_keys = sorted(required_keys - set(successful_replacements))
            print("Successful replacements:", len(successful_replacements))
            print("Unresolved after configured attempts:", len(unresolved_keys))
            if unresolved_keys:
                display(pd.DataFrame([
                    {
                        "agent_name": key[0],
                        "problem_id": key[1],
                        "attempts": attempt_counts[key],
                        "original_stop_reason": source_by_key[key].stop_reason,
                    }
                    for key in unresolved_keys
                ]))
                print(
                    "Increase MAX_RECOVERY_ATTEMPTS_PER_TRAJECTORY, then rerun this cell. "
                    "Accepted checkpoint trajectories will not be repeated."
                )
            '''
        ),
        markdown("## 9. Merge accepted replacements, export, and enforce the 1,800-record gate"),
        code(
            r'''
            merged_by_key = dict(source_by_key)
            merged_by_key.update(successful_replacements)
            if len(merged_by_key) != 1800:
                raise RuntimeError(f"Merged key count changed: {len(merged_by_key)}")

            remaining_invalid = {
                key: trajectory_validation_reasons(trajectory)
                for key, trajectory in merged_by_key.items()
                if not is_publishable_trajectory(trajectory)
            }
            model_order = {spec.name: index for index, spec in enumerate(MODEL_SPECS)}
            problem_order = {problem["id"]: index for index, problem in enumerate(source_problems)}
            merged_records = sorted(
                [trajectory.to_dict() for trajectory in merged_by_key.values()],
                key=lambda item: (
                    problem_order[item["problem_id"]],
                    model_order[item["agent_name"]],
                ),
            )

            health_rows = []
            for model_name in sorted(model_order, key=model_order.get):
                rows = [item for item in merged_by_key.values() if item.agent_name == model_name]
                stops = pd.Series([item.stop_reason for item in rows]).value_counts().to_dict()
                health_rows.append({
                    "agent": model_name,
                    "records": len(rows),
                    "publishable": sum(is_publishable_trajectory(item) for item in rows),
                    "publishable_rate": np.mean([is_publishable_trajectory(item) for item in rows]),
                    "remaining_invalid": sum(not is_publishable_trajectory(item) for item in rows),
                    "stop_reasons": json.dumps({str(k): int(v) for k, v in stops.items()}, sort_keys=True),
                })
            recovery_health = pd.DataFrame(health_rows)
            display(recovery_health)

            export_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            export_name = f"{RECOVERY_RUN_NAME}_{export_timestamp}"
            export_dir = Path(recovery_cfg.output_dir) / export_name
            export_dir.mkdir(parents=True, exist_ok=False)


            def write_jsonl(path, records):
                with Path(path).open("w", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


            write_jsonl(export_dir / "trajectories.jsonl", merged_records)
            write_jsonl(export_dir / "problems.jsonl", source_problems)
            recovery_health.to_csv(export_dir / "generation_health.csv", index=False)
            required_reruns_df.to_csv(export_dir / "required_reruns.csv", index=False)

            unresolved_frame = pd.DataFrame([
                {
                    "agent_name": key[0],
                    "problem_id": key[1],
                    "attempts": attempt_counts[key],
                    "original_stop_reason": source_by_key[key].stop_reason,
                    "remaining_reasons": "|".join(reasons),
                }
                for key, reasons in sorted(remaining_invalid.items())
            ])
            unresolved_frame.to_csv(export_dir / "unresolved_after_recovery.csv", index=False)

            checkpoint_copy = export_dir / "recovery_attempts.jsonl"
            if recovery_checkpoint.path.exists():
                shutil.copy2(recovery_checkpoint.path, checkpoint_copy)
            else:
                checkpoint_copy.write_text("", encoding="utf-8")

            recovery_audit = {
                "source_run": source_manifest.get("run_name"),
                "source_trajectory_count": len(source_by_key),
                "selection_predicate": (
                    "finished and not timed_out and stop_reason=explicit_final_answer and "
                    "nonempty final_answer and steps and answer_step"
                ),
                "rerun_protocol_failures": RERUN_PROTOCOL_FAILURES,
                "selected_for_rerun": len(required_keys),
                "successful_replacements": len(successful_replacements),
                "remaining_invalid": len(remaining_invalid),
                "attempt_counts": {
                    f"{agent}::{problem_id}": int(attempt_counts[(agent, problem_id)])
                    for agent, problem_id in sorted(required_keys)
                },
                "nvidia_key_slots": NVIDIA_RECOVERY_KEY_SLOTS,
                "key_values_exported": False,
                "methodology_warning": (
                    "Protocol-format recovery is a repaired conditional dataset and must not "
                    "replace raw operational completion reporting."
                ),
            }
            (export_dir / "recovery_audit.json").write_text(
                json.dumps(recovery_audit, indent=2), encoding="utf-8"
            )

            data_files = [
                "trajectories.jsonl", "problems.jsonl", "generation_health.csv",
                "required_reruns.csv", "unresolved_after_recovery.csv",
                "recovery_attempts.jsonl", "recovery_audit.json",
            ]
            export_manifest = {
                "artifact_type": "fully_recovered_generation",
                "raw_data_schema_version": source_manifest.get("raw_data_schema_version", 1),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "run_name": export_name,
                "source_run": source_manifest.get("run_name"),
                "models": source_manifest.get("models", []),
                "trajectory_count": len(merged_records),
                "problem_count": len(source_problems),
                "publishable_count": len(merged_records) - len(remaining_invalid),
                "data_files": data_files,
                "sha256": {
                    name: hashlib.sha256((export_dir / name).read_bytes()).hexdigest()
                    for name in data_files
                },
            }
            (export_dir / "generation_manifest.json").write_text(
                json.dumps(export_manifest, indent=2), encoding="utf-8"
            )
            recovery_zip = Path(shutil.make_archive(
                str(export_dir), "zip", root_dir=export_dir
            ))
            print("Recovery export:", recovery_zip)
            show_download_link(recovery_zip)

            PUBLISHABLE_COMPLETE = len(merged_records) == 1800 and not remaining_invalid
            print("PUBLISHABLE_COMPLETE =", PUBLISHABLE_COMPLETE)
            '''
        ),
        code(
            r'''
            # Final publication gate. Do not calculate final paper metrics unless this passes.
            if REQUIRE_ALL_PUBLISHABLE and not PUBLISHABLE_COMPLETE:
                raise RuntimeError(
                    f"Recovery is incomplete: {len(remaining_invalid)} invalid trajectories remain. "
                    "Rerun the recovery engine cell; checkpoints preserve accepted replacements."
                )
            assert len(merged_records) == 1800
            assert len({(item["agent_name"], item["problem_id"]) for item in merged_records}) == 1800
            assert all(is_publishable_trajectory(Trajectory.from_dict(item)) for item in merged_records)
            print("FINAL GATE PASSED: exactly 1,800 unique publishable trajectories.")
            print("Use this recovery ZIP as the only input to strive-metrics-from-trajectories.ipynb")
            '''
        ),
        markdown(
            """
            ## 10. Rerun scope

            On the first execution, run the notebook from top to bottom.

            If the final gate reports unresolved trajectories, change only
            `MAX_RECOVERY_ATTEMPTS_PER_TRAJECTORY` if desired, then rerun:

            1. **Recovery engine cell** in Section 8.
            2. **Merge/export cell** in Section 9.
            3. **Final publication gate** in Section 9.

            Do not rerun the source generation, sandbox tests, or already accepted
            trajectories. The append-only recovery checkpoint resumes automatically.

            After the gate passes, use the recovered ZIP as the sole `TRAJECTORY_INPUTS`
            entry in `strive-metrics-from-trajectories.ipynb`. Then pass that notebook's
            metrics ZIP to `strive-advanced-metric-analysis.ipynb`.
            """
        ),
    ]
)

for index, cell in enumerate(cells):
    cell["id"] = f"strive-recover-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": copy.deepcopy(base.get("metadata", {})),
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
