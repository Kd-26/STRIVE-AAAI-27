#!/usr/bin/env python3
"""Build the cleaned STRIVE multi-model evaluation notebook."""

import json
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "strive-maths-final-core-metrics-multimodel.ipynb"
SECTION2_OUT = ROOT / "outputs" / "strive_section2_configuration_cell.py"
ANALYSIS_OUT = ROOT / "outputs" / "strive-results-analysis-only.ipynb"


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
        # STRIVE-Math: Six-Model Multi-Step Evaluation

        Clean paper-run notebook for 5 NVIDIA NIM models and 1 OpenAI model.

        - Main set: 200 MATH-500 problems + 100 text-only, single-answer OlympiadBench problems.
        - Smoke set: 5 calibration problems sampled from rows disjoint from the frozen main set.
        - Scheduling: six concurrent model lanes. Each model processes problems sequentially with its own
          key, while faster models advance without waiting for slower model lanes.
        - Each trajectory is sequential: model action -> real sandbox observation -> next model action.
        - Core metrics: correctness C, grounding G, verified success V, calibrated step quality Q,
          process-aware token utility PTU, and harmful redundancy R_harmful.
        - Latency is decomposed into provider service, RPM pacing, retry backoff, and total time.
        - Final exports preserve trajectories, model inputs/outputs, tool traces, both PRMs, judge
          records, selected problems, metrics, and checksums for later analysis-only sessions.
        - Information gain is not a core metric. The composite E is optional only.

        The subprocess sandbox is appropriate for trusted benchmark-generated math code. It applies an
        AST allow-list, resource limits, a clean environment, and per-trajectory working directories,
        but it is not a hardened container boundary for adversarial code.
        """
    ),
    md("## 1. Setup"),
    code(
        r'''
        # Run once, then restart the kernel if pip reports dependency changes.
        %pip install -q "openai>=1.68,<3" "datasets>=3.2,<5" "transformers>=4.46,<5" \
            "accelerate>=1.2" "bitsandbytes>=0.45" "sentence-transformers>=3.3" \
            "math-verify>=0.7" "antlr4-python3-runtime==4.11.*" sympy scipy \
            pandas matplotlib seaborn tqdm tiktoken
        '''
    ),
    code(
        r'''
        import ast
        import contextlib
        import gc
        import hashlib
        import io
        import json
        import math
        import os
        import random
        import re
        import shutil
        import subprocess
        import sys
        import tempfile
        import threading
        import time
        import traceback

        from collections import defaultdict
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from dataclasses import asdict, dataclass, field, replace
        from datetime import datetime, timezone
        from pathlib import Path
        from typing import Any, Dict, List, Optional, Tuple

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import torch
        import openai
        import transformers
        from datasets import load_dataset
        from tqdm.auto import tqdm

        random.seed(42)
        np.random.seed(42)
        plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
        print(
            f"Runtime versions: Python={sys.version.split()[0]}, openai={openai.__version__}, "
            f"transformers={transformers.__version__}, torch={torch.__version__}, "
            f"CUDA={torch.cuda.is_available()}"
        )
        '''
    ),
    md("## 2. Experiment configuration"),
    code(
        r'''
        @dataclass(frozen=True)
        class ModelSpec:
            name: str
            provider: str
            model_id: str
            api_key_env: str
            rpm: int
            base_url: Optional[str] = None
            request_params: Dict[str, Any] = field(default_factory=dict)
            system_directive: str = ""


        @dataclass
        class Config:
            protocol_version: str = "strive-react-v11"
            seed: int = 42
            math_count: int = 200
            olympiad_count: int = 100
            smoke_count: int = 5
            # None processes the complete 200+100 set. Set an integer for a balanced smaller paper-run subset.
            paper_sample_limit: Optional[int] = None
            # Fair controller budget shared by every generation model:
            # at most four executed tool actions followed by up to two answer attempts.
            max_steps: int = 6
            max_tool_actions: int = 4
            max_consecutive_format_errors: int = 2
            max_repeated_actions: int = 2
            max_answer_only_violations: int = 1
            # Shared visible+provider output allowance. The v4 smoke showed that 1024 tokens
            # truncated three GPT-OSS actions before any protocol action became visible.
            max_output_tokens_per_step: int = 2048
            temperature: float = 0.0
            request_timeout_sec: int = 90
            request_total_timeout_sec: int = 150
            trajectory_timeout_sec: int = 300
            max_retries: int = 3
            retry_base_sec: float = 2.0
            max_retry_wait_sec: float = 70.0
            rate_limit_cooldown_sec: float = 60.0
            resource_exhausted_cooldown_sec: float = 20.0
            lane_api_error_cooldown_sec: float = 30.0
            lane_breaker_after_failures: int = 3
            lane_breaker_cooldown_sec: float = 120.0
            rpm_utilization: float = 0.90

            sandbox_timeout_sec: int = 20
            sandbox_memory_mb: int = 3072
            max_observation_chars: int = 8000
            max_history_chars: int = 24000

            checkpoint_every_problems: int = 1
            parallel_models: int = 6
            paper_batch_count: int = 3
            retry_infrastructure_failures_on_resume: bool = True
            live_trace: bool = True
            log_api_retries: bool = True
            store_full_model_inputs: bool = True
            require_distinct_nvidia_keys: bool = True
            preflight_probe_endpoints: bool = True
            preflight_probe_tokens: int = 32
            generation_probe_timeout_sec: int = 45
            evaluator_probe_timeout_sec: int = 150
            smoke_min_completion_rate: float = 0.80
            # Recovered truncations remain reported but do not invalidate smoke by themselves.
            smoke_max_truncated_steps_per_model: Optional[int] = None

            # The two PRMs are loaded one at a time. Four-bit loading fits a 16 GB GPU.
            prm_load_in_4bit: bool = True
            prm_max_length: int = 4096
            prm_models: list = field(default_factory=lambda: [
                {
                    "key": "math_shepherd_mistral_7b",
                    "model_name": "peiyi9979/math-shepherd-mistral-7b-prm",
                    "type": "math_shepherd",
                },
                {
                    "key": "qwen25_math_prm_7b",
                    "model_name": "Qwen/Qwen2.5-Math-PRM-7B",
                    "type": "qwen_prm",
                    # Pin the reviewed remote-code revision for paper reproducibility.
                    "revision": "0610740060112df12585d00a1c5f4624d2f59051",
                },
            ])

            sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
            semantic_threshold: float = 0.85
            prm_delta_scale: float = 0.10
            progressive_threshold: float = 0.20
            regressive_threshold: float = -0.20

            # Hybrid step score. Missing critic evidence is neutral (0), not a penalty.
            w_prm: float = 0.45
            w_critic: float = 0.25
            w_tool_gain: float = 0.15
            w_redundancy: float = -0.25
            w_error: float = -0.50

            # Selective LLM review is used only for deterministic-inconclusive correctness and
            # near-boundary non-answer steps. Grounding remains deterministic provenance tracing.
            use_critic_llm: bool = True
            use_correctness_judge_fallback: bool = True
            critic_model_id: str = "stepfun-ai/step-3.5-flash"
            # Slot 6 is a dedicated judge credential, independent of every generation lane.
            judge_key_slot: int = 6  # Choose 1..6 explicitly; use 0 for NVIDIA_CRITIC_API_KEY.
            judge_rpm: int = 40

            output_dir: str = field(default_factory=lambda: (
                "/kaggle/working/strive_math_v11"
                if Path("/kaggle/working").exists()
                else str(Path.cwd() / "outputs" / "strive_math_v11")
            ))


        cfg = Config()
        cfg.output_dir = str(Path(cfg.output_dir).resolve())
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.output_dir, "checkpoints").mkdir(parents=True, exist_ok=True)

        REQUIRED_SECRET_NAMES = [
            "NVIDIA_API_KEY_1",
            "NVIDIA_API_KEY_2",
            "NVIDIA_API_KEY_3",
            "NVIDIA_API_KEY_4",
            "NVIDIA_API_KEY_5",
            "OPENAI_API_KEY",
        ]
        OPTIONAL_SECRET_NAMES = ["NVIDIA_API_KEY_6", "NVIDIA_CRITIC_API_KEY"]


        def load_api_secrets(secret_names=REQUIRED_SECRET_NAMES):
            """Load Kaggle Secrets into environment variables without printing secret values."""
            loaded = []
            if Path("/kaggle/working").exists():
                try:
                    from kaggle_secrets import UserSecretsClient
                    secret_client = UserSecretsClient()
                    for name in secret_names:
                        if os.environ.get(name):
                            loaded.append(name)
                            continue
                        try:
                            value = secret_client.get_secret(name)
                            if value:
                                os.environ[name] = value
                                loaded.append(name)
                        except Exception:
                            pass
                except Exception:
                    pass
            else:
                loaded = [name for name in secret_names if os.environ.get(name)]
            print(f"API secrets available: {len(loaded)}/{len(secret_names)} (values are never displayed)")
            return loaded


        # Edit only this block when entering credentials directly in the notebook. Leave a value as
        # an empty string to use an existing environment variable or Kaggle Secret with the same name.
        # Before sharing or publishing the notebook, clear these values and its saved cell output.
        API_KEYS = {
            "NVIDIA_API_KEY_1": "",  # Ministral
            "NVIDIA_API_KEY_2": "",  # GLM
            "NVIDIA_API_KEY_3": "",  # MiniMax M3
            "NVIDIA_API_KEY_4": "",  # Nemotron
            "NVIDIA_API_KEY_5": "",  # GPT-OSS
            "NVIDIA_API_KEY_6": "",  # Dedicated Gemma judge key (or explicit generation standby)
            "OPENAI_API_KEY": "",    # GPT-5 Nano
            # Optional dedicated judge key. If empty, the evaluator uses NVIDIA_API_KEY_5.
            "NVIDIA_CRITIC_API_KEY": "",
        }

        load_api_secrets(REQUIRED_SECRET_NAMES + OPTIONAL_SECRET_NAMES)
        for secret_name, secret_value in API_KEYS.items():
            if str(secret_value).strip():
                os.environ[secret_name] = str(secret_value).strip()
        if cfg.judge_key_slot not in {0, 1, 2, 3, 4, 5, 6}:
            raise ValueError("cfg.judge_key_slot must be 0 (dedicated key) or one of 1..6")
        configured_count = sum(bool(os.environ.get(name, "").strip()) for name in REQUIRED_SECRET_NAMES)
        print(f"API keys configured: {configured_count}/{len(REQUIRED_SECRET_NAMES)}")

        # Each NVIDIA model uses a distinct key slot.
        MODEL_SPECS = [
            ModelSpec(
                name="ministral-14b",
                provider="nvidia",
                model_id="mistralai/ministral-14b-instruct-2512",
                api_key_env="NVIDIA_API_KEY_1",
                rpm=12,
                base_url="https://integrate.api.nvidia.com/v1",
                request_params={
                    "temperature": 0.15,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                },
            ),
            ModelSpec(
                name="glm-5.2",
                provider="nvidia",
                model_id="z-ai/glm-5.2",
                api_key_env="NVIDIA_API_KEY_2",
                # 6 RPM at 90% utilization gives an 11.1-second minimum request gap.
                rpm=6,
                base_url="https://integrate.api.nvidia.com/v1",
                request_params={"temperature": 0.20, "top_p": 0.95, "seed": 42},
            ),
            ModelSpec(
                name="minimax-m3",
                provider="nvidia",
                model_id="minimaxai/minimax-m3",
                api_key_env="NVIDIA_API_KEY_3",
                rpm=12,
                base_url="https://integrate.api.nvidia.com/v1",
                request_params={
                    "temperature": 1.0,
                    "top_p": 0.95,
                },
            ),
            ModelSpec(
                name="nemotron-3-nano",
                provider="nvidia",
                model_id="nvidia/nemotron-3-nano-30b-a3b",
                api_key_env="NVIDIA_API_KEY_4",
                rpm=20,
                base_url="https://integrate.api.nvidia.com/v1",
                request_params={
                    "temperature": 0.20,
                    "top_p": 0.95,
                    # Instruct mode avoids spending the whole action budget in reasoning_content.
                    "extra_body": {
                        "top_k": 1,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                },
            ),
            ModelSpec(
                name="gpt-oss-20b",
                provider="nvidia",
                model_id="openai/gpt-oss-20b",
                api_key_env="NVIDIA_API_KEY_5",
                rpm=12,
                base_url="https://integrate.api.nvidia.com/v1",
                request_params={
                    "temperature": 1.0,
                    "top_p": 1.0,
                    # NVIDIA's real control; the text directive alone did not constrain reasoning_content.
                    "reasoning_effort": "low",
                },
                system_directive="Reasoning: low",
            ),
            ModelSpec(
                name="gpt-5-nano",
                provider="openai",
                model_id="gpt-5-nano",
                api_key_env="OPENAI_API_KEY",
                rpm=60,
                request_params={"reasoning_effort": "minimal"},
            ),
        ]

        # Optional explicit standby assignment. This never rotates automatically on HTTP 429.
        # Example: {"minimax-m3": "NVIDIA_API_KEY_6"}
        NVIDIA_MODEL_KEY_OVERRIDES = {}
        allowed_model_key_slots = {f"NVIDIA_API_KEY_{i}" for i in range(1, 7)}
        known_model_names = {spec.name for spec in MODEL_SPECS if spec.provider == "nvidia"}
        unknown_models = set(NVIDIA_MODEL_KEY_OVERRIDES) - known_model_names
        invalid_slots = set(NVIDIA_MODEL_KEY_OVERRIDES.values()) - allowed_model_key_slots
        if unknown_models or invalid_slots:
            raise ValueError(
                f"Invalid NVIDIA_MODEL_KEY_OVERRIDES: unknown_models={unknown_models}, "
                f"invalid_key_slots={invalid_slots}"
            )
        MODEL_SPECS = [
            replace(spec, api_key_env=NVIDIA_MODEL_KEY_OVERRIDES.get(spec.name, spec.api_key_env))
            for spec in MODEL_SPECS
        ]

        print(f"Output directory: {cfg.output_dir}")
        print(f"Paper set: {cfg.math_count} MATH + {cfg.olympiad_count} OlympiadBench")
        print(f"Processing limit: {cfg.paper_sample_limit or 'all selected problems'}")
        pd.DataFrame([asdict(x) | {"base_url": x.base_url or "OpenAI"} for x in MODEL_SPECS])
        '''
    ),
    md("## 3. Dataset loading"),
    code(
        r'''
        def balanced_take(rows: List[Dict], n: int, group_keys: Tuple[str, ...], seed: int) -> List[Dict]:
            """Deterministic round-robin sample across available subject/level groups."""
            rng = random.Random(seed)
            groups = defaultdict(list)
            for row in rows:
                groups[tuple(str(row.get(k, "unknown")) for k in group_keys)].append(row)
            for values in groups.values():
                rng.shuffle(values)
            keys = sorted(groups)
            rng.shuffle(keys)
            chosen = []
            while len(chosen) < n and keys:
                next_keys = []
                for key in keys:
                    if groups[key] and len(chosen) < n:
                        chosen.append(groups[key].pop())
                    if groups[key]:
                        next_keys.append(key)
                keys = next_keys
            if len(chosen) != n:
                raise ValueError(f"Requested {n} examples, found only {len(chosen)} eligible rows")
            return chosen


        def first_answer(value: Any) -> str:
            if isinstance(value, (list, tuple)):
                return str(value[0]).strip() if value else ""
            return str(value or "").strip()


        def load_paper_problem_sets(cfg: Config) -> Tuple[List[Dict], List[Dict]]:
            """Return the frozen paper set and a disjoint pool for smoke/calibration runs."""
            math_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
            all_math_rows = [
                {
                    "id": f"math500:{row.get('unique_id', i)}",
                    "dataset": "MATH-500",
                    "problem": row["problem"],
                    "gold_answer": str(row["answer"]),
                    "reference_solution": str(row.get("solution", "")),
                    "subject": str(row.get("subject", "unknown")),
                    "level": str(row.get("level", "unknown")),
                }
                for i, row in enumerate(math_ds)
            ]
            math_rows = balanced_take(
                all_math_rows,
                cfg.math_count,
                ("subject", "level"),
                cfg.seed,
            )

            olympiad_ds = load_dataset(
                "Hothan/OlympiadBench",
                "OE_TO_maths_en_COMP",
                split="train",
            )
            all_olympiad_rows = []
            for i, row in enumerate(olympiad_ds):
                if row.get("error") not in (None, "", False):
                    continue
                if bool(row.get("is_multiple_answer", False)):
                    continue
                answer = first_answer(row.get("final_answer"))
                question = str(row.get("question", "")).strip()
                if not question or not answer:
                    continue
                all_olympiad_rows.append({
                    "id": f"olympiad:{row.get('id', i)}",
                    "dataset": "OlympiadBench-OE-TO-maths-en-COMP",
                    "problem": question,
                    "gold_answer": answer,
                    "reference_solution": first_answer(row.get("solution")),
                    "subject": str(row.get("subfield", "Olympiad math")),
                    "level": str(row.get("difficulty", "Competition")),
                })
            olympiad_rows = balanced_take(
                all_olympiad_rows,
                cfg.olympiad_count,
                ("subject",),
                cfg.seed + 1,
            )
            rows = math_rows + olympiad_rows
            assert len(rows) == cfg.math_count + cfg.olympiad_count
            paper_ids = {row["id"] for row in rows}
            smoke_pool = [
                row for row in all_math_rows + all_olympiad_rows
                if row["id"] not in paper_ids
            ]
            return rows, smoke_pool


        def choose_smoke_problems(problems: List[Dict], n: int = 5) -> List[Dict]:
            """Use 3 MATH and 2 OlympiadBench tasks with different subjects when possible."""
            selected, seen = [], set()
            targets = [("MATH-500", 3), ("OlympiadBench", 2)]
            for dataset_prefix, count in targets:
                candidates = [p for p in problems if p["dataset"].startswith(dataset_prefix)]
                for p in candidates:
                    key = (p["dataset"], p["subject"])
                    if key not in seen:
                        selected.append(p)
                        seen.add(key)
                    if sum(x["dataset"].startswith(dataset_prefix) for x in selected) >= count:
                        break
            return selected[:n]


        def choose_processing_problems(problems: List[Dict], limit: Optional[int], seed: int) -> List[Dict]:
            if limit is None:
                return list(problems)
            limit = int(limit)
            if limit < 1 or limit > len(problems):
                raise ValueError(f"paper_sample_limit must be between 1 and {len(problems)}, got {limit}")
            return balanced_take(problems, limit, ("dataset", "subject"), seed + 10)


        def stratified_problem_batches(
            problems: List[Dict], batch_count: int, seed: int
        ) -> List[List[Dict]]:
            """Distribute every dataset/subject group round-robin so each batch has similar difficulty."""
            if batch_count < 1:
                raise ValueError("batch_count must be positive")
            rng = random.Random(seed)
            groups = defaultdict(list)
            for problem in problems:
                groups[(problem["dataset"], problem.get("subject", "unknown"))].append(problem)
            batches = [[] for _ in range(batch_count)]
            offset = 0
            for key in sorted(groups):
                values = list(groups[key])
                rng.shuffle(values)
                for index, problem in enumerate(values):
                    batches[(offset + index) % batch_count].append(problem)
                offset = (offset + len(values)) % batch_count
            for batch in batches:
                rng.shuffle(batch)
            flattened = [p["id"] for batch in batches for p in batch]
            assert len(flattened) == len(set(flattened)) == len(problems)
            return batches


        paper_problems, smoke_pool = load_paper_problem_sets(cfg)
        experiment_problems = choose_processing_problems(paper_problems, cfg.paper_sample_limit, cfg.seed)
        paper_batches = stratified_problem_batches(
            experiment_problems, cfg.paper_batch_count, cfg.seed + 20
        )
        smoke_problems = choose_smoke_problems(smoke_pool, cfg.smoke_count)
        assert len(smoke_problems) == cfg.smoke_count
        assert {p["id"] for p in paper_problems}.isdisjoint({p["id"] for p in smoke_problems})
        display(pd.DataFrame(paper_problems)[["dataset", "subject", "level"]].value_counts().reset_index(name="n"))
        display(pd.DataFrame(smoke_problems)[["id", "dataset", "subject", "level"]])
        print(f"Full-run processing count: {len(experiment_problems)}")
        print("Paper generation batches:", [len(batch) for batch in paper_batches])
        if cfg.paper_sample_limit is None:
            assert [len(batch) for batch in paper_batches] == [100, 100, 100]
        print("Smoke/paper overlap: 0 problems (asserted)")
        '''
    ),
]

cells.extend([
    md("## 4. Stateful multi-step Python sandbox"),
    code(
        r'''
        SAFE_IMPORT_ROOTS = {
            "math", "cmath", "fractions", "decimal", "statistics", "itertools",
            "collections", "functools", "operator", "sympy", "numpy", "scipy",
        }
        BLOCKED_CALLS = {
            "open", "exec", "eval", "compile", "input", "breakpoint", "__import__",
            "globals", "locals", "vars", "help", "dir",
        }


        def validate_math_code(source: str) -> Tuple[bool, str]:
            """Reject filesystem, network, process, and introspection primitives before execution."""
            try:
                tree = ast.parse(source)
            except SyntaxError as exc:
                return False, f"SyntaxError: {exc}"
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] not in SAFE_IMPORT_ROOTS:
                            return False, f"Blocked import: {alias.name}"
                elif isinstance(node, ast.ImportFrom):
                    root = (node.module or "").split(".")[0]
                    if root not in SAFE_IMPORT_ROOTS:
                        return False, f"Blocked import: {node.module}"
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in BLOCKED_CALLS:
                        return False, f"Blocked call: {node.func.id}"
                elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                    return False, f"Blocked private attribute: {node.attr}"
            return True, "ok"


        def capture_final_expression(source: str) -> str:
            """Give subprocess execution Jupyter-like display semantics for a final expression."""
            try:
                tree = ast.parse(source)
                if not tree.body or not isinstance(tree.body[-1], ast.Expr):
                    return source
                value = tree.body[-1].value
                already_printed = (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "print"
                )
                if already_printed:
                    return source
                tree.body[-1] = ast.Expr(
                    value=ast.Call(func=ast.Name(id="print", ctx=ast.Load()), args=[value], keywords=[])
                )
                ast.fix_missing_locations(tree)
                return ast.unparse(tree)
            except Exception:
                return source


        def truncate_middle(text: str, max_chars: int) -> str:
            text = str(text or "")
            if len(text) <= max_chars:
                return text
            half = max(1, (max_chars - 80) // 2)
            return text[:half] + "\n...[observation truncated]...\n" + text[-half:]


        def _apply_resource_limits(memory_mb: int, cpu_seconds: int):
            if os.name != "posix":
                return
            try:
                import resource
                memory = int(memory_mb) * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 2))
                resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
                if hasattr(resource, "RLIMIT_NPROC"):
                    resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
            except Exception:
                pass


        SAFE_PREAMBLE = r"""
        import math
        import cmath
        import sympy
        import numpy as np
        from decimal import Decimal
        from fractions import Fraction
        from itertools import combinations, permutations, product
        from collections import Counter, defaultdict
        from sympy import *

        np.random.seed(0)
        x, y, z, a, b, c, n, k, i, j, t = sympy.symbols("x y z a b c n k i j t")
        """


        @dataclass
        class SandboxResult:
            ok: bool
            stdout: str
            stderr: str
            returncode: int
            elapsed_sec: float
            timed_out: bool = False
            blocked: bool = False

            def observation(self, max_chars: int) -> str:
                status = "SUCCESS" if self.ok else "FAILURE"
                parts = [f"Execution status: {status}", f"Elapsed: {self.elapsed_sec:.3f}s"]
                if self.stdout.strip():
                    parts.append("STDOUT:\n" + self.stdout.strip())
                if self.stderr.strip():
                    parts.append("STDERR:\n" + self.stderr.strip())
                if self.timed_out:
                    parts.append("TimeoutError: execution exceeded the step limit.")
                if not self.stdout.strip() and not self.stderr.strip() and self.ok:
                    parts.append("No printed output. Use print(...) to expose a result.")
                return truncate_middle("\n".join(parts), max_chars)


        class MultiStepPythonSandbox:
            """Fresh subprocess per action with successful code replayed silently."""

            def __init__(self, cfg: Config):
                self.cfg = cfg
                self.workdir = Path(tempfile.mkdtemp(prefix="strive_math_"))
                self.successful_blocks: List[str] = []
                self.closed = False

            def _runner_source(self, current: str) -> str:
                history = "\n\n".join(self.successful_blocks)
                return f"""{SAFE_PREAMBLE}
        import contextlib
        import io
        import sys
        import traceback

        _globals = globals()
        _history = {history!r}
        _current = {current!r}
        try:
            if _history.strip():
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exec(_history, _globals, _globals)
            exec(_current, _globals, _globals)
        except BaseException:
            traceback.print_exc()
            sys.exit(1)
        """

            def run(self, source: str) -> SandboxResult:
                allowed, reason = validate_math_code(source)
                if not allowed:
                    return SandboxResult(False, "", reason, 126, 0.0, blocked=True)

                execution_source = capture_final_expression(source)
                script = self.workdir / "runner.py"
                script.write_text(self._runner_source(execution_source), encoding="utf-8")
                clean_env = {
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONPATH": "",
                    "PYTHONHASHSEED": "0",
                    "HOME": str(self.workdir),
                    "TMPDIR": str(self.workdir),
                    "MPLBACKEND": "Agg",
                }
                start = time.perf_counter()
                try:
                    proc = subprocess.run(
                        [sys.executable, "-I", str(script)],
                        cwd=self.workdir,
                        env=clean_env,
                        text=True,
                        capture_output=True,
                        timeout=self.cfg.sandbox_timeout_sec,
                        preexec_fn=(
                            lambda: _apply_resource_limits(
                                self.cfg.sandbox_memory_mb,
                                self.cfg.sandbox_timeout_sec,
                            )
                        ) if os.name == "posix" else None,
                    )
                    result = SandboxResult(
                        ok=proc.returncode == 0,
                        stdout=proc.stdout,
                        stderr=proc.stderr,
                        returncode=proc.returncode,
                        elapsed_sec=time.perf_counter() - start,
                    )
                except subprocess.TimeoutExpired as exc:
                    result = SandboxResult(
                        ok=False,
                        stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
                        stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else "",
                        returncode=124,
                        elapsed_sec=time.perf_counter() - start,
                        timed_out=True,
                    )
                if result.ok:
                    self.successful_blocks.append(source)
                return result

            def close(self):
                if not self.closed:
                    shutil.rmtree(self.workdir, ignore_errors=True)
                    self.closed = True

            def __enter__(self):
                return self

            def __exit__(self, *_):
                self.close()
        '''
    ),
    code(
        r'''
        # Deterministic infrastructure test: state persists, failed code is not committed,
        # stdout is not replayed, and unsafe imports are blocked.
        with MultiStepPythonSandbox(cfg) as sb:
            r1 = sb.run("u = 41\nprint('created', u)")
            r2 = sb.run("u += 1\nprint('answer', u)")
            r3 = sb.run("u = 999\nraise ValueError('do not commit this state')")
            r4 = sb.run("print('state_after_failure', u)")
            r5 = sb.run("import os\nprint(os.environ)")
            r6 = sb.run("u * 2")

        assert r1.ok and "created 41" in r1.stdout
        assert r2.ok and "answer 42" in r2.stdout and "created 41" not in r2.stdout
        assert not r3.ok
        assert r4.ok and "state_after_failure 42" in r4.stdout
        assert r5.blocked and "Blocked import" in r5.stderr
        assert r6.ok and r6.stdout.strip() == "84"
        print("Sandbox self-test passed: persistence, rollback, muted replay, final-expression display, and policy checks work.")
        '''
    ),
    md("## 5. API adapters, shared rate limits, and retries"),
    code(
        r'''
        from openai import APIConnectionError, APITimeoutError, OpenAI


        class PacedRateLimiter:
            """Smooth request starts to avoid a burst of 20 calls followed by a 60-second stall."""

            def __init__(self, rpm: int, utilization: float = 0.90):
                self.lock = threading.Lock()
                self.next_start = 0.0
                self.configure(rpm, utilization)

            def configure(self, rpm: int, utilization: float):
                self.rpm = max(1, int(rpm))
                self.utilization = float(np.clip(utilization, 0.1, 1.0))
                self.interval_sec = 60.0 / (self.rpm * self.utilization)

            def acquire(self) -> float:
                with self.lock:
                    now = time.monotonic()
                    scheduled = max(now, self.next_start)
                    self.next_start = scheduled + self.interval_sec
                    wait = max(0.0, scheduled - now)
                if wait:
                    time.sleep(wait)
                return wait


        _clients: Dict[Tuple[str, str, str], OpenAI] = {}
        _limiters: Dict[str, PacedRateLimiter] = {}
        _registry_lock = threading.Lock()


        def resolve_api_key(spec: ModelSpec) -> str:
            key = os.environ.get(spec.api_key_env, "").strip()
            if not key and spec.name == "external-nim-evaluator":
                key = os.environ.get("NVIDIA_API_KEY_5", "").strip()
            if not key and spec.provider == "nvidia" and not cfg.require_distinct_nvidia_keys:
                key = os.environ.get("NVIDIA_API_KEY", "").strip()
            if not key:
                raise RuntimeError(
                    f"Missing API key for {spec.name}. Set {spec.api_key_env}"
                    + (
                        " (distinct NVIDIA key slots are required)"
                        if spec.provider == "nvidia" and cfg.require_distinct_nvidia_keys
                        else (" or NVIDIA_API_KEY" if spec.provider == "nvidia" else "")
                    )
                )
            return key


        def client_and_limiter(spec: ModelSpec):
            key = resolve_api_key(spec)
            fingerprint = hashlib.sha256(key.encode()).hexdigest()[:16]
            client_key = (spec.provider, spec.base_url or "OpenAI", fingerprint)
            limiter_key = f"{spec.provider}:{fingerprint}"
            with _registry_lock:
                if client_key not in _clients:
                    # Disable SDK-internal retries; the explicit loop below is the sole retry policy.
                    kwargs = {
                        "api_key": key,
                        "timeout": cfg.request_timeout_sec,
                        "max_retries": 0,
                    }
                    if spec.base_url:
                        kwargs["base_url"] = spec.base_url
                    _clients[client_key] = OpenAI(**kwargs)
                if limiter_key not in _limiters:
                    _limiters[limiter_key] = PacedRateLimiter(spec.rpm, cfg.rpm_utilization)
                else:
                    # If several models share a key, respect the most conservative configured RPM.
                    conservative_rpm = min(_limiters[limiter_key].rpm, spec.rpm)
                    _limiters[limiter_key].configure(conservative_rpm, cfg.rpm_utilization)
            return _clients[client_key], _limiters[limiter_key]


        def usage_value(obj, name: str, default: int = 0) -> int:
            value = getattr(obj, name, default) if obj is not None else default
            return int(value or 0)


        def extract_openai_response_text(response) -> str:
            """Support both the SDK output_text helper and explicit Responses output items."""
            direct = getattr(response, "output_text", "") or ""
            if direct.strip():
                return direct.strip()
            chunks = []
            for item in getattr(response, "output", None) or []:
                for part in getattr(item, "content", None) or []:
                    text_value = getattr(part, "text", None)
                    if text_value:
                        chunks.append(str(text_value))
            return "\n".join(chunks).strip()


        class ModelCallError(RuntimeError):
            def __init__(self, message: str, meta: Dict):
                super().__init__(message)
                self.meta = meta


        def call_model(
            spec: ModelSpec,
            messages: List[Dict],
            cfg: Config,
            deadline_monotonic: Optional[float] = None,
            context: str = "",
            max_output_tokens_override: Optional[int] = None,
        ) -> Tuple[str, Dict]:
            client, limiter = client_and_limiter(spec)
            last_error = None
            call_started = time.monotonic()
            call_deadline = call_started + cfg.request_total_timeout_sec
            if deadline_monotonic is not None:
                call_deadline = min(call_deadline, deadline_monotonic)
            rate_wait_total = 0.0
            retry_wait_total = 0.0
            service_time_total = 0.0
            error_history = []
            output_budget = int(max_output_tokens_override or cfg.max_output_tokens_per_step)
            for attempt in range(cfg.max_retries):
                remaining = call_deadline - time.monotonic()
                if remaining <= 1.0:
                    last_error = TimeoutError("request retry deadline exhausted")
                    break
                rate_wait = limiter.acquire()
                rate_wait_total += rate_wait
                remaining = call_deadline - time.monotonic()
                if remaining <= 1.0:
                    last_error = TimeoutError("request deadline exhausted after local RPM pacing")
                    break
                attempt_timeout = max(1.0, min(float(cfg.request_timeout_sec), remaining))
                request_client = client.with_options(timeout=attempt_timeout, max_retries=0)
                start = time.perf_counter()
                try:
                    if spec.provider == "openai":
                        reasoning_effort = spec.request_params.get("reasoning_effort", "low")
                        response = request_client.responses.create(
                            model=spec.model_id,
                            input=messages,
                            max_output_tokens=output_budget,
                            reasoning={"effort": reasoning_effort},
                        )
                        service_latency = time.perf_counter() - start
                        service_time_total += service_latency
                        text = extract_openai_response_text(response)
                        usage = response.usage
                        details = getattr(usage, "output_tokens_details", None)
                        hidden = usage_value(details, "reasoning_tokens")
                        finish_reason = str(getattr(response, "status", "") or "")
                        meta = {
                            "input_tokens": usage_value(usage, "input_tokens"),
                            "output_tokens": usage_value(usage, "output_tokens"),
                            "hidden_reasoning_tokens": hidden,
                            "provider_reasoning": "",
                        }
                    else:
                        request_params = dict(spec.request_params)
                        configured_max = int(request_params.pop("max_tokens", output_budget))
                        max_tokens = min(configured_max, output_budget)
                        response = request_client.chat.completions.create(
                            model=spec.model_id,
                            messages=messages,
                            max_tokens=max_tokens,
                            stream=False,
                            **request_params,
                        )
                        service_latency = time.perf_counter() - start
                        service_time_total += service_latency
                        message = response.choices[0].message
                        text = message.content or ""
                        provider_reasoning = (
                            getattr(message, "reasoning", None)
                            or getattr(message, "reasoning_content", None)
                            or ""
                        )
                        usage = response.usage
                        details = getattr(usage, "completion_tokens_details", None)
                        hidden = usage_value(details, "reasoning_tokens")
                        finish_reason = str(response.choices[0].finish_reason or "")
                        meta = {
                            "input_tokens": usage_value(usage, "prompt_tokens"),
                            "output_tokens": usage_value(usage, "completion_tokens"),
                            "hidden_reasoning_tokens": hidden,
                            "provider_reasoning": provider_reasoning,
                        }
                    meta.update({
                        "latency_sec": time.monotonic() - call_started,
                        "service_latency_sec": service_latency,
                        "service_time_all_attempts_sec": service_time_total,
                        "rate_limit_wait_sec": rate_wait_total,
                        "retry_wait_sec": retry_wait_total,
                        "request_attempts": attempt + 1,
                        "retry_errors": error_history,
                        "finish_reason": finish_reason,
                        "provider_truncated": finish_reason.lower() in {
                            "length", "max_tokens", "incomplete"
                        },
                    })
                    return text.strip(), meta
                except Exception as exc:
                    service_time_total += time.perf_counter() - start
                    last_error = exc
                    status = getattr(exc, "status_code", None)
                    retryable = (
                        status in {408, 409, 429, 500, 502, 503, 504}
                        or isinstance(exc, (APIConnectionError, APITimeoutError))
                    )
                    error_history.append({
                        "attempt": attempt + 1,
                        "status": status,
                        "type": type(exc).__name__,
                        "message": truncate_middle(str(exc), 500),
                    })
                    if not retryable or attempt + 1 == cfg.max_retries:
                        break
                    retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("retry-after")
                    try:
                        delay = float(retry_after)
                    except (TypeError, ValueError):
                        if status == 429:
                            # A 2-4 second retry loop amplified the observed GLM quota failure.
                            delay = cfg.rate_limit_cooldown_sec + random.random() * 3.0
                        elif status == 503 and "ResourceExhausted" in str(exc):
                            delay = cfg.resource_exhausted_cooldown_sec + random.random() * 3.0
                        else:
                            delay = cfg.retry_base_sec * (2 ** attempt) + random.random()
                    delay = min(float(delay), cfg.max_retry_wait_sec)
                    delay = min(delay, max(0.0, call_deadline - time.monotonic() - 1.0))
                    if cfg.log_api_retries:
                        label = context or spec.name
                        tqdm.write(
                            f"[{label}] API retry {attempt + 2}/{cfg.max_retries} after "
                            f"{type(exc).__name__}; waiting {delay:.1f}s"
                        )
                    if delay > 0:
                        time.sleep(delay)
                        retry_wait_total += delay
            summary = truncate_middle(str(last_error), 1000)
            failure_meta = {
                "latency_sec": time.monotonic() - call_started,
                "service_latency_sec": 0.0,
                "service_time_all_attempts_sec": service_time_total,
                "rate_limit_wait_sec": rate_wait_total,
                "retry_wait_sec": retry_wait_total,
                "request_attempts": len(error_history),
                "retry_errors": error_history,
                "finish_reason": "error",
                "provider_truncated": False,
                "provider_reasoning": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "hidden_reasoning_tokens": 0,
            }
            raise ModelCallError(
                f"API call failed for {spec.name} after {len(error_history)} failed attempt(s): {summary}",
                failure_meta,
            )


        def probe_model_endpoint(spec: ModelSpec, cfg: Config) -> Dict:
            """Make one tiny real request so inaccessible model aliases fail before generation."""
            started = time.monotonic()
            timeout = (
                cfg.evaluator_probe_timeout_sec
                if spec.name == "external-nim-evaluator"
                else cfg.generation_probe_timeout_sec
            )
            try:
                _, meta = call_model(
                    spec,
                    [{"role": "user", "content": "Reply with exactly: OK"}],
                    cfg,
                    deadline_monotonic=started + min(float(timeout), cfg.request_total_timeout_sec),
                    context=f"preflight:{spec.name}",
                    max_output_tokens_override=cfg.preflight_probe_tokens,
                )
                return {
                    "endpoint": "reachable",
                    "probe_sec": round(float(meta.get("latency_sec", 0.0)), 2),
                    "probe_error": "",
                }
            except Exception as exc:
                retry_details = ""
                if isinstance(exc, ModelCallError):
                    retry_details = f" retries={exc.meta.get('retry_errors', [])}"
                return {
                    "endpoint": "FAILED",
                    "probe_sec": round(time.monotonic() - started, 2),
                    "probe_error": truncate_middle(str(exc) + retry_details, 900),
                }


        def preflight_models(specs: List[ModelSpec], probe_endpoints: Optional[bool] = None):
            probe_endpoints = cfg.preflight_probe_endpoints if probe_endpoints is None else probe_endpoints
            rows = []
            nvidia_fingerprints = []
            for spec in specs:
                try:
                    key = resolve_api_key(spec)
                    fingerprint = hashlib.sha256(key.encode()).hexdigest()
                    if spec.provider == "nvidia":
                        nvidia_fingerprints.append((spec.name, fingerprint))
                    row = {
                        "model": spec.name,
                        "provider": spec.provider,
                        "key_slot": spec.api_key_env,
                        "key": "configured",
                        "rpm": spec.rpm,
                    }
                    rows.append(row)
                except Exception as exc:
                    rows.append({
                        "model": spec.name,
                        "provider": spec.provider,
                        "key_slot": spec.api_key_env,
                        "key": str(exc),
                        "rpm": spec.rpm,
                        "endpoint": "not tested",
                        "probe_sec": 0.0,
                        "probe_error": "",
                    })
            frame = pd.DataFrame(rows)
            missing = frame[frame["key"] != "configured"]
            if not missing.empty:
                display(frame)
                raise RuntimeError(
                    "Missing API keys. Enter each value once in the API_KEYS block in Section 2, "
                    "then rerun Section 2 and this cell. No password prompt is used."
                )
            if cfg.require_distinct_nvidia_keys:
                fingerprints = [value for _, value in nvidia_fingerprints]
                if len(fingerprints) != len(set(fingerprints)):
                    raise RuntimeError(
                        "The five configured NVIDIA model lanes must resolve to five distinct authorized key values."
                    )
            if probe_endpoints:
                # Distinct model/key lanes are independent, so capability probes run concurrently.
                with ThreadPoolExecutor(max_workers=len(specs)) as pool:
                    futures = {pool.submit(probe_model_endpoint, spec, cfg): spec.name for spec in specs}
                    probe_results = {futures[future]: future.result() for future in as_completed(futures)}
                for row in rows:
                    row.update(probe_results[row["model"]])
                frame = pd.DataFrame(rows)
                display(frame)
                failed = frame[frame["endpoint"] != "reachable"]
                if not failed.empty:
                    details = failed[["model", "endpoint", "probe_error"]].to_dict("records")
                    raise RuntimeError(
                        "Model endpoint preflight failed. No benchmark trajectories were started. "
                        f"Fix or replace these endpoints: {details}"
                    )
            else:
                display(frame)
            return frame
        '''
    ),
])

cells.extend([
    md("## 8. Step quality, token utility, redundancy, and optional efficiency"),
    code(
        r'''
        def count_tokens(text: str) -> int:
            try:
                import tiktoken
                return len(tiktoken.get_encoding("o200k_base").encode(str(text or "")))
            except Exception:
                return max(0, len(str(text or "").split()))


        def math_normalize(text: str) -> str:
            text = normalize_answer(text)
            text = re.sub(r"\b([a-z])\b", "VAR", text)
            text = re.sub(r"\d+(?:\.\d+)?", "NUM", text)
            return text


        def step_text(step: Dict) -> str:
            return "\n".join([
                step.get("reasoning", ""),
                step.get("code", ""),
                step.get("stdout", ""),
                step.get("final_answer", ""),
            ]).strip()


        def load_sbert(cfg: Config):
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(cfg.sbert_model)


        def pairwise_redundancy(steps: List[Dict], sbert_model, cfg: Config) -> List[Dict]:
            texts = [step_text(s) for s in steps]
            normalized = [math_normalize(x) for x in texts]
            embeddings = sbert_model.encode(texts, normalize_embeddings=True) if texts else np.empty((0, 1))
            rows = []
            for i, step in enumerate(steps):
                if i == 0:
                    rows.append({"max_similarity": 0.0, "exact": False, "semantic": False})
                    continue
                sims = embeddings[:i] @ embeddings[i]
                max_sim = float(np.max(sims)) if len(sims) else 0.0
                exact = any(normalized[i] == normalized[j] and normalized[i] for j in range(i))
                rows.append({
                    "max_similarity": max_sim,
                    "exact": bool(exact),
                    "semantic": bool(max_sim >= cfg.semantic_threshold),
                })
            return rows


        def is_useful_verification(step: Dict, final_answer: str) -> bool:
            blob = (step.get("reasoning", "") + "\n" + step.get("code", "")).lower()
            verification_language = any(x in blob for x in (
                "verify", "check", "substitut", "simplif", "factor", "assert", "residual"
            ))
            output_matches = any(
                check_symbolic_equivalence(final_answer, candidate) is True
                for candidate in extract_output_candidates(step.get("stdout", ""))
            ) if final_answer else False
            return bool(step.get("code_success") and verification_language and output_matches)


        def is_algebraic_restatement(index: int, steps: List[Dict]) -> bool:
            if index <= 0:
                return False
            current = extract_output_candidates(steps[index].get("stdout", ""))
            previous = [
                candidate
                for step in steps[:index]
                for candidate in extract_output_candidates(step.get("stdout", ""))
            ]
            for left in current:
                for right in previous:
                    if normalize_answer(left) != normalize_answer(right):
                        if check_symbolic_equivalence(left, right) is True:
                            return True
            return False


        def first_solution_evidence_step(traj: Trajectory, cg: Dict) -> Optional[int]:
            if not cg["C_final"]:
                return None
            for index, step in enumerate(traj.steps):
                if step.get("action_type") == "code" and step.get("code_success"):
                    if any(
                        check_symbolic_equivalence(traj.final_answer, candidate) is True
                        for candidate in extract_output_candidates(step.get("stdout", ""))
                    ):
                        return index
                if step.get("action_type") == "answer":
                    return index
            return None


        def critic_step_fallback(traj: Trajectory, index: int, cfg: Config) -> Dict:
            prefix = "\n\n".join(
                f"Step {i + 1}:\n{format_step_for_prm(step)}"
                for i, step in enumerate(traj.steps[:index + 1])
            )
            prompt = f"""Classify only the final step in this trajectory prefix.
        Labels: progressive, neutral_useful, neutral_waste, redundant, regressive.

        Problem:
        {traj.problem_text}

        Prefix:
        {prefix}

        Return JSON only: {{"label": "one_label", "confidence": number_0_to_1, "reason": "brief"}}
        """
            try:
                raw, _ = call_model(
                    evaluator_spec(cfg),
                    [{
                        "role": "user",
                        "content": (
                            "Judge mathematical process steps independently of prose style. "
                            "Return only the requested JSON object.\n\n" + prompt
                        ),
                    }],
                    cfg,
                )
                parsed = extract_json_object(raw) or {}
                label = parsed.get("label")
                mapping = {
                    "progressive": 1.0,
                    "neutral_useful": 0.25,
                    "neutral_waste": 0.0,
                    "redundant": -0.5,
                    "regressive": -1.0,
                }
                return {
                    "used": label in mapping,
                    "signal": mapping.get(label, 0.0),
                    "label": label,
                    "confidence": float(parsed.get("confidence", 0.0)),
                    "raw": raw,
                }
            except Exception as exc:
                return {"used": False, "signal": 0.0, "error": str(exc)}


        def compute_step_quality(
            traj: Trajectory,
            prm_entry: Dict,
            sbert_model,
            cfg: Config,
        ) -> Dict:
            prm = list(prm_entry.get("ensemble", [0.5] * len(traj.steps)))
            if len(prm) < len(traj.steps):
                prm += [prm[-1] if prm else 0.5] * (len(traj.steps) - len(prm))
            redundant = pairwise_redundancy(traj.steps, sbert_model, cfg)
            details, labels = [], []
            previous_prm = 0.5
            previous_outputs = set()
            for index, step in enumerate(traj.steps):
                delta = float(prm[index]) - previous_prm
                previous_prm = float(prm[index])
                prm_signal = float(np.tanh(delta / max(cfg.prm_delta_scale, 1e-6)))
                output_norm = math_normalize(step.get("stdout", ""))
                new_output = bool(output_norm and output_norm not in previous_outputs)
                if output_norm:
                    previous_outputs.add(output_norm)
                tool_gain = 1.0 if step.get("code_success") and new_output else 0.0
                error_flag = float(
                    bool(step.get("error"))
                    or (step.get("action_type") == "code" and not step.get("code_success"))
                )
                useful_check = is_useful_verification(step, traj.final_answer)
                repetition = redundant[index]
                algebraic = is_algebraic_restatement(index, traj.steps)
                harmful_repeat = float(
                    step.get("action_type") != "answer"
                    and (repetition["exact"] or repetition["semantic"] or algebraic)
                    and not useful_check
                    and tool_gain == 0.0
                )
                critic = {"used": False, "signal": 0.0}
                base_score = (
                    cfg.w_prm * prm_signal
                    + cfg.w_tool_gain * tool_gain
                    + cfg.w_redundancy * harmful_repeat
                    + cfg.w_error * error_flag
                )
                if cfg.use_critic_llm and abs(base_score) < 0.08 and step.get("action_type") != "answer":
                    critic = critic_step_fallback(traj, index, cfg)
                critic_signal = float(critic.get("signal", 0.0))
                score = base_score + cfg.w_critic * critic_signal
                if error_flag or score <= cfg.regressive_threshold:
                    label = "regressive"
                elif harmful_repeat:
                    label = "redundant"
                elif score >= cfg.progressive_threshold:
                    label = "progressive"
                elif useful_check:
                    label = "neutral_useful"
                else:
                    label = "neutral_waste"
                labels.append(label)
                details.append({
                    "step_num": step.get("step_num", index + 1),
                    "prm_score": float(prm[index]),
                    "prm_delta": delta,
                    "prm_signal": prm_signal,
                    "critic_signal": critic_signal,
                    "critic_used": bool(critic.get("used", False)),
                    "critic_record": critic,
                    "tool_gain": tool_gain,
                    "max_similarity": repetition["max_similarity"],
                    "exact_repetition": repetition["exact"],
                    "semantic_repetition": repetition["semantic"],
                    "algebraic_restatement": algebraic,
                    "harmful_repeat": harmful_repeat,
                    "useful_verification": useful_check,
                    "error_flag": error_flag,
                    "hybrid_score": float(score),
                    "label": label,
                })
            class_value = {
                "progressive": 1.0,
                "neutral_useful": 0.5,
                "neutral_waste": 0.0,
                "redundant": -0.5,
                "regressive": -1.0,
            }
            signed_quality = float(np.mean([class_value[x] for x in labels])) if labels else -1.0
            q_step = float(np.clip((signed_quality + 1.0) / 2.0, 0.0, 1.0))
            rates = {name: labels.count(name) / max(len(labels), 1) for name in class_value}
            return {"Q_step": q_step, "signed_quality": signed_quality, "step_labels": labels,
                    "step_details": details, **{f"{k}_rate": v for k, v in rates.items()}}


        def compute_token_utility(traj: Trajectory, cg: Dict, quality: Dict) -> Dict:
            labels = quality["step_labels"]
            solution_index = first_solution_evidence_step(traj, cg)
            useful = pre_waste = post_waste = answer_reporting = 0
            visible_process = 0
            for index, (step, label) in enumerate(zip(traj.steps, labels)):
                output_tokens = max(0, int(step.get("output_tokens", 0)) - int(step.get("hidden_reasoning_tokens", 0)))
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

            api_input = sum(int(s.get("input_tokens", 0)) for s in traj.steps)
            api_output = sum(int(s.get("output_tokens", 0)) for s in traj.steps)
            hidden = sum(int(s.get("hidden_reasoning_tokens", 0)) for s in traj.steps)
            tool_output = sum(count_tokens(s.get("observation", "")) for s in traj.steps if s.get("action_type") == "code")
            code_tokens = sum(count_tokens(s.get("code", "")) for s in traj.steps)
            reasoning_tokens = sum(count_tokens(s.get("reasoning", "")) for s in traj.steps)
            denominator = max(useful + pre_waste + post_waste, 1)
            return {
                "PTU": useful / denominator,
                "T_api_input": api_input,
                "T_api_output": api_output,
                "T_billable": api_input + api_output,
                "T_hidden_reasoning": hidden,
                "T_tool_output": tool_output,
                "T_code": code_tokens,
                "T_visible_reasoning_est": reasoning_tokens,
                "T_visible_process": visible_process,
                "T_useful": useful,
                "T_pre_waste": pre_waste,
                "T_post_waste": post_waste,
                "T_answer_reporting": answer_reporting,
                "solution_evidence_step": None if solution_index is None else solution_index + 1,
                "PreWasteRate": pre_waste / denominator,
                "PostWasteRate": post_waste / denominator,
            }


        def compute_redundancy(quality: Dict) -> Dict:
            details = quality["step_details"]
            n = max(len(details), 1)
            exact = sum(d["exact_repetition"] for d in details) / n
            semantic = sum(d["semantic_repetition"] for d in details) / n
            algebraic = sum(d["algebraic_restatement"] for d in details) / n
            useful = sum(d["useful_verification"] for d in details) / n
            harmful = sum(d["harmful_repeat"] for d in details) / n
            circular = sum(d["harmful_repeat"] and d["tool_gain"] == 0 for d in details) / n
            return {
                "R_exact": exact,
                "R_semantic_raw": semantic,
                "R_algebraic_restatement": algebraic,
                "R_useful_verification": useful,
                "R_circular": circular,
                "R_harmful": harmful,
            }


        def compute_latency(traj: Trajectory) -> Dict:
            steps = traj.steps
            return {
                "L_trajectory_sec": float(traj.elapsed_sec),
                "L_request_total_sec": float(sum(s.get("request_latency_sec", 0.0) for s in steps)),
                "L_provider_service_sec": float(
                    sum(s.get("service_time_all_attempts_sec", 0.0) for s in steps)
                ),
                "L_rate_limit_wait_sec": float(sum(s.get("rate_limit_wait_sec", 0.0) for s in steps)),
                "L_retry_wait_sec": float(sum(s.get("retry_wait_sec", 0.0) for s in steps)),
                "request_attempts": int(sum(s.get("request_attempts", 0) for s in steps)),
                "retry_count": int(sum(max(0, s.get("request_attempts", 0) - 1) for s in steps)),
                "truncated_step_count": int(sum(bool(s.get("provider_truncated", False)) for s in steps)),
                "trajectory_timed_out": bool(traj.timed_out),
            }


        def compute_all_metrics(all_trajectories: Dict, prm_scores: Dict, sbert_model, cfg: Config) -> Dict:
            results = {}
            for agent, trajectories in all_trajectories.items():
                rows = []
                for traj in trajectories:
                    cg = compute_correctness_grounding(traj, cfg)
                    prm_entry = prm_scores.get(agent, {}).get(traj.problem_id, {"ensemble": [0.5] * traj.total_steps})
                    quality = compute_step_quality(traj, prm_entry, sbert_model, cfg)
                    tokens = compute_token_utility(traj, cg, quality)
                    redundancy = compute_redundancy(quality)
                    latency = compute_latency(traj)
                    e_optional = float(np.clip(
                        cg["V"] * tokens["PTU"] * quality["Q_step"] * (1.0 - redundancy["R_harmful"]),
                        0.0,
                        1.0,
                    ))
                    rows.append({
                        "problem_id": traj.problem_id,
                        "dataset": traj.dataset,
                        "subject": traj.subject,
                        "agent": agent,
                        "C_G_V": cg,
                        "Q": quality,
                        "T": tokens,
                        "R": redundancy,
                        "L": latency,
                        "E_optional": e_optional,
                        "finished": traj.finished,
                        "stop_reason": traj.stop_reason,
                        "total_steps": traj.total_steps,
                        "elapsed_sec": traj.elapsed_sec,
                        "final_answer": traj.final_answer,
                        "gold_answer": traj.gold_answer,
                    })
                results[agent] = rows
            return results
        '''
    ),
    md("## 9. Summary, plots, and exports"),
    code(
        r'''
        def summarize_metrics(metrics: Dict) -> pd.DataFrame:
            rows = []
            for agent, values in metrics.items():
                n = max(len(values), 1)
                verified = sum(x["C_G_V"]["V"] for x in values)
                billable = sum(x["T"]["T_billable"] for x in values)
                rows.append({
                    "agent": agent,
                    "n": len(values),
                    "C": np.mean([x["C_G_V"]["C_final"] for x in values]),
                    "G": np.mean([x["C_G_V"]["G"] for x in values]),
                    "V": np.mean([x["C_G_V"]["V"] for x in values]),
                    "Q": np.mean([x["Q"]["Q_step"] for x in values]),
                    "PTU": np.mean([x["T"]["PTU"] for x in values]),
                    "R_harmful": np.mean([x["R"]["R_harmful"] for x in values]),
                    "E_optional": np.mean([x["E_optional"] for x in values]),
                    "avg_billable_tokens": billable / n,
                    "tokens_per_verified_solve": billable / verified if verified else np.nan,
                    "avg_latency_sec": np.mean([x["elapsed_sec"] for x in values]),
                    "median_latency_sec": np.median([x["elapsed_sec"] for x in values]),
                    "p95_latency_sec": np.percentile([x["elapsed_sec"] for x in values], 95),
                    "avg_provider_service_sec": np.mean([
                        x["L"]["L_provider_service_sec"] for x in values
                    ]),
                    "avg_rate_limit_wait_sec": np.mean([
                        x["L"]["L_rate_limit_wait_sec"] for x in values
                    ]),
                    "avg_retry_wait_sec": np.mean([x["L"]["L_retry_wait_sec"] for x in values]),
                    "avg_steps": np.mean([x["total_steps"] for x in values]),
                    "completion_rate": np.mean([x["finished"] for x in values]),
                    "timeout_rate": np.mean([x["L"]["trajectory_timed_out"] for x in values]),
                    "retry_count": sum(x["L"]["retry_count"] for x in values),
                    "truncated_steps": sum(x["L"]["truncated_step_count"] for x in values),
                    "symbolic_inconclusive": sum(x["C_G_V"]["C_sym"] is None for x in values),
                    "judge_fallback_count": sum(x["C_G_V"]["judge_used"] for x in values),
                    "judge_queue": sum(x["C_G_V"]["needs_judge"] for x in values),
                    "critic_call_count": sum(
                        bool(detail.get("critic_record", {}).get("used"))
                        or bool(detail.get("critic_record", {}).get("raw"))
                        or bool(detail.get("critic_record", {}).get("error"))
                        for x in values for detail in x["Q"].get("step_details", [])
                    ),
                    "critic_error_count": sum(
                        bool(detail.get("critic_record", {}).get("error"))
                        for x in values for detail in x["Q"].get("step_details", [])
                    ),
                })
            return pd.DataFrame(rows).sort_values(["V", "C"], ascending=False).reset_index(drop=True)


        def validate_smoke_health(
            all_trajectories: Dict,
            summary: pd.DataFrame,
            prm_errors: List,
            cfg: Config,
        ) -> Tuple[pd.DataFrame, List[str]]:
            """Fail closed: a complete row count alone is not a successful infrastructure test."""
            infrastructure_stops = {"api_error", "worker_exception", "trajectory_timeout"}
            rows, failures = [], []
            by_agent = summary.set_index("agent")
            for agent, trajectories in all_trajectories.items():
                stop_counts = dict(pd.Series([t.stop_reason for t in trajectories]).value_counts())
                infra_count = sum(stop_counts.get(reason, 0) for reason in infrastructure_stops)
                row = by_agent.loc[agent]
                health = {
                    "agent": agent,
                    "completion_rate": float(row["completion_rate"]),
                    "infrastructure_failures": int(infra_count),
                    "truncated_steps": int(row["truncated_steps"]),
                    "judge_queue": int(row["judge_queue"]),
                    "critic_errors": int(row["critic_error_count"]),
                    "stop_reasons": json.dumps(stop_counts, sort_keys=True),
                }
                rows.append(health)
                if infra_count:
                    failures.append(f"{agent}: {infra_count} infrastructure failure(s)")
                if health["completion_rate"] < cfg.smoke_min_completion_rate:
                    failures.append(
                        f"{agent}: completion_rate={health['completion_rate']:.2f} "
                        f"< {cfg.smoke_min_completion_rate:.2f}"
                    )
                if (
                    cfg.smoke_max_truncated_steps_per_model is not None
                    and health["truncated_steps"] > cfg.smoke_max_truncated_steps_per_model
                ):
                    failures.append(
                        f"{agent}: {health['truncated_steps']} truncated step(s) > "
                        f"{cfg.smoke_max_truncated_steps_per_model}"
                    )
                if health["judge_queue"]:
                    failures.append(f"{agent}: {health['judge_queue']} unresolved judge decision(s)")
                if health["critic_errors"]:
                    failures.append(f"{agent}: {health['critic_errors']} failed critic call(s)")
            if prm_errors:
                failures.append(f"PRM errors: {prm_errors[:5]}")
            frame = pd.DataFrame(rows).sort_values("agent").reset_index(drop=True)
            display(frame)
            return frame, failures


        def generation_health_table(all_trajectories: Dict) -> pd.DataFrame:
            rows = []
            for agent, trajectories in all_trajectories.items():
                stops = dict(pd.Series([traj.stop_reason for traj in trajectories]).value_counts())
                infrastructure = sum(
                    stops.get(reason, 0)
                    for reason in ("api_error", "worker_exception", "trajectory_timeout")
                )
                rows.append({
                    "agent": agent,
                    "records": len(trajectories),
                    "explicit_answers": sum(bool(traj.finished) for traj in trajectories),
                    "infrastructure_failures": infrastructure,
                    "other_incomplete": sum(not traj.finished for traj in trajectories) - infrastructure,
                    "completion_rate": np.mean([traj.finished for traj in trajectories]) if trajectories else 0.0,
                    "stop_reasons": json.dumps(stops, sort_keys=True),
                })
            frame = pd.DataFrame(rows).sort_values("agent").reset_index(drop=True)
            display(frame)
            return frame


        def merge_trajectory_batches(batch_results: List[Dict], specs: List[ModelSpec]) -> Dict:
            merged = {spec.name: [] for spec in specs}
            seen = set()
            for result in batch_results:
                for agent, trajectories in result.items():
                    for traj in trajectories:
                        key = (agent, traj.problem_id)
                        if key in seen:
                            raise RuntimeError(f"Duplicate trajectory while merging batches: {key}")
                        seen.add(key)
                        merged[agent].append(traj)
            return merged


        def assert_no_infrastructure_failures(all_trajectories: Dict) -> None:
            failures = []
            for agent, trajectories in all_trajectories.items():
                for traj in trajectories:
                    if traj.stop_reason in {"api_error", "worker_exception", "trajectory_timeout"}:
                        failures.append((agent, traj.problem_id, traj.stop_reason))
            if failures:
                raise RuntimeError(
                    "Infrastructure failures are present; do not score or publish this run. "
                    f"First failures: {failures[:20]}"
                )


        def plot_core_results(metrics: Dict, cfg: Config, run_name: str):
            summary = summarize_metrics(metrics)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            core = summary.set_index("agent")[["C", "G", "V", "Q", "PTU"]]
            core.plot(kind="bar", ax=axes[0, 0], ylim=(0, 1), title="STRIVE core metrics")
            axes[0, 0].set_ylabel("Mean score")
            axes[0, 0].tick_params(axis="x", rotation=25)

            waste_rows = []
            for agent, values in metrics.items():
                waste_rows.append({
                    "agent": agent,
                    "useful": np.mean([x["T"]["T_useful"] for x in values]),
                    "pre-solution waste": np.mean([x["T"]["T_pre_waste"] for x in values]),
                    "post-solution waste": np.mean([x["T"]["T_post_waste"] for x in values]),
                    "answer reporting": np.mean([x["T"]["T_answer_reporting"] for x in values]),
                })
            pd.DataFrame(waste_rows).set_index("agent").plot(
                kind="bar", stacked=True, ax=axes[0, 1], title="Visible output token decomposition"
            )
            axes[0, 1].set_ylabel("Mean tokens")
            axes[0, 1].tick_params(axis="x", rotation=25)

            axes[1, 0].scatter(summary["avg_billable_tokens"], summary["V"], s=90)
            label_offsets = [(6, 6), (6, -12), (-72, 8), (-72, -12), (8, 16), (-72, 18)]
            for offset, (_, row) in zip(label_offsets, summary.iterrows()):
                axes[1, 0].annotate(
                    row["agent"],
                    (row["avg_billable_tokens"], row["V"]),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=8,
                )
            axes[1, 0].set_xlabel("Average billable tokens")
            axes[1, 0].set_ylabel("Verified solve rate V")
            axes[1, 0].set_title("Cost vs verified success")

            redundancy = pd.DataFrame([
                {
                    "agent": agent,
                    "harmful": np.mean([x["R"]["R_harmful"] for x in values]),
                    "useful verification": np.mean([x["R"]["R_useful_verification"] for x in values]),
                    "raw semantic": np.mean([x["R"]["R_semantic_raw"] for x in values]),
                }
                for agent, values in metrics.items()
            ]).set_index("agent")
            redundancy.plot(kind="bar", ax=axes[1, 1], ylim=(0, 1), title="Reasoning-aware redundancy")
            axes[1, 1].tick_params(axis="x", rotation=25)
            plt.tight_layout()
            path = Path(cfg.output_dir) / f"{run_name}_core_dashboard.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.show()

            latency = summary.set_index("agent")[[
                "avg_provider_service_sec", "avg_rate_limit_wait_sec", "avg_retry_wait_sec"
            ]]
            latency.columns = ["provider service", "RPM pacing", "retry backoff"]
            ax = latency.plot(
                kind="bar",
                stacked=True,
                figsize=(11, 5),
                title="Mean API latency decomposition per trajectory",
            )
            ax.set_ylabel("Seconds")
            ax.tick_params(axis="x", rotation=25)
            plt.tight_layout()
            latency_path = Path(cfg.output_dir) / f"{run_name}_latency_decomposition.png"
            ax.figure.savefig(latency_path, dpi=180, bbox_inches="tight")
            plt.show()
            print(f"Latency plot: {latency_path}")
            return summary, path


        def show_download_link(path: Path):
            try:
                from IPython.display import FileLink, display as ipy_display
                ipy_display(FileLink(str(path)))
            except Exception:
                print(f"Download file: {path}")


        def write_trajectory_jsonl(path: Path, all_trajectories: Dict):
            with path.open("w", encoding="utf-8") as handle:
                for trajectories in all_trajectories.values():
                    for traj in trajectories:
                        handle.write(json.dumps(traj.to_dict(), ensure_ascii=True) + "\n")


        def write_problem_jsonl(path: Path, problems: List[Dict]):
            with path.open("w", encoding="utf-8") as handle:
                for problem in problems:
                    handle.write(json.dumps(problem, ensure_ascii=True) + "\n")


        def export_generation_snapshot(
            all_trajectories: Dict,
            problems: List[Dict],
            cfg: Config,
            run_name: str,
        ) -> Tuple[Path, Path]:
            """Freeze completed generation before PRMs, judges, or metric formulas are run."""
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            root = Path(cfg.output_dir) / f"{run_name}_generation_{stamp}"
            root.mkdir(parents=True, exist_ok=False)
            write_trajectory_jsonl(root / "trajectories.jsonl", all_trajectories)
            write_problem_jsonl(root / "problems.jsonl", problems)
            files = ["trajectories.jsonl", "problems.jsonl"]
            manifest = {
                "artifact_type": "generation_snapshot",
                "raw_data_schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "run_name": run_name,
                "config": asdict(cfg),
                "models": [asdict(x) for x in MODEL_SPECS],
                "trajectory_count": sum(len(v) for v in all_trajectories.values()),
                "problem_count": len(problems),
                "data_files": files,
                "sha256": {
                    name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in files
                },
            }
            (root / "generation_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            archive = Path(shutil.make_archive(str(root), "zip", root_dir=root))
            print(f"Complete generation snapshot: {root}")
            show_download_link(archive)
            return root, archive


        def load_generation_snapshot(snapshot_dir: str) -> Dict:
            root = Path(snapshot_dir).expanduser().resolve()
            manifest = json.loads((root / "generation_manifest.json").read_text(encoding="utf-8"))
            for name, expected in manifest.get("sha256", {}).items():
                actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
                if actual != expected:
                    raise RuntimeError(f"Checksum mismatch for {name}: expected {expected}, got {actual}")
            trajectories = defaultdict(list)
            with (root / "trajectories.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        traj = Trajectory.from_dict(json.loads(line))
                        trajectories[traj.agent_name].append(traj)
            with (root / "problems.jsonl").open(encoding="utf-8") as handle:
                problems = [json.loads(line) for line in handle if line.strip()]
            return {
                "root": root,
                "manifest": manifest,
                "trajectories": dict(trajectories),
                "problems": problems,
            }


        def export_run(
            all_trajectories: Dict,
            metrics: Dict,
            cfg: Config,
            run_name: str,
            prm_scores: Optional[Dict] = None,
            problems: Optional[List[Dict]] = None,
        ):
            """Write a portable, lossless run directory for later analysis-only sessions."""
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            root = Path(cfg.output_dir) / f"{run_name}_{stamp}"
            root.mkdir(parents=True, exist_ok=False)
            trajectory_path = root / "trajectories.jsonl"
            metric_path = root / "metrics.jsonl"
            write_trajectory_jsonl(trajectory_path, all_trajectories)
            with metric_path.open("w", encoding="utf-8") as handle:
                for values in metrics.values():
                    for row in values:
                        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            if prm_scores is not None:
                (root / "prm_scores.json").write_text(
                    json.dumps(prm_scores, ensure_ascii=True), encoding="utf-8"
                )
            if problems is not None:
                write_problem_jsonl(root / "problems.jsonl", problems)
            summary = summarize_metrics(metrics)
            summary.to_csv(root / "summary.csv", index=False)
            data_files = [
                "trajectories.jsonl", "metrics.jsonl", "summary.csv",
                *( ["prm_scores.json"] if prm_scores is not None else [] ),
                *( ["problems.jsonl"] if problems is not None else [] ),
            ]
            checksums = {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in data_files
            }
            manifest = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "run_name": run_name,
                "config": asdict(cfg),
                "models": [asdict(x) for x in MODEL_SPECS],
                "core_metrics": ["C", "G", "V", "Q", "PTU", "R_harmful"],
                "latency_metrics": [
                    "provider_service", "rpm_pacing", "retry_backoff", "trajectory_total"
                ],
                "optional_metric": "E_optional = V * PTU * Q * (1 - R_harmful)",
                "ig_in_core": False,
                "raw_data_schema_version": 1,
                "data_files": data_files,
                "sha256": checksums,
                "judge": {
                    "model_id": cfg.critic_model_id,
                    "key_slot": cfg.judge_key_slot,
                    "rpm": cfg.judge_rpm,
                    "correctness_fallback": cfg.use_correctness_judge_fallback,
                    "selective_step_critic": cfg.use_critic_llm,
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"Portable raw results saved: {root}")
            archive = Path(shutil.make_archive(str(root), "zip", root_dir=root))
            show_download_link(archive)
            return root


        def load_exported_run(export_dir: str) -> Dict:
            """Load a saved run without generation APIs or PRM model weights."""
            root = Path(export_dir).expanduser().resolve()
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            for name, expected in manifest.get("sha256", {}).items():
                actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
                if actual != expected:
                    raise RuntimeError(f"Checksum mismatch for {name}: expected {expected}, got {actual}")

            trajectories = defaultdict(list)
            with (root / "trajectories.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        traj = Trajectory.from_dict(json.loads(line))
                        trajectories[traj.agent_name].append(traj)

            metrics = defaultdict(list)
            with (root / "metrics.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        metrics[row["agent"]].append(row)

            prm_path = root / "prm_scores.json"
            problem_path = root / "problems.jsonl"
            prm_scores = json.loads(prm_path.read_text(encoding="utf-8")) if prm_path.exists() else {}
            problems = []
            if problem_path.exists():
                with problem_path.open(encoding="utf-8") as handle:
                    problems = [json.loads(line) for line in handle if line.strip()]
            return {
                "root": root,
                "manifest": manifest,
                "trajectories": dict(trajectories),
                "metrics": dict(metrics),
                "prm_scores": prm_scores,
                "problems": problems,
                "summary": pd.read_csv(root / "summary.csv"),
            }
        '''
    ),
    code(
        r'''
        # ANALYSIS-ONLY SESSION EXAMPLE (run later after setting the exported directory):
        # saved = load_exported_run("/kaggle/input/your-strive-export/run_directory")
        # display(saved["summary"])
        # replot_summary, replot_path = plot_core_results(
        #     saved["metrics"], cfg, "reloaded_comparison"
        # )
        # To recalculate metric formulas without API generation or PRM loading:
        # recalculated = compute_all_metrics(
        #     saved["trajectories"], saved["prm_scores"], load_sbert(cfg), cfg
        # )
        print("Raw-results reload helper ready. Set an export directory in this cell in a later session.")
        '''
    ),
    md(
        """
        ## 10. One-cell end-to-end smoke test

        This is the required real small-scale run. It checks credentials, runs 5 tasks across all 6
        generation models, shows live code and observations, scores every trajectory with both local
        PRMs, computes the full core metric set, exports artifacts, and plots the dashboard.

        Expected size: 5 problems x 6 models = 30 trajectories. This is an infrastructure smoke test,
        not a paper result.
        """
    ),
    code(
        r'''
        # REAL END-TO-END SMOKE TEST: run this cell before the 300-problem paper experiment.
        assert cfg.use_correctness_judge_fallback, (
            "Smoke requires cfg.use_correctness_judge_fallback=True; otherwise symbolic-inconclusive "
            "answers remain unresolved without any Gemma call."
        )
        assert cfg.use_critic_llm, (
            "Smoke requires cfg.use_critic_llm=True so near-boundary step classifications are exercised."
        )
        preflight_models(MODEL_SPECS)
        if cfg.use_correctness_judge_fallback or cfg.use_critic_llm:
            preflight_models([evaluator_spec(cfg)])
        smoke_trajectories = run_experiment(
            smoke_problems,
            MODEL_SPECS,
            cfg,
            run_name="smoke_5x6_v11",
            live=True,
            resume=True,
        )
        assert set(smoke_trajectories) == {x.name for x in MODEL_SPECS}
        assert all(len(v) == len(smoke_problems) for v in smoke_trajectories.values())

        smoke_prm_scores = compute_prm_scores(smoke_trajectories, cfg, run_name="smoke_5x6_v11")
        prm_errors = [
            (agent, problem_id, errors)
            for agent, by_problem in smoke_prm_scores.items()
            for problem_id, entry in by_problem.items()
            for errors in [entry.get("errors", {})]
            if errors
        ]
        smoke_sbert = load_sbert(cfg)
        smoke_metrics = compute_all_metrics(smoke_trajectories, smoke_prm_scores, smoke_sbert, cfg)
        smoke_summary, smoke_plot = plot_core_results(smoke_metrics, cfg, "smoke_5x6_v11")
        smoke_export_dir = export_run(
            smoke_trajectories,
            smoke_metrics,
            cfg,
            "smoke_5x6_v11",
            prm_scores=smoke_prm_scores,
            problems=smoke_problems,
        )

        expected = len(smoke_problems) * len(MODEL_SPECS)
        observed = sum(len(x) for x in smoke_trajectories.values())
        assert observed == expected, (observed, expected)
        assert all(len(row["Q"]["step_labels"]) >= 1 for rows in smoke_metrics.values() for row in rows)
        display(smoke_summary)
        smoke_health, smoke_failures = validate_smoke_health(
            smoke_trajectories, smoke_summary, prm_errors, cfg
        )
        smoke_health.to_csv(Path(smoke_export_dir) / "smoke_health.csv", index=False)
        if smoke_failures:
            raise RuntimeError(
                "SMOKE TEST FAILED. Raw data was saved, but do not start the paper run.\n- "
                + "\n- ".join(smoke_failures)
            )
        print(f"Smoke test passed: {observed} trajectories; plot={smoke_plot}; export={smoke_export_dir}")
        '''
    ),
    md("## 11. Full 300-problem paper run"),
    code(
        r'''
        # PAPER GENERATION BATCH 1/3: 100 problems x 6 models = 600 trajectories.
        assert cfg.use_correctness_judge_fallback and cfg.use_critic_llm, (
            "Paper protocol requires both Gemma correctness fallback and selective step critic enabled."
        )

        def run_paper_generation_batch(batch_index: int):
            if batch_index not in range(len(paper_batches)):
                raise IndexError(batch_index)
            batch_number = batch_index + 1
            problems = paper_batches[batch_index]
            run_name = f"paper_math200_olympiad100_v11_batch{batch_number:02d}"
            print(
                f"Starting paper batch {batch_number}/{len(paper_batches)}: "
                f"{len(problems)} problems x {len(MODEL_SPECS)} models"
            )
            preflight_models(MODEL_SPECS)
            trajectories = run_experiment(
                problems, MODEL_SPECS, cfg, run_name=run_name, live=False, resume=True
            )
            health = generation_health_table(trajectories)
            export_dir, export_zip = export_generation_snapshot(
                trajectories, problems, cfg, run_name
            )
            health.to_csv(export_dir / "generation_health.csv", index=False)
            export_zip = Path(shutil.make_archive(str(export_dir), "zip", root_dir=export_dir))
            print(f"Batch {batch_number} artifacts:", export_dir, export_zip)
            expected = len(problems) * len(MODEL_SPECS)
            observed = sum(len(values) for values in trajectories.values())
            if observed != expected:
                raise RuntimeError(f"Batch {batch_number} has {observed}/{expected} records")
            # The snapshot is safely written before this assertion. Rerun this same cell to retry
            # only retryable infrastructure-failed checkpoints after provider cooldown.
            assert_no_infrastructure_failures(trajectories)
            return trajectories


        paper_batch_1_trajectories = run_paper_generation_batch(0)
        '''
    ),
    code(
        r'''
        # PAPER GENERATION BATCH 2/3: independently checkpointed and resumable.
        paper_batch_2_trajectories = run_paper_generation_batch(1)
        '''
    ),
    code(
        r'''
        # PAPER GENERATION BATCH 3/3, then merge all 1,800 successful records.
        paper_batch_3_trajectories = run_paper_generation_batch(2)
        paper_trajectories = merge_trajectory_batches(
            [paper_batch_1_trajectories, paper_batch_2_trajectories, paper_batch_3_trajectories],
            MODEL_SPECS,
        )
        assert all(len(values) == len(experiment_problems) for values in paper_trajectories.values())
        full_generation_health = generation_health_table(paper_trajectories)
        assert_no_infrastructure_failures(paper_trajectories)
        paper_generation_export_dir, paper_generation_zip = export_generation_snapshot(
            paper_trajectories,
            experiment_problems,
            cfg,
            "paper_math200_olympiad100_v11_complete",
        )
        full_generation_health.to_csv(
            paper_generation_export_dir / "generation_health.csv", index=False
        )
        paper_generation_zip = Path(shutil.make_archive(
            str(paper_generation_export_dir), "zip", root_dir=paper_generation_export_dir
        ))
        print("Complete 1,800-trajectory generation artifacts:", paper_generation_zip)
        '''
    ),
    code(
        r'''
        # Offline metric phase. The two 7B PRMs load one at a time and cache after every trajectory.
        # To resume metrics from an uploaded generation snapshot in a prepared metric session:
        # generation_only = load_generation_snapshot("/path/to/unzipped_generation_snapshot")
        # paper_trajectories = generation_only["trajectories"]
        # experiment_problems = generation_only["problems"]
        preflight_models([evaluator_spec(cfg)])
        paper_prm_scores = compute_prm_scores(
            paper_trajectories,
            cfg,
            run_name="paper_math200_olympiad100_v11",
        )
        paper_sbert = load_sbert(cfg)
        paper_metrics = compute_all_metrics(paper_trajectories, paper_prm_scores, paper_sbert, cfg)
        paper_summary, paper_plot = plot_core_results(
            paper_metrics,
            cfg,
            "paper_math200_olympiad100_v11",
        )
        paper_export_dir = export_run(
            paper_trajectories,
            paper_metrics,
            cfg,
            "paper_math200_olympiad100_v11",
            prm_scores=paper_prm_scores,
            problems=experiment_problems,
        )
        display(paper_summary)
        print("Paper artifacts:", paper_export_dir)
        evaluator_failures = paper_summary[
            (paper_summary["judge_queue"] > 0) | (paper_summary["critic_error_count"] > 0)
        ]
        if not evaluator_failures.empty:
            raise RuntimeError(
                "Evaluator failures remain. Raw paper data was saved, but do not publish these metrics.\n"
                + evaluator_failures[
                    ["agent", "judge_queue", "critic_call_count", "critic_error_count"]
                ].to_string(index=False)
            )
        '''
    ),
    md(
        """
        ## 12. Paper-run protocol notes

        - Report the exact model IDs and run date because hosted aliases can change.
        - Five NVIDIA lanes and one OpenAI lane run concurrently. Each lane advances to its next problem
          immediately after checkpointing, without a cross-model problem barrier. Preflight requires five
          distinct authorized NVIDIA key values; an explicitly selected lane may use standby slot _6.
        - A model trajectory itself cannot be parallelized across steps: each next action depends on the
          real observation from the preceding code execution.
        - The five-task smoke set is asserted disjoint from the frozen 300-problem paper set and proves
          plumbing only. With the default sample counts, use all 1,800 paper trajectories for comparison.
        - Paper generation is split into three stratified 100-problem batches. Each batch has separate
          checkpoints and exports; rerunning a failed batch retries only retryable infrastructure errors.
        - MiniMax M3 replaces the unstable DeepSeek Flash lane. Nemotron uses non-thinking mode and
          GPT-OSS sends NVIDIA's explicit low-reasoning API parameter. Visible actions and provider
          reasoning remain separate audit channels.
        - All models share the same action and token budgets. Timeouts, retries, truncations, provider
          service time, RPM pacing, and retry backoff are exported and must be reported.
        - Per-model request pacing follows observed endpoint behavior. 429 responses receive a long
          cooldown and repeated lane failures trigger a breaker pause instead of rapid failed records.
        - Placeholder text and provider-only reasoning do not count as final answers. The final answer
          must appear in visible assistant content; raw and provider-reasoning channels remain stored.
        - Symbolically inconclusive answers are exported with `needs_judge=True`; adjudicate those with a
          frozen external judge protocol and audit a sample before the final paper tables.
        - Gemma 4 31B is used only for deterministic-inconclusive correctness and near-boundary step
          labels. Grounding remains deterministic. `cfg.judge_key_slot=6` selects the dedicated NVIDIA
          judge key by default, with a 150-second evaluator preflight deadline; HTTP 429 uses backoff,
          not automatic account rotation.
        - Code tokens are a subtype of model output, not an additive cost. `T_billable` is API input plus
          API output. Tool-output tokens are reported separately.
        - Each final export includes raw trajectories, metrics, both PRM outputs, selected problems,
          manifest checksums, and a loader for analysis-only notebook sessions.
        - G is based only on successful sandbox output. A correct intermediate output does not ground a
          different final answer.
        """
    ),
])


cells.extend([
    md("## 7. Correctness, provenance, and PRM scoring"),
    code(
        r'''
        try:
            from math_verify import parse as math_parse, verify as math_verify
            HAS_MATH_VERIFY = True
        except Exception:
            HAS_MATH_VERIFY = False


        def strip_boxed(text: str) -> str:
            text = str(text or "").strip()
            match = re.search(r"\\boxed\{([^{}]+)\}", text)
            return match.group(1).strip() if match else text


        def normalize_answer(text: str) -> str:
            text = strip_boxed(text).lower().strip()
            text = text.replace("$", "").replace("\\,", "").replace(" ", "")
            text = text.replace("\\left", "").replace("\\right", "")
            text = re.sub(r"^(finalanswer|answer|result)[:=]", "", text)
            return text.rstrip(".")


        def check_symbolic_equivalence(prediction: str, gold: str) -> Optional[bool]:
            """True/False when deterministically decidable; None routes to adjudication."""
            pred, ref = str(prediction or "").strip(), str(gold or "").strip()
            if not pred:
                return False
            if normalize_answer(pred) == normalize_answer(ref):
                return True
            if HAS_MATH_VERIFY:
                try:
                    pred_parsed, ref_parsed = math_parse(pred), math_parse(ref)
                    if pred_parsed and ref_parsed:
                        return bool(math_verify(ref_parsed, pred_parsed))
                except Exception:
                    pass
            try:
                import sympy
                from sympy.parsing.sympy_parser import parse_expr
                p = parse_expr(normalize_answer(pred).replace("^", "**"), evaluate=True)
                g = parse_expr(normalize_answer(ref).replace("^", "**"), evaluate=True)
                return bool(sympy.simplify(p - g) == 0)
            except Exception:
                return None


        def extract_output_candidates(stdout: str) -> List[str]:
            text = str(stdout or "")
            candidates = re.findall(r"\\boxed\{([^{}]+)\}", text)
            candidates += re.findall(
                r"(?:final answer|answer|result|ans)\s*[:=]\s*([^\n]+)",
                text,
                flags=re.IGNORECASE,
            )
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            candidates += lines[-4:]
            candidates += re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:/\d+)?", text)
            return list(dict.fromkeys(x.strip().rstrip(".") for x in candidates if x.strip()))


        def trace_final_answer(traj: Trajectory) -> Dict:
            """Ground only against successful real tool outputs, never the model's final answer text."""
            evidence_count = 0
            for step in traj.steps:
                if step.get("action_type") != "code" or not step.get("code_success"):
                    continue
                evidence_count += 1
                for candidate in extract_output_candidates(step.get("stdout", "")):
                    if check_symbolic_equivalence(traj.final_answer, candidate) is True:
                        return {
                            "grounded": True,
                            "evidence_step": step["step_num"],
                            "candidate": candidate,
                            "evidence_hash": step.get("evidence_hash", ""),
                            "method": "symbolic_tool_output_match",
                            "evidence_count": evidence_count,
                        }
            return {
                "grounded": False,
                "evidence_step": None,
                "candidate": None,
                "evidence_hash": None,
                "method": "no_tool_evidence" if evidence_count == 0 else "tool_used_not_traceable",
                "evidence_count": evidence_count,
            }


        def extract_json_object(text: str) -> Optional[Dict]:
            match = re.search(r"\{.*\}", str(text or ""), flags=re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except Exception:
                return None


        def evaluator_spec(cfg: Config) -> ModelSpec:
            key_env = (
                "NVIDIA_CRITIC_API_KEY"
                if cfg.judge_key_slot == 0
                else f"NVIDIA_API_KEY_{cfg.judge_key_slot}"
            )
            allowed_judge_keys = {
                "NVIDIA_CRITIC_API_KEY",
                "NVIDIA_API_KEY_1",
                "NVIDIA_API_KEY_2",
                "NVIDIA_API_KEY_3",
                "NVIDIA_API_KEY_4",
                "NVIDIA_API_KEY_5",
                "NVIDIA_API_KEY_6",
            }
            if key_env not in allowed_judge_keys:
                raise RuntimeError(
                    "The judge must use an NVIDIA credential. OPENAI_API_KEY and other key sources "
                    f"are forbidden for evaluator calls; received {key_env!r}."
                )
            return ModelSpec(
                name="external-nim-evaluator",
                provider="nvidia",
                model_id=cfg.critic_model_id,
                api_key_env=key_env,
                rpm=cfg.judge_rpm,
                base_url="https://integrate.api.nvidia.com/v1",
                request_params={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 256,
                    # Short classification JSON; suppress Gemma's thought channel.
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                },
            )


        _evaluator_contract = evaluator_spec(cfg)
        assert _evaluator_contract.provider == "nvidia"
        assert _evaluator_contract.api_key_env != "OPENAI_API_KEY"
        assert _evaluator_contract.base_url == "https://integrate.api.nvidia.com/v1"


        def judge_correctness_fallback(traj: Trajectory, cfg: Config) -> Dict:
            prompt = f"""Determine whether the predicted final answer is mathematically equivalent to the reference.
        Use the problem to interpret notation. Ignore reasoning style and provenance.

        Problem:
        {traj.problem_text}

        Reference answer:
        {traj.gold_answer}

        Predicted answer:
        {traj.final_answer}

        Return JSON only: {{"correct": true_or_false, "confidence": number_0_to_1, "reason": "brief"}}
        """
            try:
                raw, _ = call_model(
                    evaluator_spec(cfg),
                    [{
                        "role": "user",
                        "content": (
                            "Act as a neutral mathematical answer adjudicator. "
                            "Return only the requested JSON object.\n\n" + prompt
                        ),
                    }],
                    cfg,
                )
                parsed = extract_json_object(raw)
                if not isinstance(parsed, dict) or not isinstance(parsed.get("correct"), bool):
                    return {"decision": None, "raw": raw, "error": "invalid_json_schema"}
                return {
                    "decision": int(parsed["correct"]),
                    "confidence": float(parsed.get("confidence", 0.0)),
                    "reason": str(parsed.get("reason", "")),
                    "raw": raw,
                    "error": "",
                }
            except Exception as exc:
                return {"decision": None, "raw": "", "error": str(exc)}


        def compute_correctness_grounding(traj: Trajectory, cfg: Config) -> Dict:
            c_sym = check_symbolic_equivalence(traj.final_answer, traj.gold_answer)
            judge = None
            if c_sym is None and cfg.use_correctness_judge_fallback:
                judge = judge_correctness_fallback(traj, cfg)
            if c_sym is not None:
                c_final = int(c_sym)
                correctness_source = "symbolic"
            elif judge and judge.get("decision") is not None:
                c_final = int(judge["decision"])
                correctness_source = "judge_fallback"
            else:
                c_final = 0
                correctness_source = "unresolved_needs_adjudication"
            provenance = trace_final_answer(traj)
            if provenance["evidence_count"] == 0:
                g_level = "G0"
            elif not provenance["grounded"]:
                g_level = "G1"
            elif c_final:
                g_level = "G3"
            else:
                g_level = "G2"
            grounded = int(g_level in {"G2", "G3"})
            return {
                "C_sym": None if c_sym is None else int(c_sym),
                "C_judge": None if not judge else judge.get("decision"),
                "C_final": c_final,
                "correctness_source": correctness_source,
                "judge_used": judge is not None,
                "judge_record": judge,
                "needs_judge": c_sym is None and (not judge or judge.get("decision") is None),
                "G_level": g_level,
                "G": grounded,
                "V": c_final * grounded,
                "grounding_source": provenance["method"],
                "grounding_evidence_step": provenance["evidence_step"],
                "grounding_candidate": provenance["candidate"],
                "grounding_evidence_hash": provenance["evidence_hash"],
                "tool_evidence_count": provenance["evidence_count"],
            }


        def format_step_for_prm(step: Dict) -> str:
            parts = [step.get("reasoning", "").strip()]
            if step.get("code"):
                parts.append("Code:\n" + step["code"])
            if step.get("action_type") == "code":
                parts.append("Observation:\n" + step.get("observation", ""))
            if step.get("action_type") == "answer":
                parts.append("Final answer: " + step.get("final_answer", ""))
            return "\n".join(x for x in parts if x).strip()


        def quantization_kwargs(cfg: Config) -> Dict:
            if not cfg.prm_load_in_4bit or not torch.cuda.is_available():
                return {"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
            from transformers import BitsAndBytesConfig
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            }


        def load_prm(spec: Dict, cfg: Config):
            from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

            repo_kwargs = {"trust_remote_code": True}
            if spec.get("revision"):
                repo_kwargs["revision"] = spec["revision"]
            tokenizer = AutoTokenizer.from_pretrained(spec["model_name"], **repo_kwargs)
            kwargs = {"device_map": "auto", **repo_kwargs} | quantization_kwargs(cfg)
            if spec["type"] == "math_shepherd":
                model = AutoModelForCausalLM.from_pretrained(spec["model_name"], **kwargs).eval()
            elif spec["type"] == "qwen_prm":
                # The repository's config.json omits pad_token_id even though its
                # generation_config.json defines it. Repair it before construction.
                config = AutoConfig.from_pretrained(spec["model_name"], **repo_kwargs)
                pad_token_id = tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = getattr(config, "bos_token_id", None) or 151643
                    tokenizer.pad_token_id = int(pad_token_id)
                config.pad_token_id = int(pad_token_id)
                config.use_cache = False
                kwargs["config"] = config
                model = AutoModel.from_pretrained(spec["model_name"], **kwargs).eval()
            else:
                raise ValueError(spec["type"])
            return model, tokenizer


        def model_device(model):
            return next(model.parameters()).device


        def score_math_shepherd(traj: Trajectory, model, tokenizer, cfg: Config) -> List[float]:
            step_tag = "ки"
            candidate_tokens = tokenizer.encode("+ -")[1:]
            if len(candidate_tokens) != 2:
                raise RuntimeError(f"Unexpected Math-Shepherd candidate tokenization: {candidate_tokens}")
            tag_id = tokenizer.encode(step_tag)[-1]
            body = "\n".join(
                f"Step {i + 1}: {format_step_for_prm(step)} {step_tag}"
                for i, step in enumerate(traj.steps)
            )
            inputs = tokenizer(
                traj.problem_text + "\n" + body,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.prm_max_length,
            ).to(model_device(model))
            with torch.inference_mode():
                logits = model(**inputs).logits[0, :, candidate_tokens]
            positive = torch.softmax(logits.float(), dim=-1)[:, 0]
            values = positive[inputs["input_ids"][0] == tag_id].detach().cpu().tolist()
            return [float(x) for x in values[-len(traj.steps):]]


        def score_qwen_prm(traj: Trajectory, model, tokenizer, cfg: Config) -> List[float]:
            import torch.nn.functional as F
            response_steps = [format_step_for_prm(step) for step in traj.steps]
            messages = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": traj.problem_text},
                {"role": "assistant", "content": "<extra_0>".join(response_steps) + "<extra_0>"},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.prm_max_length,
            ).to(model_device(model))
            with torch.inference_mode():
                output = model(**inputs)
                logits = output.logits if hasattr(output, "logits") else output[0]
            sep_id = tokenizer.convert_tokens_to_ids("<extra_0>")
            if sep_id is None or sep_id == tokenizer.unk_token_id:
                encoded_sep = tokenizer.encode("<extra_0>", add_special_tokens=False)
                if len(encoded_sep) != 1:
                    raise RuntimeError(f"<extra_0> is not a single special token: {encoded_sep}")
                sep_id = encoded_sep[0]
            mask = inputs["input_ids"] == sep_id
            if int(mask.sum()) == 0:
                raise RuntimeError("No <extra_0> separator tokens remained after tokenization/truncation")
            probabilities = F.softmax(logits.float(), dim=-1) * mask.unsqueeze(-1)
            sample = probabilities[0]
            rewards = sample[sample != 0].view(-1, 2)[:, 1].detach().cpu().tolist()
            return [float(x) for x in rewards[-len(traj.steps):]]


        def compute_prm_scores(all_trajectories: Dict[str, List[Trajectory]], cfg: Config, run_name: str) -> Dict:
            """One forward pass per trajectory per PRM; models load sequentially and results are cached."""
            if cfg.prm_load_in_4bit and not torch.cuda.is_available():
                raise RuntimeError(
                    "Four-bit PRM scoring requires a CUDA GPU. Enable a Kaggle GPU accelerator, "
                    "restart the kernel, and rerun Sections 1-10. Generation checkpoints remain safe."
                )
            cache = Path(cfg.output_dir) / "checkpoints" / run_name / "prm_scores.json"
            cache.parent.mkdir(parents=True, exist_ok=True)

            def write_cache_atomic(value: Dict):
                temporary = cache.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(value), encoding="utf-8")
                os.replace(temporary, cache)

            if cache.exists():
                try:
                    scores = json.loads(cache.read_text())
                except Exception as exc:
                    raise RuntimeError(
                        f"PRM cache is unreadable: {cache}. Preserve it for diagnosis, then rerun "
                        f"with a new run name. Original error: {exc}"
                    ) from exc
            else:
                scores = {}
            for spec in cfg.prm_models:
                key = spec["key"]
                pending_count = sum(
                    (
                        key not in scores.get(agent, {}).get(traj.problem_id, {}).get("by_model", {})
                        or key in scores.get(agent, {}).get(traj.problem_id, {}).get("errors", {})
                    )
                    for agent, trajectories in all_trajectories.items()
                    for traj in trajectories
                )
                if pending_count == 0:
                    print(f"\nPRM phase: {key} already complete in cache; skipping model load")
                    continue
                model = tokenizer = None
                try:
                    print(
                        f"\nPRM phase: loading only {key}: {spec['model_name']} "
                        f"({pending_count} trajectories pending)"
                    )
                    model, tokenizer = load_prm(spec, cfg)
                    for agent, trajectories in all_trajectories.items():
                        scores.setdefault(agent, {})
                        for traj in tqdm(trajectories, desc=f"{key} | {agent}"):
                            entry = scores[agent].setdefault(traj.problem_id, {"by_model": {}})
                            if key in entry["by_model"] and key not in entry.get("errors", {}):
                                continue
                            entry.setdefault("errors", {})
                            try:
                                if spec["type"] == "math_shepherd":
                                    values = score_math_shepherd(traj, model, tokenizer, cfg)
                                else:
                                    values = score_qwen_prm(traj, model, tokenizer, cfg)
                                entry["errors"].pop(key, None)
                            except Exception as exc:
                                entry["errors"][key] = repr(exc)
                                values = [0.5] * len(traj.steps)
                            if len(values) < len(traj.steps):
                                values = values + [values[-1] if values else 0.5] * (len(traj.steps) - len(values))
                            entry["by_model"][key] = values[:len(traj.steps)]
                            available = list(entry["by_model"].values())
                            entry["ensemble"] = [
                                float(np.mean([v[i] for v in available if i < len(v)]))
                                for i in range(len(traj.steps))
                            ]
                            write_cache_atomic(scores)
                finally:
                    print(f"PRM phase: offloading {key} before the next PRM is loaded")
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    if model is not None:
                        del model
                    if tokenizer is not None:
                        del tokenizer
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        try:
                            torch.cuda.ipc_collect()
                        except Exception:
                            pass
                        print(
                            "CUDA after offload: "
                            f"allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GB, "
                            f"reserved={torch.cuda.memory_reserved() / 1024**3:.2f} GB"
                        )
            return scores
        '''
    ),
])

cells.extend([
    md("## 6. ReAct-style controller and trajectory collection"),
    code(
        r'''
        REACT_SYSTEM_PROMPT = """You are a mathematical problem-solving agent with one Python tool.

        At each turn, return exactly one action and no surrounding discussion:
        - A line beginning with `Reasoning step:` followed by one fenced `python` block containing
          executable Python that prints the required result; or
        - A line beginning with `Reasoning step:` followed by one fenced `answer` block containing
          only a concrete final mathematical answer.

        Rules:
        - Never invent, predict, or write an Observation. The controller supplies real execution output.
        - Never copy format descriptions such as `Python code`, `final answer only`, or template text.
        - Use only math libraries available in the sandbox: math, fractions, decimal, itertools,
          collections, sympy, numpy, and scipy.
        - Each code action must make new progress or perform a useful verification.
        - Do not repeat a failed action unchanged.
        - Use no more than four Python actions. Once enough evidence exists, answer immediately.
        - The answer block must contain the answer to the original problem, not an intermediate value.
        - Diagrams are not necessarily to scale. Asymptote/TikZ coordinates are rendering instructions,
          not mathematical givens; use only stated labels, relations, and problem conditions.
        """


        @dataclass
        class StepRecord:
            step_num: int
            model_input_messages: List[Dict] = field(default_factory=list)
            model_input_hash: str = ""
            reasoning: str = ""
            provider_reasoning: str = ""
            raw_response: str = ""
            action_type: str = "none"
            action_source: str = ""
            parse_error: str = ""
            code: str = ""
            final_answer: str = ""
            stdout: str = ""
            stderr: str = ""
            observation: str = ""
            code_success: bool = False
            code_blocked: bool = False
            tool_elapsed_sec: float = 0.0
            request_latency_sec: float = 0.0
            service_latency_sec: float = 0.0
            service_time_all_attempts_sec: float = 0.0
            rate_limit_wait_sec: float = 0.0
            retry_wait_sec: float = 0.0
            request_attempts: int = 0
            retry_errors: List[Dict] = field(default_factory=list)
            finish_reason: str = ""
            provider_truncated: bool = False
            visible_content_chars: int = 0
            provider_reasoning_chars: int = 0
            input_tokens: int = 0
            output_tokens: int = 0
            hidden_reasoning_tokens: int = 0
            evidence_hash: str = ""
            error: str = ""


        @dataclass
        class Trajectory:
            problem_id: str
            dataset: str
            subject: str
            level: str
            problem_text: str
            gold_answer: str
            agent_name: str
            provider: str
            model_id: str
            steps: List[Dict] = field(default_factory=list)
            final_answer: str = ""
            finished: bool = False
            stop_reason: str = ""
            started_at: str = ""
            elapsed_sec: float = 0.0
            timed_out: bool = False

            @property
            def total_steps(self):
                return len(self.steps)

            def to_dict(self):
                return asdict(self)

            @classmethod
            def from_dict(cls, value: Dict):
                allowed = set(cls.__dataclass_fields__)
                return cls(**{k: v for k, v in value.items() if k in allowed})


        CODE_RE = re.compile(r"```python\s*(.*?)```", re.IGNORECASE | re.DOTALL)
        ANSWER_RE = re.compile(r"```answer\s*(.*?)```", re.IGNORECASE | re.DOTALL)
        OPEN_CODE_RE = re.compile(r"```python\s*(.*)$", re.IGNORECASE | re.DOTALL)
        REASON_RE = re.compile(r"Reasoning\s*step\s*:\s*(.*?)(?=```|$)", re.IGNORECASE | re.DOTALL)
        OBS_RE = re.compile(r"(?:^|\n)\s*Observation\s*:", re.IGNORECASE)
        PLACEHOLDER_MARKERS = (
            "<python code", "<final answer", "<one concise sentence",
            "python code; always print", "final answer only", "your final mathematical value",
            "template text", "final_value",
        )


        def sanitize_action_content(action_type: str, content: str) -> str:
            content = str(content or "").strip().strip("`").strip()
            content = re.sub(r"^(?:\\n)+|(?:\\n)+$", "", content).strip()
            if action_type == "answer":
                content = re.sub(
                    r"^\s*(?:final\s+answer|answer)\s*(?:is|:|=)?\s*(?:\r?\n)+",
                    "",
                    content,
                    flags=re.IGNORECASE,
                ).strip()
                content = re.sub(
                    r"^\s*(?:final\s+answer|answer)\s*(?:is|:|=)\s*",
                    "",
                    content,
                    flags=re.IGNORECASE,
                ).strip()
            return content


        def placeholder_reason(action_type: str, content: str) -> str:
            normalized = re.sub(r"\s+", " ", str(content or "").replace("\\n", " ")).strip().lower()
            if not normalized:
                return "empty_action"
            if any(marker in normalized for marker in PLACEHOLDER_MARKERS):
                return "template_placeholder"
            if action_type == "code" and normalized in {"...", "pass", "print(...)"}:
                return "non_executable_template"
            if action_type == "code" and re.match(
                r"^(?:reasoning\s+step|answer|final\s+answer)\s*:?(?:\s|$)",
                normalized,
            ):
                return "answer_text_in_code_block"
            return ""


        def concise_answer_fallback(text: str) -> str:
            text = str(text or "").strip()
            bold_labelled = list(re.finditer(
                r"(?:^|\n)\s*\*{1,2}(?:final\s+)?answer\s*:\*{1,2}"
                r"\s*(?:\r?\n)+\s*([^\n]{1,100})",
                text,
                flags=re.IGNORECASE,
            ))
            if bold_labelled:
                return sanitize_action_content("answer", bold_labelled[-1].group(1))
            labelled_matches = list(re.finditer(
                r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*([^\n]{1,100})",
                text,
                flags=re.IGNORECASE,
            ))
            if labelled_matches:
                return sanitize_action_content("answer", labelled_matches[-1].group(1))
            boxed = re.search(r"\\boxed\{([^{}]+)\}", text)
            if boxed:
                return sanitize_action_content("answer", boxed.group(1))
            markdown_answers = list(re.finditer(
                r"(?:^|\n)\s*\*{0,2}(?:final\s+)?answer\*{0,2}\s*:?[ \t]*\n+(.+)$",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            ))
            if markdown_answers:
                candidate = markdown_answers[-1].group(1).strip()
                candidate = re.sub(r"^(?:\\\[|\$\$?)\s*", "", candidate)
                candidate = re.sub(r"\s*(?:\\\]|\$\$?)$", "", candidate).strip()
                if (
                    0 < len(candidate) <= 200
                    and len(candidate.splitlines()) <= 4
                    and not any(token in candidate.lower() for token in ("```", "import ", "print("))
                ):
                    return sanitize_action_content("answer", candidate)
            generic_fences = list(re.finditer(
                r"```(?:text|latex|math)?\s*\n?\s*([^`]{1,100}?)\s*```",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            ))
            if generic_fences:
                candidate = generic_fences[-1].group(1).strip()
                if (
                    len(candidate.splitlines()) <= 2
                    and not any(token in candidate.lower() for token in ("import ", "print(", "def ", "="))
                ):
                    return sanitize_action_content("answer", candidate)
            displays = list(re.finditer(r"\\\[(.{1,300}?)\\\]", text, flags=re.DOTALL))
            if displays:
                candidate = re.sub(r"\s+", " ", displays[-1].group(1)).strip()
                if "=" in candidate:
                    candidate = candidate.rsplit("=", 1)[-1].strip()
                if 0 < len(candidate) <= 100 and not any(
                    token in candidate.lower() for token in ("import ", "print(", "def ")
                ):
                    return sanitize_action_content("answer", candidate)
            if len(text) <= 80 and len(text.splitlines()) <= 2 and re.search(r"\d", text):
                if not any(token in text.lower() for token in ("reasoning", "python", "import ", "print(")):
                    return sanitize_action_content("answer", text)
            return ""


        def parse_action(raw: str, provider_reasoning: str = "") -> Dict:
            """Parse only visible assistant content; provider reasoning remains audit metadata."""
            raw = str(raw or "")
            marker = OBS_RE.search(raw)
            clean = raw[:marker.start()] if marker else raw
            reasoning_match = REASON_RE.search(clean)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            code_match = CODE_RE.search(clean)
            answer_match = ANSWER_RE.search(clean)
            candidates = []
            if code_match:
                candidates.append((code_match.start(), "code", code_match.group(1).strip()))
            if answer_match:
                candidates.append((answer_match.start(), "answer", answer_match.group(1).strip()))
            if not candidates:
                open_code = OPEN_CODE_RE.search(clean)
                if open_code and open_code.group(1).strip():
                    candidates.append((open_code.start(), "code", open_code.group(1).strip().rstrip("`")))
            if len({kind for _, kind, _ in candidates}) > 1:
                return {
                    "reasoning": reasoning,
                    "action_type": "none",
                    "content": "",
                    "action_source": "visible_content",
                    "parse_error": "multiple_actions",
                }
            if not candidates:
                fallback_answer = concise_answer_fallback(clean)
                placeholder = placeholder_reason("answer", fallback_answer) if fallback_answer else ""
                if fallback_answer and not placeholder:
                    return {
                        "reasoning": reasoning,
                        "action_type": "answer",
                        "content": fallback_answer,
                        "action_source": "visible_fallback",
                        "parse_error": "",
                    }
                return {
                    "reasoning": reasoning,
                    "action_type": "none",
                    "content": "",
                    "action_source": "",
                    "parse_error": placeholder or "no_visible_action",
                }
            _, action_type, content = sorted(candidates, key=lambda x: x[0])[0]
            content = sanitize_action_content(action_type, content)
            placeholder = placeholder_reason(action_type, content)
            if placeholder:
                return {
                    "reasoning": reasoning,
                    "action_type": "none",
                    "content": "",
                    "action_source": "visible_content",
                    "parse_error": placeholder,
                }
            return {
                "reasoning": reasoning,
                "action_type": action_type,
                "content": content,
                "action_source": "visible_content",
                "parse_error": "",
            }


        def assistant_turn(step: Dict) -> str:
            if step["action_type"] == "code":
                return f"Reasoning step: {step['reasoning']}\n```python\n{step['code']}\n```"
            if step["action_type"] == "answer":
                return f"Reasoning step: {step['reasoning']}\n```answer\n{step['final_answer']}\n```"
            return step.get("raw_response", "")


        def build_messages(
            problem: str,
            steps: List[Dict],
            spec: ModelSpec,
            cfg: Config,
            force_final_answer: bool = False,
            prefer_final_answer: bool = False,
        ) -> List[Dict]:
            system_content = REACT_SYSTEM_PROMPT
            if spec.system_directive:
                system_content = spec.system_directive.strip() + "\n\n" + system_content
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Solve this problem:\n\n{problem}"},
            ]
            for step in steps:
                messages.append({"role": "assistant", "content": assistant_turn(step)})
                if step["action_type"] == "code":
                    messages.append({"role": "user", "content": "Observation:\n" + step["observation"]})
                elif step["action_type"] == "none":
                    feedback = step.get("observation") or step.get("error") or "No valid action was found."
                    messages.append({
                        "role": "user",
                        "content": (
                            "Controller feedback: " + feedback + "\n"
                            "Return one concrete fenced Python action or one concrete fenced answer. "
                            "Do not repeat template wording or the previous action."
                        ),
                    })
            if force_final_answer:
                messages.append({
                    "role": "user",
                    "content": (
                        "Tool-action budget reached. Do not emit more Python. Using the problem and "
                        "real observations already present, emit exactly one concise fenced ```answer``` "
                        "block containing the final answer."
                    ),
                })
            elif prefer_final_answer:
                messages.append({
                    "role": "user",
                    "content": (
                        "A real tool result is now available. If it resolves the original problem, "
                        "return the concrete final answer now; otherwise make exactly one new useful action."
                    ),
                })
            # Bound context while keeping the system prompt and problem. Full trace remains on disk.
            while sum(len(x["content"]) for x in messages) > cfg.max_history_chars and len(messages) > 4:
                del messages[2:4]
            return messages


        _print_lock = threading.Lock()


        def live_event(problem_id: str, model: str, step: int, title: str, body: str = ""):
            if not cfg.live_trace:
                return
            with _print_lock:
                tqdm.write(f"[{problem_id}] [{model}] step {step} | {title}")
                if body:
                    tqdm.write(truncate_middle(body, 1800))


        def run_react_trajectory(problem: Dict, spec: ModelSpec, cfg: Config) -> Trajectory:
            start = time.perf_counter()
            trajectory_deadline = time.monotonic() + cfg.trajectory_timeout_sec
            traj = Trajectory(
                problem_id=problem["id"],
                dataset=problem["dataset"],
                subject=problem["subject"],
                level=problem["level"],
                problem_text=problem["problem"],
                gold_answer=problem["gold_answer"],
                agent_name=spec.name,
                provider=spec.provider,
                model_id=spec.model_id,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            try:
                with MultiStepPythonSandbox(cfg) as sandbox:
                    executed_tool_actions = 0
                    consecutive_format_errors = 0
                    consecutive_code_failures = 0
                    repeated_actions = 0
                    answer_only_violations = 0
                    previous_action_signature = ""
                    force_final_next = False
                    prefer_final_next = False
                    for step_num in range(1, cfg.max_steps + 1):
                        if time.monotonic() >= trajectory_deadline:
                            traj.stop_reason = "trajectory_timeout"
                            traj.timed_out = True
                            live_event(problem["id"], spec.name, step_num, "TRAJECTORY TIMEOUT")
                            break
                        force_final_answer = (
                            force_final_next
                            or
                            executed_tool_actions >= cfg.max_tool_actions
                            or step_num == cfg.max_steps
                        )
                        messages = build_messages(
                            problem["problem"],
                            traj.steps,
                            spec,
                            cfg,
                            force_final_answer=force_final_answer,
                            prefer_final_answer=prefer_final_next and not force_final_answer,
                        )
                        live_event(
                            problem["id"],
                            spec.name,
                            step_num,
                            "API REQUEST",
                            f"deadline={cfg.request_total_timeout_sec}s; trajectory remaining="
                            f"{max(0.0, trajectory_deadline - time.monotonic()):.1f}s",
                        )
                        try:
                            raw, usage = call_model(
                                spec,
                                messages,
                                cfg,
                                deadline_monotonic=trajectory_deadline,
                                context=f"{problem['id']} | {spec.name} | step {step_num}",
                            )
                        except Exception as exc:
                            if time.monotonic() >= trajectory_deadline:
                                traj.stop_reason = "trajectory_timeout"
                                traj.timed_out = True
                            else:
                                traj.stop_reason = "api_error"
                            failure = getattr(exc, "meta", {})
                            traj.steps.append(asdict(StepRecord(
                                step_num=step_num,
                                error=str(exc),
                                request_latency_sec=float(failure.get("latency_sec", 0.0)),
                                service_latency_sec=float(failure.get("service_latency_sec", 0.0)),
                                service_time_all_attempts_sec=float(
                                    failure.get("service_time_all_attempts_sec", 0.0)
                                ),
                                rate_limit_wait_sec=float(failure.get("rate_limit_wait_sec", 0.0)),
                                retry_wait_sec=float(failure.get("retry_wait_sec", 0.0)),
                                request_attempts=int(failure.get("request_attempts", 0)),
                                retry_errors=list(failure.get("retry_errors", [])),
                                finish_reason=str(failure.get("finish_reason", "error")),
                            )))
                            live_event(problem["id"], spec.name, step_num, "API ERROR", str(exc))
                            break

                        action = parse_action(raw, usage.get("provider_reasoning", ""))
                        serialized_input = json.dumps(messages, ensure_ascii=True, sort_keys=True)
                        record = StepRecord(
                            step_num=step_num,
                            model_input_messages=messages if cfg.store_full_model_inputs else [],
                            model_input_hash=hashlib.sha256(serialized_input.encode()).hexdigest()[:16],
                            reasoning=action["reasoning"],
                            provider_reasoning=usage.get("provider_reasoning", ""),
                            raw_response=raw,
                            action_type=action["action_type"],
                            action_source=action.get("action_source", ""),
                            parse_error=action.get("parse_error", ""),
                            request_latency_sec=float(usage.get("latency_sec", 0.0)),
                            service_latency_sec=float(usage.get("service_latency_sec", 0.0)),
                            service_time_all_attempts_sec=float(
                                usage.get("service_time_all_attempts_sec", 0.0)
                            ),
                            rate_limit_wait_sec=float(usage.get("rate_limit_wait_sec", 0.0)),
                            retry_wait_sec=float(usage.get("retry_wait_sec", 0.0)),
                            request_attempts=int(usage.get("request_attempts", 1)),
                            retry_errors=list(usage.get("retry_errors", [])),
                            finish_reason=str(usage.get("finish_reason", "")),
                            provider_truncated=bool(usage.get("provider_truncated", False)),
                            visible_content_chars=len(raw),
                            provider_reasoning_chars=len(usage.get("provider_reasoning", "")),
                            input_tokens=int(usage.get("input_tokens", 0)),
                            output_tokens=int(usage.get("output_tokens", 0)),
                            hidden_reasoning_tokens=int(usage.get("hidden_reasoning_tokens", 0)),
                        )
                        live_event(
                            problem["id"],
                            spec.name,
                            step_num,
                            "API RESPONSE",
                            (
                                f"total={record.request_latency_sec:.2f}s, "
                                f"service={record.service_time_all_attempts_sec:.2f}s, "
                                f"rpm_wait={record.rate_limit_wait_sec:.2f}s, "
                                f"retry_wait={record.retry_wait_sec:.2f}s, "
                                f"attempts={record.request_attempts}, finish={record.finish_reason or 'unknown'}"
                            ),
                        )
                        trace_reasoning = record.reasoning
                        if record.provider_reasoning:
                            trace_reasoning += "\nProvider reasoning channel:\n" + record.provider_reasoning
                        live_event(problem["id"], spec.name, step_num, "REASONING", trace_reasoning)

                        if action["action_type"] == "answer":
                            consecutive_format_errors = 0
                            record.final_answer = action["content"]
                            traj.final_answer = action["content"]
                            traj.finished = bool(traj.final_answer.strip())
                            traj.stop_reason = "explicit_final_answer"
                            traj.steps.append(asdict(record))
                            live_event(problem["id"], spec.name, step_num, "FINAL ANSWER", traj.final_answer)
                            break

                        if action["action_type"] == "code":
                            record.code = action["content"]
                            if force_final_answer:
                                answer_only_violations += 1
                                record.action_type = "none"
                                record.error = "Final-answer-only instruction was ignored"
                                record.observation = (
                                    "No code executed: a concrete final answer was required on this turn."
                                )
                                traj.steps.append(asdict(record))
                                live_event(problem["id"], spec.name, step_num, "ANSWER REQUIRED", record.code)
                                force_final_next = True
                                prefer_final_next = False
                                if answer_only_violations >= cfg.max_answer_only_violations:
                                    traj.stop_reason = "answer_only_violation_limit"
                                    break
                                continue
                            action_signature = hashlib.sha256(
                                re.sub(r"\s+", "", record.code).encode()
                            ).hexdigest()
                            if action_signature == previous_action_signature:
                                repeated_actions += 1
                                record.action_type = "none"
                                record.error = "Repeated code action was not executed"
                                record.observation = "No code executed: exact repeated action made no new progress."
                                traj.steps.append(asdict(record))
                                live_event(problem["id"], spec.name, step_num, "REPEATED ACTION", record.code)
                                force_final_next = True
                                prefer_final_next = False
                                if repeated_actions >= cfg.max_repeated_actions:
                                    traj.stop_reason = "repeated_action_limit"
                                    break
                                continue
                            previous_action_signature = action_signature
                            repeated_actions = 0
                            consecutive_format_errors = 0
                            force_final_next = False
                            prefer_final_next = False
                            live_event(problem["id"], spec.name, step_num, "CODE", record.code)
                            result = sandbox.run(record.code)
                            executed_tool_actions += 1
                            record.stdout = result.stdout
                            record.stderr = result.stderr
                            record.code_success = result.ok
                            record.code_blocked = result.blocked
                            record.tool_elapsed_sec = result.elapsed_sec
                            record.observation = result.observation(cfg.max_observation_chars)
                            record.evidence_hash = hashlib.sha256(
                                (record.code + "\n" + record.observation).encode()
                            ).hexdigest()[:16]
                            traj.steps.append(asdict(record))
                            live_event(problem["id"], spec.name, step_num, "OBSERVATION", record.observation)
                            if result.ok:
                                consecutive_code_failures = 0
                                prefer_final_next = bool(result.stdout.strip())
                            else:
                                consecutive_code_failures += 1
                                if consecutive_code_failures >= 2:
                                    force_final_next = True
                            continue

                        record.error = "No valid action parsed: " + (record.parse_error or "unknown")
                        if record.provider_truncated:
                            record.observation = (
                                "No action accepted: the provider stopped at its output limit before "
                                "returning a concrete visible action."
                            )
                        elif record.parse_error == "template_placeholder":
                            record.observation = (
                                "No action accepted: template placeholder text is neither executable "
                                "code nor a concrete final answer."
                            )
                        else:
                            record.observation = (
                                "No action accepted: response did not contain one concrete visible action "
                                f"({record.parse_error or 'unknown parse error'})."
                            )
                        traj.steps.append(asdict(record))
                        format_body = raw or record.provider_reasoning
                        live_event(problem["id"], spec.name, step_num, "FORMAT ERROR", format_body)
                        consecutive_format_errors += 1
                        force_final_next = True
                        prefer_final_next = False
                        if consecutive_format_errors >= cfg.max_consecutive_format_errors:
                            traj.stop_reason = "format_error_limit"
                            break
                    else:
                        traj.stop_reason = "max_steps"
            finally:
                traj.elapsed_sec = time.perf_counter() - start
            # Never promote an intermediate tool value to final_answer. Only an explicit answer action counts.
            return traj
        '''
    ),
    code(
        r'''
        # Deterministic controller regressions from observed provider outputs.
        placeholder_code = parse_action(
            "Reasoning step: compute\n```python\n<Python code; always print the result you need>\n```"
        )
        placeholder_answer = parse_action(
            "Reasoning step: conclude\n```answer\n\\n<final answer only>\\n\n```"
        )
        prefixed_answer = parse_action("Reasoning step: conclude\n```answer\nanswer\n48\n```")
        provider_only = parse_action("", "Thus the final answer is 10.")
        multiple_actions = parse_action(
            "Reasoning step: compute\n```python\nprint(1)\n```\n```answer\n1\n```"
        )
        mislabeled_reasoning = parse_action("```python\nReasoning step: use the area formula\n```")
        mislabeled_answer = parse_action("```python\nanswer\nmu = 1/(2*n + 1)\n```")
        markdown_answer = parse_action("**Answer**\n\n\\[p-q\\]")
        bold_colon_answer = parse_action("**Reasoning step:** done\n\n**Answer:**  \n1")
        generic_fenced_answer = parse_action("```\n3\n```")
        plain_display_answer = parse_action(
            "The double series equals the following.\n\\[\\sum_{j,k}a_{j,k}=p-q\\]"
        )

        assert placeholder_code["parse_error"] == "template_placeholder"
        assert placeholder_answer["parse_error"] == "template_placeholder"
        assert prefixed_answer["action_type"] == "answer" and prefixed_answer["content"] == "48"
        assert provider_only["action_type"] == "none"  # Reasoning metadata is not a final answer span.
        assert multiple_actions["parse_error"] == "multiple_actions"
        assert mislabeled_reasoning["parse_error"] == "answer_text_in_code_block"
        assert mislabeled_answer["parse_error"] == "answer_text_in_code_block"
        assert markdown_answer["action_type"] == "answer" and markdown_answer["content"] == "p-q"
        assert bold_colon_answer["action_type"] == "answer" and bold_colon_answer["content"] == "1"
        assert generic_fenced_answer["action_type"] == "answer" and generic_fenced_answer["content"] == "3"
        assert plain_display_answer["action_type"] == "answer" and plain_display_answer["content"] == "p-q"
        print("Controller self-test passed: placeholders and mislabeled code rejected; visible-answer policy enforced.")
        '''
    ),
    code(
        r'''
        class JSONLCheckpoint:
            def __init__(self, output_dir: str, run_name: str, specs: List[ModelSpec]):
                self.root = Path(output_dir) / "checkpoints" / run_name
                self.root.mkdir(parents=True, exist_ok=True)
                self.paths = {s.name: self.root / f"{s.name}.jsonl" for s in specs}
                self.lock = threading.Lock()

            def load(self) -> Dict[str, Dict[str, Trajectory]]:
                data = {name: {} for name in self.paths}
                for name, path in self.paths.items():
                    if not path.exists():
                        continue
                    for line_number, line in enumerate(
                        path.read_text(encoding="utf-8").splitlines(),
                        start=1,
                    ):
                        if line.strip():
                            try:
                                traj = Trajectory.from_dict(json.loads(line))
                            except Exception as exc:
                                tqdm.write(
                                    f"Skipping corrupt checkpoint line {path.name}:{line_number}: {exc}"
                                )
                                continue
                            data[name][traj.problem_id] = traj
                return data

            def append(self, traj: Trajectory):
                path = self.paths[traj.agent_name]
                with self.lock, path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(traj.to_dict(), ensure_ascii=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())

            def clear(self):
                for path in self.paths.values():
                    if path.exists():
                        path.unlink()


        def retryable_infrastructure_checkpoint(traj: Trajectory) -> bool:
            if traj.stop_reason in {"worker_exception", "trajectory_timeout"}:
                return True
            if traj.stop_reason != "api_error":
                return False
            errors = [
                error
                for step in traj.steps
                for error in (step.get("retry_errors") or [])
            ]
            if not errors:
                return True
            retryable_statuses = {408, 409, 429, 500, 502, 503, 504}
            retryable_types = {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"}
            return any(
                error.get("status") in retryable_statuses or error.get("type") in retryable_types
                for error in errors
            )


        def run_experiment(
            problems: List[Dict],
            specs: List[ModelSpec],
            cfg: Config,
            run_name: str,
            live: bool = True,
            resume: bool = True,
        ) -> Dict[str, List[Trajectory]]:
            """Run one independent sequential problem lane per model, with all model lanes concurrent."""
            old_live = cfg.live_trace
            cfg.live_trace = live
            checkpoint = JSONLCheckpoint(cfg.output_dir, run_name, specs)
            if not resume:
                checkpoint.clear()
            existing = checkpoint.load() if resume else {s.name: {} for s in specs}
            retrying = []
            if resume and cfg.retry_infrastructure_failures_on_resume:
                for spec in specs:
                    for problem_id, traj in list(existing[spec.name].items()):
                        if retryable_infrastructure_checkpoint(traj):
                            retrying.append((spec.name, problem_id, traj.stop_reason))
                            del existing[spec.name][problem_id]
                if retrying:
                    print(
                        f"Retrying {len(retrying)} infrastructure-failed checkpoint trajectories; "
                        f"first entries: {retrying[:10]}"
                    )
            problem_ids = {problem["id"] for problem in problems}
            already_complete = sum(
                problem_id in existing[spec.name]
                for spec in specs
                for problem_id in problem_ids
            )
            progress = tqdm(
                total=len(problems) * len(specs),
                initial=already_complete,
                desc=f"{run_name}: generated trajectories",
                unit="traj",
                dynamic_ncols=True,
            )

            def run_model_lane(spec: ModelSpec):
                """One key/model processes its problems sequentially; different model lanes overlap."""
                consecutive_infrastructure_failures = 0
                for problem_index, problem in enumerate(problems, start=1):
                    if problem["id"] in existing[spec.name]:
                        continue
                    tqdm.write(
                        f"[{spec.name}] starting problem {problem_index}/{len(problems)}: {problem['id']}"
                    )
                    try:
                        traj = run_react_trajectory(problem, spec, cfg)
                    except Exception:
                        traj = Trajectory(
                            problem_id=problem["id"], dataset=problem["dataset"],
                            subject=problem["subject"], level=problem["level"],
                            problem_text=problem["problem"], gold_answer=problem["gold_answer"],
                            agent_name=spec.name, provider=spec.provider, model_id=spec.model_id,
                            stop_reason="worker_exception",
                            steps=[asdict(StepRecord(step_num=0, error=traceback.format_exc()))],
                        )
                    checkpoint.append(traj)
                    existing[spec.name][problem["id"]] = traj
                    with _print_lock:
                        progress.update(1)
                        progress.set_postfix(
                            lane=spec.name,
                            problem=f"{problem_index}/{len(problems)}",
                            steps=traj.total_steps,
                            sec=f"{traj.elapsed_sec:.1f}",
                            finished=traj.finished,
                            refresh=True,
                        )
                    tqdm.write(
                        f"[{spec.name}] checkpointed problem {problem_index}/{len(problems)}: "
                        f"steps={traj.total_steps}, finished={traj.finished}, stop={traj.stop_reason}"
                    )
                    if retryable_infrastructure_checkpoint(traj):
                        consecutive_infrastructure_failures += 1
                        cooldown = (
                            cfg.lane_breaker_cooldown_sec
                            if consecutive_infrastructure_failures >= cfg.lane_breaker_after_failures
                            else cfg.lane_api_error_cooldown_sec
                        )
                        tqdm.write(
                            f"[{spec.name}] infrastructure cooldown {cooldown:.0f}s after "
                            f"{consecutive_infrastructure_failures} consecutive failure(s)"
                        )
                        time.sleep(cooldown)
                    else:
                        consecutive_infrastructure_failures = 0

            try:
                with ThreadPoolExecutor(max_workers=min(cfg.parallel_models, len(specs))) as pool:
                    futures = {pool.submit(run_model_lane, spec): spec for spec in specs}
                    for future in as_completed(futures):
                        spec = futures[future]
                        future.result()
                        tqdm.write(f"[{spec.name}] model lane complete")
            finally:
                progress.close()
                cfg.live_trace = old_live
            ordered = {}
            order = {p["id"]: i for i, p in enumerate(problems)}
            for spec in specs:
                selected = [
                    traj for problem_id, traj in existing[spec.name].items()
                    if problem_id in order
                ]
                ordered[spec.name] = sorted(selected, key=lambda t: order[t.problem_id])
            return ordered
        '''
    ),
])


# Patches above are grouped by implementation concern. Order the rendered notebook by section number.
intro = [cells[0]]
sections = defaultdict(list)
current_section = None
for item in cells[1:]:
    source = item.get("source", "").lstrip()
    match = re.match(r"##\s+(\d+)\.", source) if item.get("cell_type") == "markdown" else None
    if match:
        current_section = int(match.group(1))
    if current_section is None:
        intro.append(item)
    else:
        sections[current_section].append(item)
cells = intro + [item for number in sorted(sections) for item in sections[number]]
for index, item in enumerate(cells):
    item["id"] = hashlib.sha1(f"{index}:{item['source']}".encode()).hexdigest()[:8]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=True), encoding="utf-8")
section2_code = next(
    item["source"]
    for index, item in enumerate(cells)
    if index > 0
    and cells[index - 1].get("cell_type") == "markdown"
    and cells[index - 1].get("source", "").lstrip().startswith("## 2.")
    and item.get("cell_type") == "code"
)
SECTION2_OUT.write_text(section2_code, encoding="utf-8")
print(OUT)
print(SECTION2_OUT)


analysis_cells = [
    md(
        """
        # STRIVE Results Analysis Only

        Upload one or more completed STRIVE metrics ZIP files or unzipped export directories. This
        notebook verifies checksums, loads raw trajectories/metrics/PRM records, compares runs, and
        generates plots without generation APIs, NVIDIA/OpenAI keys, sandbox execution, or PRM weights.
        """
    ),
    code(
        r'''
        %pip install -q pandas numpy matplotlib seaborn
        '''
    ),
    code(
        r'''
        import hashlib
        import json
        import shutil
        import zipfile
        from collections import defaultdict
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
        '''
    ),
    code(
        r'''
        # Add metrics-result ZIP files or unzipped result directories here.
        RESULT_INPUTS = [
            # "/kaggle/input/strive-results/paper_math200_olympiad100_v7_....zip",
        ]
        ANALYSIS_OUTPUT_DIR = Path("/kaggle/working/strive_analysis")
        ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        '''
    ),
    code(
        r'''
        def resolve_bundle_root(value: str) -> Path:
            source = Path(value).expanduser().resolve()
            if not source.exists():
                raise FileNotFoundError(source)
            if source.is_file():
                if source.suffix.lower() != ".zip":
                    raise ValueError(f"Expected a ZIP or directory, received {source}")
                target = ANALYSIS_OUTPUT_DIR / "loaded" / source.stem
                if not target.exists():
                    target.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(source) as archive:
                        archive.extractall(target)
                source = target
            manifests = list(source.rglob("manifest.json"))
            if len(manifests) != 1:
                raise RuntimeError(f"Expected one manifest.json under {source}, found {len(manifests)}")
            return manifests[0].parent


        def read_jsonl(path: Path):
            with path.open(encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]


        def load_metrics_bundle(value: str):
            root = resolve_bundle_root(value)
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            for name, expected in manifest.get("sha256", {}).items():
                actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
                if actual != expected:
                    raise RuntimeError(f"Checksum mismatch: {root / name}")
            metrics = read_jsonl(root / "metrics.jsonl")
            trajectories = read_jsonl(root / "trajectories.jsonl")
            prm_path, problems_path = root / "prm_scores.json", root / "problems.jsonl"
            return {
                "root": root,
                "run_name": manifest["run_name"],
                "manifest": manifest,
                "summary": pd.read_csv(root / "summary.csv"),
                "metrics": metrics,
                "metrics_frame": pd.json_normalize(metrics, sep="."),
                "trajectories": trajectories,
                "trajectories_frame": pd.json_normalize(trajectories, sep="."),
                "prm_scores": json.loads(prm_path.read_text()) if prm_path.exists() else {},
                "problems": read_jsonl(problems_path) if problems_path.exists() else [],
            }


        def compare_bundles(bundles):
            combined = pd.concat(
                [bundle["summary"].assign(run=bundle["run_name"]) for bundle in bundles],
                ignore_index=True,
            )
            display(combined)
            core = combined.melt(
                id_vars=["run", "agent"],
                value_vars=["C", "G", "V", "Q", "PTU", "R_harmful"],
                var_name="metric",
                value_name="score",
            )
            grid = sns.catplot(
                data=core, x="agent", y="score", hue="metric", col="run",
                kind="bar", height=5, aspect=1.5, sharex=False,
            )
            grid.set_xticklabels(rotation=30)
            grid.set(ylim=(0, 1))
            grid.fig.suptitle("STRIVE core metrics by run", y=1.04)
            core_path = ANALYSIS_OUTPUT_DIR / "comparison_core_metrics.png"
            grid.fig.savefig(core_path, dpi=180, bbox_inches="tight")
            plt.show()

            fig, ax = plt.subplots(figsize=(9, 6))
            sns.scatterplot(
                data=combined, x="avg_billable_tokens", y="V", hue="agent", style="run", s=100, ax=ax
            )
            ax.set_title("Cost versus verified success")
            fig.tight_layout()
            cost_path = ANALYSIS_OUTPUT_DIR / "comparison_cost_vs_verified.png"
            fig.savefig(cost_path, dpi=180, bbox_inches="tight")
            plt.show()
            return combined, {"core_plot": core_path, "cost_plot": cost_path}
        '''
    ),
    code(
        r'''
        if not RESULT_INPUTS:
            print("Set RESULT_INPUTS to one or more uploaded metrics ZIP files or directories.")
        else:
            bundles = [load_metrics_bundle(path) for path in RESULT_INPUTS]
            comparison_table, comparison_plots = compare_bundles(bundles)
            comparison_table.to_csv(ANALYSIS_OUTPUT_DIR / "comparison_summary.csv", index=False)
            print("Analysis outputs:", ANALYSIS_OUTPUT_DIR)
        '''
    ),
]
for index, item in enumerate(analysis_cells):
    item["id"] = hashlib.sha1(f"analysis:{index}:{item['source']}".encode()).hexdigest()[:8]
analysis_notebook = {
    "cells": analysis_cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
ANALYSIS_OUT.write_text(json.dumps(analysis_notebook, indent=1, ensure_ascii=True), encoding="utf-8")
print(ANALYSIS_OUT)
