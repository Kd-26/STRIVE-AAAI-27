#!/usr/bin/env python3
"""Lightweight structural and secret-pattern checks for shipped STRIVE notebooks."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
FORBIDDEN = [
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"nvapi-[A-Za-z0-9_-]{20,}"),
    re.compile(r"/Users/kshitijdahiya/"),
]
REQUIRED = {
    "01_generate_trajectories.ipynb": ["STRIVE-Math", "Python sandbox", "PRM"],
    "05_compute_final_metrics.ipynb": ["Final Metrics", "Deduplicate"],
    "06_advanced_analysis.ipynb": ["Advanced Analysis", "trace-derived"],
}


def text(cell) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def main() -> int:
    failures = []
    notebooks = sorted(NOTEBOOKS.glob("*.ipynb"))
    if len(notebooks) != 6:
        failures.append(f"Expected six notebooks; found {len(notebooks)}")
    for path in notebooks:
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            cells = notebook.get("cells", [])
            if not cells or any(cell.get("cell_type") not in {"code", "markdown", "raw"} for cell in cells):
                failures.append(f"{path.name}: invalid cell structure")
                continue
            joined = "\n".join(text(cell) for cell in cells)
            for pattern in FORBIDDEN:
                if pattern.search(joined):
                    failures.append(f"{path.name}: forbidden personal path or credential pattern")
            for phrase in REQUIRED.get(path.name, []):
                if phrase not in joined:
                    failures.append(f"{path.name}: missing expected phrase {phrase!r}")
        except Exception as exc:
            failures.append(f"{path.name}: {exc}")
    if failures:
        print("NOTEBOOK VALIDATION FAILED", file=sys.stderr)
        print("\n".join(f"- {value}" for value in failures), file=sys.stderr)
        return 1
    print(f"NOTEBOOK VALIDATION PASSED: {len(notebooks)} notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
