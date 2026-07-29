#!/usr/bin/env python3
"""Validate a STRIVE generation ZIP or extracted directory without third-party packages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict


REQUIRED = {
    "generation_manifest.json",
    "generation_health.csv",
    "problems.jsonl",
    "trajectories.jsonl",
}


def find_member(names, basename: str) -> str:
    matches = [name for name in names if Path(name).name == basename]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {basename}; found {len(matches)}")
    return matches[0]


def read_payloads(path: Path) -> Dict[str, bytes]:
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            return {name: archive.read(find_member(names, name)) for name in REQUIRED}
    if path.is_dir():
        candidates = {name: list(path.rglob(name)) for name in REQUIRED}
        bad = {name: values for name, values in candidates.items() if len(values) != 1}
        if bad:
            raise ValueError(f"Expected one copy of each required file: {bad}")
        return {name: candidates[name][0].read_bytes() for name in REQUIRED}
    raise FileNotFoundError(f"Expected a generation ZIP or directory: {path}")


def jsonl_count(payload: bytes, label: str) -> int:
    count = 0
    for line_no, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{label} line {line_no} is not an object")
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", help="Generation ZIP or extracted directory")
    args = parser.parse_args()
    path = Path(args.artifact).expanduser().resolve()
    payloads = read_payloads(path)
    manifest = json.loads(payloads["generation_manifest.json"].decode("utf-8"))

    checksums = manifest.get("sha256", {})
    for name, expected in checksums.items():
        basename = Path(name).name
        if basename in payloads:
            actual = hashlib.sha256(payloads[basename]).hexdigest()
            if actual != expected:
                raise ValueError(f"Checksum mismatch for {basename}: {actual} != {expected}")

    problems = jsonl_count(payloads["problems.jsonl"], "problems.jsonl")
    trajectories = jsonl_count(payloads["trajectories.jsonl"], "trajectories.jsonl")
    health_rows = list(csv.DictReader(io.StringIO(payloads["generation_health.csv"].decode("utf-8"))))
    declared_problems = manifest.get("problem_count")
    declared_trajectories = manifest.get("trajectory_count")
    if declared_problems is not None and int(declared_problems) != problems:
        raise ValueError(f"Problem count mismatch: manifest={declared_problems}, file={problems}")
    if declared_trajectories is not None and int(declared_trajectories) != trajectories:
        raise ValueError(f"Trajectory count mismatch: manifest={declared_trajectories}, file={trajectories}")

    print(f"VALID: {path}")
    print(f"  protocol: {manifest.get('config', {}).get('protocol_version', 'unknown')}")
    print(f"  problems: {problems}")
    print(f"  trajectories: {trajectories}")
    print(f"  health rows: {len(health_rows)}")
    print(f"  checksum fields verified: {len(checksums)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        raise SystemExit(1)
