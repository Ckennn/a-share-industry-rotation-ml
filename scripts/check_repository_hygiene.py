#!/usr/bin/env python3
"""Reject private paths, secrets, binary deliverables, and oversized Git files."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

FORBIDDEN_SUFFIXES = {".db", ".sqlite", ".pdf", ".docx", ".xlsx", ".pptx", ".zip", ".7z"}
PRIVATE_PATTERNS = (
    "/home/" + "kken/projects/" + "whz20260304",
    "C:" + "\\Users\\" + "26634",
    "maj" + "fung@",
    "OPENAI_" + "API_KEY=",
    "GITHUB_" + "TOKEN=",
)
MAX_TRACKED_BYTES = 5 * 1024 * 1024
ALLOWED_CSV = {"examples/synthetic_panel.csv"}


@dataclass(frozen=True)
class HygieneFinding:
    path: str
    rule: str
    matched: str

    def __str__(self) -> str:
        return f"{self.path}: {self.rule}: {self.matched}"


def _tracked_files(root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [item.decode("utf-8") for item in completed.stdout.split(b"\0") if item]


def scan_repository(root: Path) -> list[HygieneFinding]:
    """Scan Git-tracked files only, leaving local data and environments outside scope."""

    root = Path(root).resolve()
    findings: list[HygieneFinding] = []
    for relative in _tracked_files(root):
        path = root / relative
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES:
            findings.append(HygieneFinding(relative, "forbidden suffix", suffix))
        if suffix == ".csv" and relative not in ALLOWED_CSV and not relative.startswith("tests/"):
            findings.append(HygieneFinding(relative, "unapproved tracked CSV", suffix))
        size = path.stat().st_size
        if size > MAX_TRACKED_BYTES:
            findings.append(HygieneFinding(relative, "oversized tracked file", str(size)))
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in PRIVATE_PATTERNS:
            if pattern in text:
                findings.append(HygieneFinding(relative, "private or secret pattern", pattern))
    return findings


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    findings = scan_repository(args.root)
    for finding in findings:
        print(finding)
    if findings:
        print(f"repository hygiene failed with {len(findings)} finding(s)")
        return 1
    print("repository hygiene passed: 0 findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
