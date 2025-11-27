#!/usr/bin/env python3
"""Runs the CLI encoder with strict NumGuard to highlight mismatches."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python main.py <pdf>")
        raise SystemExit(1)
    pdf = Path(sys.argv[1])
    if not pdf.exists():
        raise SystemExit(f"missing file: {pdf}")
    with tempfile.NamedTemporaryFile(suffix=".3dcf") as tmp:
        cmd = [
            "cargo",
            "run",
            "-q",
            "-p",
            "three_dcf_cli",
            "--",
            "encode",
            str(pdf),
            "--out",
            tmp.name,
            "--preset",
            "reports",
            "--budget",
            "256",
            "--strict-numguard",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("[3DCF] encode failed")
            print(result.stderr.strip())
            raise SystemExit(result.returncode)
        stderr = result.stderr.strip()
        if "numeric guard issues" in stderr:
            print("[3DCF] NumGuard warning:")
            print(stderr)
        else:
            print("[3DCF] No numeric mismatches detected.")


if __name__ == "__main__":
    main()
