#!/usr/bin/env python3
"""Example client for the /encode microservice."""

from __future__ import annotations

import json
import sys
from typing import Any, Dict

import requests

SERVICE_URL = "http://localhost:8000/encode"


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python client.py <pdf>")
        raise SystemExit(1)
    path = sys.argv[1]
    with open(path, "rb") as fh:
        response = requests.post(
            SERVICE_URL,
            params={"preset": "reports", "budget": 256, "tokenizer": "cl100k_base"},
            files={"file": (path, fh, "application/pdf")},
            timeout=60,
        )
    response.raise_for_status()
    payload: Dict[str, Any] = response.json()
    print("[3DCF] Context preview:")
    print(payload["context_text"][:500], "...\n", sep="")
    print("[3DCF] Metrics:")
    print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
