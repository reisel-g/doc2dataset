#!/usr/bin/env python3
"""Example: encode a PDF, send to OpenAI, print metrics + answer."""

from __future__ import annotations

import os
import sys
import textwrap
from typing import Any, Dict

import requests

from three_dcf_py import encode_to_context

QUESTION = "Summarize the top risks for 2025."
MODEL = "gpt-4.1-mini"
API_URL = "https://api.openai.com/v1/responses"


def build_prompt(context_text: str, question: str) -> str:
    return f"{context_text}\n\nQuestion: {question}\n\nAnswer:"


def post_openai(prompt: str) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": MODEL, "input": prompt},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def extract_answer(payload: Dict[str, Any]) -> str:
    output = payload.get("output") or []
    for block in output:
        for chunk in block.get("content", []):
            text = chunk.get("text")
            if text:
                return text.strip()
    choices = payload.get("choices") or []
    if choices:
        return choices[0].get("message", {}).get("content", "").strip()
    return "(empty response)"


def format_number(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python main.py <pdf-or-text>")
        raise SystemExit(1)
    path = sys.argv[1]
    context = encode_to_context(path, preset="reports", budget=256, tokenizer="cl100k_base")
    prompt = build_prompt(context.text, QUESTION)
    response = post_openai(prompt)
    answer = extract_answer(response)
    usage = response.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    print("[3DCF] Context ready")
    print(f"[3DCF] Pages:                  {context.metrics['pages']}")
    print(
        f"[3DCF] Cells total / kept:     {context.metrics['cells_total']} / {context.metrics['cells_kept']}"
    )
    if context.metrics["raw_tokens_estimate"]:
        raw = context.metrics["raw_tokens_estimate"]
        print(f"[3DCF] Est. raw tokens:        {format_number(raw)}")
    if context.metrics["compressed_tokens_estimate"]:
        comp = context.metrics["compressed_tokens_estimate"]
        print(f"[3DCF] Compressed tokens:      {format_number(comp)}")
    print(f"[3DCF] Prompt tokens (LLM):    {format_number(prompt_tokens)}")
    print(f"[3DCF] Completion tokens:      {format_number(completion_tokens)}")
    print()  # answer goes last for piping
    print(textwrap.dedent(answer).strip())


if __name__ == "__main__":
    main()
