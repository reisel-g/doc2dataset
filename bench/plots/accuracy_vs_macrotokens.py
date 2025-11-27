#!/usr/bin/env python3
"""Generate accuracy vs macro-token scatter plot from bench doc rows."""
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate doc-level accuracy metrics from bench JSONL logs.")
    parser.add_argument("inputs", nargs="+", type=Path, help="One or more bench JSONL files")
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("accuracy_vs_macrotokens.csv"),
        help="Summary CSV output path",
    )
    parser.add_argument(
        "--png-out",
        type=Path,
        default=Path("accuracy_vs_macrotokens.png"),
        help="Scatter plot PNG output path",
    )
    parser.add_argument(
        "--metric",
        choices=["cer", "wer"],
        default="cer",
        help="Accuracy metric to plot on the Y axis",
    )
    return parser.parse_args()


def load_rows(paths: List[Path]) -> List[dict]:
    rows = []
    for path in paths:
        with path.open() as handle:
            for idx, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid json line {idx} in {path}: {exc}") from exc
                if row.get("row_type") != "doc":
                    continue
                rows.append(row)
    if not rows:
        raise RuntimeError("no doc rows found in provided JSONL files")
    return rows


def aggregate(rows: List[dict], metric: str):
    aggregates: Dict[str, Dict[str, float]] = {}
    for row in rows:
        run_id = row.get("run_id") or "unknown"
        record = aggregates.setdefault(
            run_id,
            {
                "preset": row.get("preset", ""),
                "budget": row.get("budget"),
                "mode": row.get("mode"),
                "docs": 0.0,
                "pages": 0.0,
                "tokens_3dcf": 0.0,
                "cells_weighted": 0.0,
                "cer_sum": 0.0,
                "cer_count": 0.0,
                "wer_sum": 0.0,
                "wer_count": 0.0,
            },
        )
        pages = max(1, int(row.get("pages") or 0))
        record["docs"] += 1.0
        record["pages"] += pages
        record["tokens_3dcf"] += row.get("tokens_3dcf", 0)
        record["cells_weighted"] += (row.get("avg_cells_kept_per_page") or 0.0) * pages
        cer = row.get("cer")
        if cer is not None:
            record["cer_sum"] += cer
            record["cer_count"] += 1.0
        wer = row.get("wer")
        if wer is not None:
            record["wer_sum"] += wer
            record["wer_count"] += 1.0
    summaries = []
    for run_id, info in aggregates.items():
        if info["pages"] == 0:
            continue
        tokens_per_page = info["tokens_3dcf"] / info["pages"]
        cells_per_page = info["cells_weighted"] / info["pages"] if info["pages"] else 0.0
        cer_mean = info["cer_sum"] / info["cer_count"] if info["cer_count"] else None
        wer_mean = info["wer_sum"] / info["wer_count"] if info["wer_count"] else None
        summaries.append(
            {
                "run_id": run_id,
                "preset": info["preset"],
                "budget": info["budget"],
                "mode": info["mode"],
                "docs": int(info["docs"]),
                "pages": int(info["pages"]),
                "tokens_per_page": tokens_per_page,
                "cells_per_page": cells_per_page,
                "cer_mean": cer_mean,
                "wer_mean": wer_mean,
                "plot_metric": cer_mean if metric == "cer" and cer_mean is not None else wer_mean,
            }
        )
    summaries = [row for row in summaries if row["plot_metric"] is not None]
    if not summaries:
        raise RuntimeError(f"no rows contained the requested metric ({metric.upper()})")
    return summaries


def write_csv(csv_path: Path, summaries: List[dict]):
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_id",
                "preset",
                "mode",
                "budget",
                "documents",
                "pages",
                "tokens_per_page",
                "cells_per_page",
                "mean_cer",
                "mean_wer",
            ]
        )
        for row in summaries:
            writer.writerow(
                [
                    row["run_id"],
                    row["preset"],
                    row["mode"],
                    row["budget"],
                    row["docs"],
                    row["pages"],
                    round(row["tokens_per_page"], 6),
                    round(row["cells_per_page"], 6),
                    "" if row["cer_mean"] is None else round(row["cer_mean"], 6),
                    "" if row["wer_mean"] is None else round(row["wer_mean"], 6),
                ]
            )


def plot_png(png_path: Path, summaries: List[dict], metric: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    x_vals = [row["tokens_per_page"] for row in summaries]
    y_vals = [row["plot_metric"] for row in summaries]
    scatter = ax.scatter(x_vals, y_vals)
    for row in summaries:
        ax.annotate(row["run_id"], (row["tokens_per_page"], row["plot_metric"]), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Average 3DCF tokens per page")
    ax.set_ylabel(f"Mean {metric.upper()}")
    ax.set_title("Accuracy vs macro tokens")
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    rows = load_rows(args.inputs)
    summaries = aggregate(rows, args.metric)
    write_csv(args.csv_out, summaries)
    plot_png(args.png_out, summaries, args.metric)


if __name__ == "__main__":
    main()
