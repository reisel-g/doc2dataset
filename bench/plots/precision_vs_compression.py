#!/usr/bin/env python3
"""Generate precision vs compression plot and CSV from 3DCF bench page rows."""
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize page-level precision/compression stats from bench JSONL logs.")
    parser.add_argument("input", type=Path, help="Path to bench JSONL with page rows")
    parser.add_argument(
        "--bins",
        default="0,200,400,600,800,1000,1200",
        help="Comma-separated gold-token bin edges (last bin extends to infinity)",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("precision_vs_compression.csv"),
        help="Summary CSV output path",
    )
    parser.add_argument(
        "--png-out",
        type=Path,
        default=Path("precision_vs_compression.png"),
        help="Chart PNG output path",
    )
    return parser.parse_args()


def parse_bins(spec: str) -> List[Tuple[float, Optional[float]]]:
    edges = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if not edges:
        raise ValueError("bin specification must include at least one value")
    edges = sorted(edges)
    ranges = []
    for idx, start in enumerate(edges):
        end = edges[idx + 1] if idx + 1 < len(edges) else None
        ranges.append((start, end))
    return ranges


def format_bin_label(start: float, end: Optional[float]) -> str:
    if end is None:
        return f"{int(start)}+"
    return f"{int(start)}-{int(end)}"


def normalize_budget(value) -> str:
    if value is None:
        return "auto"
    return str(value)


def load_rows(path: Path) -> List[dict]:
    rows = []
    with path.open() as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid json line {idx}: {exc}") from exc
            if row.get("row_type") != "page":
                continue
            rows.append(row)
    if not rows:
        raise RuntimeError("no page rows found in JSONL")
    return rows


def aggregate(rows: List[dict], bins: List[Tuple[float, Optional[float]]]):
    stats: Dict[Tuple[str, int], Dict[str, float]] = {}
    bin_labels = [format_bin_label(start, end) for start, end in bins]
    for row in rows:
        tokens_gold = row.get("tokens_gold_page")
        precision = row.get("precision_page")
        compression = row.get("compression_ratio")
        if tokens_gold is None or precision is None or compression is None:
            continue
        bin_index = None
        for idx, (start, end) in enumerate(bins):
            if tokens_gold < start:
                continue
            if end is None or tokens_gold < end:
                bin_index = idx
                break
        if bin_index is None:
            continue
        budget = normalize_budget(row.get("budget"))
        key = (budget, bin_index)
        bucket = stats.setdefault(key, {"count": 0.0, "precision_sum": 0.0, "compression_sum": 0.0})
        bucket["count"] += 1.0
        bucket["precision_sum"] += precision
        bucket["compression_sum"] += compression
    if not stats:
        raise RuntimeError("no usable page rows after filtering")
    return stats, bin_labels


def compute_series(stats: Dict[Tuple[str, int], Dict[str, float]], bin_labels: List[str]):
    budgets = sorted({budget for budget, _ in stats.keys()}, key=lambda x: (x != "auto", x))
    precision_series: Dict[str, List[float]] = {}
    compression_series: Dict[str, List[float]] = {}
    counts: Dict[Tuple[str, int], float] = {}
    bins = range(len(bin_labels))
    for budget in budgets:
        precision_series[budget] = []
        compression_series[budget] = []
        for idx in bins:
            bucket = stats.get((budget, idx))
            if bucket and bucket["count"] > 0:
                precision_series[budget].append(bucket["precision_sum"] / bucket["count"])
                compression_series[budget].append(bucket["compression_sum"] / bucket["count"])
                counts[(budget, idx)] = bucket["count"]
            else:
                precision_series[budget].append(0.0)
                compression_series[budget].append(0.0)
                counts[(budget, idx)] = 0.0
    return budgets, precision_series, compression_series, counts


def write_csv(csv_path: Path, budgets: List[str], bin_labels: List[str], counts, precision_series, compression_series):
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["budget", "bin", "count", "precision_mean", "compression_mean"])
        for budget in budgets:
            for idx, label in enumerate(bin_labels):
                writer.writerow(
                    [
                        budget,
                        label,
                        int(counts[(budget, idx)]),
                        round(precision_series[budget][idx], 6),
                        round(compression_series[budget][idx], 6),
                    ]
                )


def plot_png(png_path: Path, budgets: List[str], bin_labels: List[str], precision_series, compression_series):
    x = list(range(len(bin_labels)))
    fig, ax = plt.subplots(figsize=(max(6, len(bin_labels) * 1.2), 4 + len(budgets)))
    width = 0.8 / max(1, len(budgets))
    for idx, budget in enumerate(budgets):
        offsets = [val + (idx - (len(budgets) - 1) / 2) * width for val in x]
        ax.bar(offsets, precision_series[budget], width=width, label=f"{budget} precision")
    ax.set_ylabel("Mean precision")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Gold tokens per page (bin)")
    ax.set_ylim(0, 1)

    ax2 = ax.twinx()
    for budget in budgets:
        ax2.plot(x, compression_series[budget], linestyle=":", marker="o", label=f"{budget} compression")
    ax2.set_ylabel("Mean compression ratio")

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="upper right")
    ax.set_title("Precision vs compression per budget")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    bins = parse_bins(args.bins)
    rows = load_rows(args.input)
    stats, bin_labels = aggregate(rows, bins)
    budgets, precision_series, compression_series, counts = compute_series(stats, bin_labels)
    write_csv(args.csv_out, budgets, bin_labels, counts, precision_series, compression_series)
    plot_png(args.png_out, budgets, bin_labels, precision_series, compression_series)


if __name__ == "__main__":
    main()
