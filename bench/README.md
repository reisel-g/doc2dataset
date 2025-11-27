# Benchmarks

Use `3dcf bench <datasets>` to sweep entire corpora and capture:
- encode/decode latency per document (ms)
- token counts before/after compression
- aggregate savings (mean / median)
- optional accuracy metrics (CER/WER, numeric integrity) when `--gold` points at reference text

Outputs are appended as JSONL rows via `--output bench/results.jsonl`. Convert the log into an HTML dashboard with `3dcf report`:

```
RUST_LOG=info cargo run -p three_dcf_cli -- bench datasets/reports \
  --preset reports --tokenizer cl100k_base --budget 256 \
  --gold datasets/gold_reports --output bench/results.jsonl
cargo run -p three_dcf_cli -- report bench/results.jsonl --out bench/report.html
```

`bench/report.html` summarises mean/median savings plus per-document stats so regressions are visible at a glance.

Flags of note:
- `--mode encode|decode|full` controls whether the runner encodes sources, evaluates existing `.3dcf`
  files, or does both.
- `--budgets 64,100,auto` sweeps multiple macro-token budgets; `auto` keeps the encoder default.
- `--gold /path/to/gold` enables CER/WER/NumGuard metrics and emits per-page rows in the JSONL log.
- JSONL doc rows now include encode/decode throughput (`encode_pages_per_s`, `decode_pages_per_s`),
  peak RSS (`mem_peak_mb`), and `numguard_mismatches` so you can spot performance or numeric-integrity
  regressions across runs.
- `--tokenizer-file path/to/custom.json` lets you pair `--tokenizer custom` with a JSON BPE export
  (pat_str + mergeable_ranks + special_tokens) so savings reflect your downstream tokenizer.

## Analysis scripts

Once you have a JSONL with doc + page rows, generate the charts called out in the implementation plan:

```
python bench/plots/precision_vs_compression.py bench/results.jsonl \
  --bins 0,200,400,600,800,1000,1200 \
  --csv-out bench/precision_vs_compression.csv \
  --png-out bench/precision_vs_compression.png

python bench/plots/accuracy_vs_macrotokens.py bench/results.jsonl \
  --metric cer \
  --csv-out bench/accuracy_vs_macrotokens.csv \
  --png-out bench/accuracy_vs_macrotokens.png
```

The first script reads page-level rows, bins them by gold tokens per page, and renders a bar+line chart
capturing mean precision (bars) versus compression ratio (dotted line) for every budget present in the run.
The second script consumes doc rows (you can pass multiple JSONLs if each corresponds to a different
`run_id`) and emits a scatter plot showing how mean CER/WER moves as the average macro tokens per page
change. Both scripts also produce CSV summaries so the metrics can feed other dashboards.
