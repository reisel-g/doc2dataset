# Datasets

Place evaluation corpora here (PDF, TXT, HTML, scans). Suggested structure:

```
datasets/
  reports/
    gold/          # reference text for CER
    10k_2023.pdf
  slides/
    product_launch.pdf
  scans/
    lab_report_scan.pdf
  synthetic/
    synthetic_000.md
```

### Tiny sample set

`sample/` ships with a CC0 markdown document (`sample_report.md`) so CI and
new contributors can smoke-test `encode → decode → stats` without downloading
external PDFs.

### Synthetic corpora

Generate quick placeholder corpora with the CLI:

```
cargo run -p three_dcf_cli -- synth datasets/synthetic --count 25 --seed 1337
```

### Larger datasets

Each leaf folder can include an optional `gold/` subdirectory with reference
text extracted via `pdftotext` or Tesseract. `3dcf bench` walks folders
recursively and skips unsupported extensions automatically. Keep any licensed
corpora outside the repo and document fetch instructions (URLs, checksums) in a
separate file such as `datasets/README.local.md`.
