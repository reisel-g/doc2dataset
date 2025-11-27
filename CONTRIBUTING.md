# Contributing to 3DCF

Thank you for helping shape 3DCF! We follow a lightweight process:

1. **Discuss first** – open an issue describing the bug/feature. Attach sample docs whenever possible.
2. **Fork & branch** – branch off `main`, keep commits focused, add tests for fixes/features.
3. **Coding standards** – run `cargo fmt`, `cargo clippy --all-targets --all-features`, and `cargo test --all --all-features` before opening a PR. Snapshot changes should run `cargo insta review` locally.
4. **Licensing** – all contributions are Apache-2.0; include a note in the PR description confirming you own the code or have permission to contribute it.
5. **Review process** – expect automated CI plus human review focusing on correctness, performance, and reproducing numbers in `bench/report.html`.

Helpful targets:

- `cargo test -p three_dcf_core tests/property_roundtrip.rs -- --nocapture`
- `cargo run -p three_dcf_cli -- synth datasets/synthetic --count 10`
- `cargo run -p three_dcf_cli -- bench datasets --preset reports --tokenizer cl100k_base`

For large features (bindings, new formats, OCR engines), submit an RFC issue describing motivation, design choices, and test datasets. We love reproducible corpora and benchmark diffs! 🙌
