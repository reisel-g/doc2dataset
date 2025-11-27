# Changelog

All notable changes to this project will be documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Release checklist

Before tagging a new version:

1. Update crate versions in `Cargo.toml` and the `[workspace.package]` metadata.
2. Document the changes in the sections below (Added / Changed / Fixed / Removed).
3. Run `cargo fmt`, `cargo clippy --all-targets --all-features`, `cargo test --all --all-features`.
4. Regenerate `bench/report.html` for the public sample dataset and attach it to the release if it changed.
5. Verify docs (README, INSTALL, SECURITY, CODE_OF_CONDUCT) still reflect the intended release scope.
6. Create a signed git tag `vX.Y.Z`, push, and draft the GitHub release notes using the entries below.

## [Unreleased]
### Added
- Contributor docs: CODE_OF_CONDUCT, SECURITY policy, issue/PR templates, CHANGELOG skeleton.
- README quickstart, “Why 3DCF?”, and sample end-to-end workflow.
- CC0 sample dataset for CI smoke tests.

### Changed
- macOS install guide documents PDFium download, Leptonica symlink, and Linux linker flags.
- Cleaned repository artifacts (`.DS_Store`, nested `.git/`).

### Fixed
- Clarified bench → HTML workflow (`3dcf report`).

## [0.1.0] - 2024-01-01
### Added
- Initial release with `three_dcf_core`, `three_dcf_cli`, `.3dcf` container format, CLI subcommands, bench harness, and plotting scripts.
