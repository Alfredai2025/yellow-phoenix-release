# Changelog

All notable changes to Yellow Phoenix are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Governor weekly health exam (2026-09-29) — vine self-exam ledger snapshot (`results/governor/`).
- Community health files: this changelog, CONTRIBUTING.md, CI workflow
  (Rust type-check across all targets + Python syntax validation),
  issue templates.
- `v1.0.0` tag marking the initial public release (2026-09-17).

### Changed
- (ongoing) Development now happens in public on this repository; see
  CONTRIBUTING.md for how to participate.

## [1.0.0] - 2026-09-17

### Added
- Initial public release of the Yellow Phoenix edge-scale semantic
  retrieval engine: 512-bit binary-code HNSW index in a memory-mapped,
  flash-resident graph format; ITQ encoding; on-device query path;
  exact re-rank stage; benchmarks and test suites (Rust + Python).
- AGPL-3.0-or-later license; commercial dual licensing available
  (see LICENSE / README).

[Unreleased]: https://github.com/Alfredai2025/yellow-phoenix-release/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Alfredai2025/yellow-phoenix-release/releases/tag/v1.0.0
