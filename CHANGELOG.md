# Changelog

All notable changes to Yellow Phoenix are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Linux build verification record: `cargo build --release` green on Ubuntu 22.04
  (2 vCPU/2 GB DO droplet, rustc 1.98.1) + `query_benchmark` smoke numbers at
  10k/100k/1M scale (`results/2026-09-29_linux_build_bench.md`) (2026-09-29).

### Fixed
- `ffi_shootout.rs` allocator-purge helper: previous macOS-gated wrapper recursed
  into itself (stack overflow on `yp_shootout_unload_*`); replaced with
  platform-gated helper — macOS `malloc_zone_pressure_relief`, Linux glibc
  `malloc_trim(0)`, others no-op (2026-09-29).

### Added
- Engine smoke test: self-contained insert/search/reload suite (3 tests) + CI job running it on Ubuntu and macOS (2026-09-29).
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

### Added
- Legal & compliance stack: PRIVACY.md (zero-collection policy), TERMS.md,
  DATA_COMPLIANCE.md (GDPR/CCPA analysis), docs/IP_NOTICE.md (trademark
  status incl. informal "Yellow Phoenix" clearance notes + infringement
  reporting) (2026-09-29). Review with counsel before filing/app launch.
