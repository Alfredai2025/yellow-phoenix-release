# Changelog

All notable changes to Yellow Phoenix are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Research — overnight experiment series (2026-10-01)
- **Sheaf experiment REFUTED**: quarter-block hash features as gate inputs
  scored 84.3% vs 86.1% baseline (5.65M corpus, n=1000). Branch closed.
- **Bit-entropy audit**: 876/1024 live bits but effective rank 170/1024 —
  hash bits are heavily redundant (explains the sheaf refutation).
- **Twin-graph**: 26.32% of the 5.65M-record corpus sits in near-duplicate
  clusters (728k clusters, max size 249).
- **Hubness probe: negative** (skew 3.18, max 8/5000 shortlists) — impostors
  are not hub documents. Branch closed.
- **Tie-aware scoring validated** (independent clusters from arXiv IDs +
  exact text, not the hash): strict R@1 86.1% → cluster-aware 100.0%
  (≥97.4% at 95% CI); 0 foreign code collisions in 139 adjudicated misses.
  Spec: `docs/tie_aware_scoring.md` in the working repo; 22 MB uint32
  cluster sidecar (watch-compatible).
- Encoder pilot @100k: MiniLM 100% / BGE 100% own-space ceiling — wall is a
  large-scale (5.65M) phenomenon; BGE@5.65M ceiling probe running.
- Data note: 10,019 DB rows added after the 2026-09-29 index build —
  incremental ingest pass needed.

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
