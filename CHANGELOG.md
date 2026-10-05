# Changelog

All notable changes to Yellow Phoenix are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — ANN campaign engines and overlays (2026-10-05, all measurements SIFT-1M 10k queries, M3 Pro)

**Int8Hnsw engine (new, fifth benchmark candidate: `yellow-phoenix-int8`):**
HNSW over 128-byte residual-int8 codes (coarse k-means K=4096 + per-dim residual clip) with
dual edge sets, unified clipped-residual distance (build and query use the same per-candidate-cell
residual + maddubs SIMD identity — bit-exact), per-cell gateway entry (T7 compass), Vamana-style
prune. Measured: ef128 0.9919@~940 QPS, ef512 0.9976@~500 QPS (beyond the binary stack ceiling),
prune a0.9 -0.55pp for +11% QPS. Council of 3 external AIs reviewed all kernels and the distance
identity (2 independent correctness proofs per critical section).
New bins: build_int8_graph, int8_bench. New SIMD kernels: i8_dot_128 (AVX2 maddubs / NEON widen),
i8_l2_128, i8_sq_128, f32_i8_dot_128 (all runtime-dispatched, scalar fallback).
MEASURED-NULL record: float reconstruction-L2 v2 (identity-verified) was recall-neutral and 16%
slower (widening-chain cost) — reverted; scalar-vs-SIMD dot was bit-exact but not the bottleneck.

**BinaryHNSW overlays (ship-candidate stack for the PR family):**
- `prune_diverse` / `prune_diverse_adaptive` / `prune_diverse_geo` (post-build Vamana-style,
  alpha semantics: higher alpha cuts more) + prune_graph bin. Measured champion: alpha=0.9,
  -24% visited, -0.0016 recall @ef64, zero recall cost ef>=256, all layers.
- `search_with_ef_from` / `search_flat_from` (warm-start entry), HOPS visit counter,
  fused_score sketch fusion (thread-local, integer fixed-point, +0.0005 recall at zero cost,
  shuffle-control validated), node_hash/node_count/neighbors accessors, d==0 duplicate-prune fix.
- New bins: phyllo_bench (compass A/B), verify_skip_bench (ITQ verify-skip LUT, measured
  marginal — not in ship stack), bridge_graph (measured neutral — archived experiment).

**PqHnsw:** sdc pairwise distance, prune_diverse port, search_with_ef_from (transfer-test
harness). Measured: prune does NOT transfer to noisy ADC distances (-8.3pp); transfers cleanly to
low-noise int8 (-0.55pp) and exact binary (-0.16pp) — noise-ordered, council-predicted.

**two_stage_bench:** p99 latency added to output.
**yp_ann_server:** --funnel ADC narrow stage (measured: overhead-bound, killed — code kept as
honest record).

Artifacts (not committed, large): ~/yp_ann/data (graphs, JSONs, codec files).


### Publications — Paper 3 v1.0.4 (2026-10-04)
- **Paper 3 "Yellow Phoenix: Semantic Search on an Apple Watch" v1.0.4**
  published on Zenodo (record v3, DOI 10.5281/zenodo.23130414; concept
  10.5281/zenodo.23118337). Changes vs v1.0.3: "fully offline" claim in the
  abstract + keyword; offline sentences in §1/§3; new §10.1 Design projection
  (Faraday-cage motivation, papers-per-gigabyte, 34.8 ms wrist vs ~1 s cloud
  round-trip); Figure 3 repositioned (fixes near-empty page); watermark now
  behind text. Additions only — zero deletions vs v1.0.3 (line-level diff).
- Canonical `CLA.md` / `COMMERCIAL.md` + `examples/bench_rank_audit.rs`
  (identical-N rank audit bench used in paper 3) added to the release repo.

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
- Watch-config study (MAC-SIDE SIMULATION — physical watch bench pending):
  512-bit hash truncation scores 87.8% strict, HIGHER than full 1024-bit's
  86.1% (later bits add quantization noise). 64B/doc + 8B sidecar = 72B/doc
  → 4.97M docs in the 358MB watch budget by arithmetic; cluster-aware 100.0%
  (independent clusters, 0 FOREIGN) at the coarser code, n=1000, K=200.
  NOT yet validated on physical Apple Watch hardware — the pending YPWatch
  build (full-corpus payload + Memory Gate N_run trim + bench reporting)
  must confirm on-device. Only physical watch numbers to date remain the
  2026-09-21 legacy benches (3.03M records, Tier C 100%, p50 27ms).
- Data note: 10,019 DB rows added after the 2026-09-29 index build —
  incremental ingest pass needed.
### Research — algorithm series (2026-10-01, afternoon)
- **Two duplicate-cluster algorithms specified** (`docs/algorithms.md` in the
  working repo): BDC (batch construction — exists, validated) and IPCM
  (incremental purity-preserving maintenance — designed + implemented same day).
- **IPCM proven on a live ingest**: 13,998/13,998 exact agreement with a full
  batch rebuild; 0 deviations. Invariant: big-cluster bridge merges deferred
  (purity preservation); reversible audit log.
- **FOLD (arXiv:2606.03001) read and positioned**: inverse problem (ANN for
  dedup vs dedup for retrieval); their tie observation corroborates our
  mechanism; no purity-preserving merge algorithm exists there — that delta
  remains ours. FOLD's bitmap signatures noted as a future fuzzy-channel
  component, with credit.


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

## 2026-10-01 — canonical representative selection + sidecar system (working repo 40854885)
- Display policy for duplicate clusters: most complete text wins, tie -> lowest
  index. 735,186/5,660,333 docs now show a better copy; verified 5/5 on live
  clusters via VineReranker.canonical_of(). Corpus ids carry no version
  suffixes (stripped at source) — policy documented in sidecar meta.
- Independent BDC sidecar @yp-corpus-20261001-3e9a2336: 4,925,147 clusters
  (base-id + identical-text union-find, never the hash).
- Max exam reports strict AND cluster-aware R@1/3/10, corpus-stamped.
- Incident (resolved): index_10m_hashes.bin regenerated in wrong hash space;
  production graph restored from backup; file ownership now exclusive to
  build_index_10m.py. Standing rules in master memory.

### Added — paper 3 release artifacts (2026-10-03)
- `paper/YP_PAPER3_v1.0.2.pdf` — "Yellow Phoenix: Semantic Search on an Apple Watch — 86.5% Top-10 Fidelity at 3.9M Documents, Bit-Identical from watchOS to iOS to macOS and Linux" (author-review candidate; not on arXiv until the JOSS clock matures ~2027-03-29)
- `docs/paper3_draft.md` — paper source
- `examples/bench_{identical_1m,flat_identical,rerank_sweep}.rs` — the four-platform identical-N benches (graph + flat scan + K-sweep; certified K=200 reproduces 94.0)
- `scripts/{kill_test_ce_gt,judge_bge_gt,judge_qwen3_gt}.py` — the three-judge kill-test panel + reconciliation instrumentation (pre-registered criterion; fired on strongest judge; conditioned fidelity 87.8/84.9/79.6)
- `results/benchmark_results/` — device-run reports + reconciliation record
- Measurement record anchored in the hash-witnessed chain (tip: Entry 786, re-witnessed 2026-10-03); Zenodo deposit follows.
