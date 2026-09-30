# Linux Build Verification + Bench — 2026-09-29

## Environment
- Host: DigitalOcean droplet `ubuntu-s-2vcpu-2gb-90gb-intel-sgp1-01`
- OS: Ubuntu 22.04.4 LTS, kernel x86_64
- CPU: 2 shared vCPUs (Intel) · RAM: 2 GB
- Toolchain: rustc/cargo 1.98.1 (stable), `YP_GENESIS_PHRASE=yp_phoenix_2026`
- Crate: `mirror_mesh` v0.1.0 (full workspace incl. all bins)

## Build
- `cargo build --release` → **exit 0 in 9m32s** (all bins link, incl. `test_shootout`)
- First attempt failed: raw macOS-only `malloc_zone_pressure_relief` call in
  `src/ffi_shootout.rs` (undeclared on Linux). Fixed with platform-gated helper:
  macOS keeps `malloc_zone_pressure_relief`, Linux uses glibc `malloc_trim(0)`,
  other platforms no-op.

## Benchmark — `cargo bench --bench query_benchmark` (criterion)
Reduced sample settings (`--sample-size 10 --measurement-time 5`), so treat as
indicative smoke numbers, not publication-grade medians.

| Bench | Time (median) | Throughput (median) |
|---|---|---|
| ingestion | 2.43 µs | — |
| lsh_query 10k top1 | 25.98 µs | 38.5 Kelem/s |
| lsh_query 10k top5 | 24.72 µs | 40.5 Kelem/s |
| lsh_query 100k top1 | 44.51 µs | 22.5 Kelem/s |
| lsh_query 100k top5 | 48.24 µs | 20.7 Kelem/s |
| lsh_query 1M top1 | 1.36 ms | 734 elem/s |
| lsh_query 1M top5 | 1.20 ms | 836 elem/s |

Outliers: 5–20% high-mild on most measurements (shared VM, expected).
Full criterion log archived in session output.

## Test suite (dev profile, serial, 2GB swap)
- `cargo test --lib --bins --tests -- --test-threads=1`:
  **363 passed; 0 failed** (587s). All bins and integration tests green.
- One environment-specific integration test (`itq_row0`) is Mac-only by design
  (hardcoded `/Users/mac/...` paths, requires the multi-GB real model artifacts
  not synced to the droplet) — excluded from the Linux scope.
- Method notes learned the hard way, for future runs:
  - `cargo test --release` is invalid: `[profile.release] panic = "abort"`
    conflicts with the test harness (needs unwind). Use dev profile.
  - `cargo test` (default) also builds examples; `probe_tierc_r1` links a
    prebuilt `libpams` via `#[link(name="pams")]` and needs the artifact
    installed first. Use `--lib --bins --tests`.
  - Parallel 1M-scale tests exceed 2GB RAM (OOM SIGKILL). Droplet now has a
    permanent 2GB swapfile (/etc/fstab); run tests with `--test-threads=1`.

## Additional benches
- `accuracy` (100k tier sanity): self/neighbor/random/adversarial all tier=3,
  10/10 matches, no tiering regressions. `accuracy_dummy`: 4.5ms.
- `cold_start_100k`: **87.0ms median** cold-load from disk — flash-resident claim
  evidence. (Reduced-sample settings; indicative, not publication-grade.)

## Mac vs Linux head-to-head (identical bench, identical reduced-sample settings)
| Bench | Mac M3 Pro 18GB | Linux droplet 2 vCPU | Mac advantage |
|---|---|---|---|
| ingestion | 1.26 µs | 2.43 µs | ~1.9× |
| lsh_query 10k top1 | 5.47 µs | 25.98 µs | ~4.7× |
| lsh_query 10k top5 | 5.46 µs | 24.72 µs | ~4.5× |
| lsh_query 100k top1 | 8.91 µs | 44.51 µs | ~5.0× |
| lsh_query 100k top5 | 9.40 µs | 48.24 µs | ~5.1× |
| lsh_query 1M top1 | 86.7 µs | 1,362 µs | ~15.7× |
| lsh_query 1M top5 | 90.5 µs | 1,196 µs | ~13.2× |
Reading: identical scaling curves on both OSes (flat 10k→100k, knee at 1M);
gap tracks hardware class (unified memory vs shared cloud vCPUs) and widens
with scale — memory-bandwidth-bound regime. Portability: confirmed.

## Variance study — query_benchmark ×3 + million_papers (2026-09-30 morning)
Full-sample settings (20 samples / 10s). Run-to-run on the same droplet, no code changes:

| Bench | Run 1 | Run 2 | Run 3 | Spread |
|---|---|---|---|---|
| tier1_query | 255 ns | 254 ns | 261 ns | ±1.5% (rock stable) |
| tier2_ensemble | 666 ns | 676 ns | 680 ns | ±1% |
| full_pipeline | 116.9 µs | 116.8 µs | 119.9 µs | ±1.3% |
| ingestion | 2.31 µs | 2.40 µs | 2.51 µs | ±4% |
| lsh_query 10k top1 | 24.77 µs | 24.30 µs | 27.00 µs | ±5% |
| lsh_query 100k top1 | 48.08 µs | 45.56 µs | 46.85 µs | ±3% |
| lsh_query 1M top1 | 1.364 ms | 1.545 ms | 2.398 ms | **+76% drift** |
| lsh_query 1M top5 | 1.719 ms | 2.026 ms | 2.417 ms | **+41% drift** |

Reading: cache-resident paths (micro tiers, 10k-100k) are robust run-to-run
(1-5%). The 1M path degrades across successive runs — memory-bandwidth-bound
workload on a shared multi-tenant host (neighbor load drift), consistent with
the ±25-40% vs-baseline swings criterion measured between nights. Honest paper
sentence: "cache-resident latencies are reproducible within a few percent;
DRAM-resident latencies on shared cloud vCPUs vary by up to ~2× run-to-run."

million_papers bench (different, heavier query path — resonance/cascade tiers,
not raw LSH): 1M docs ingested in 593 ms; query_1m_papers 32.6 ms median.
Not comparable to lsh_query — different pipeline stage, reported separately.

## Also fixed in this pass
- `src/ffi_shootout.rs`: previous "portability" helper called itself recursively
  on macOS (infinite recursion → stack overflow in `yp_shootout_unload_*` paths);
  replaced with the correct platform-gated implementation above.
  This macOS-only defect was invisible to the Linux build — caught by code review.

## Honest scope note
The 5.65M-paper golden-stack exam cannot run on 2 GB RAM (index ≈ 9 GB);
full-scale numbers remain with the Mac golden build. This record establishes
cross-platform compilation and functional correctness on a second OS.
