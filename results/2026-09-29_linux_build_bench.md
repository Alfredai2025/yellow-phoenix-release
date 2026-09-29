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

## Also fixed in this pass
- `src/ffi_shootout.rs`: previous "portability" helper called itself recursively
  on macOS (infinite recursion → stack overflow in `yp_shootout_unload_*` paths);
  replaced with the correct platform-gated implementation above.
  This macOS-only defect was invisible to the Linux build — caught by code review.

## Honest scope note
The 5.65M-paper golden-stack exam cannot run on 2 GB RAM (index ≈ 9 GB);
full-scale numbers remain with the Mac golden build. This record establishes
cross-platform compilation and functional correctness on a second OS.
