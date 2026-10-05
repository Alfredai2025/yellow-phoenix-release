# ANN-benchmarks Docker validation (2026-10-05, colima aarch64 VM on M3 Pro)

Dress rehearsal for the erikbern/ann-benchmarks PR: both algorithm images built
from PUBLIC ingredients only (GitHub clone of this repo with the designated
public genesis phrase inside the image) and validated in-container with the
harness's own metric (recall@10 vs exact GT, presence-conditioned on the subset —
a subset index cannot contain neighbors outside the subset).

## Engine 1 — binary/hybrid (docker_ctx/Dockerfile)

- Build: ubuntu:22.04 + rustup 1.98.1, cargo release of `build_hnsw_from_flat_ism`
  + `yp_ann_server` from the public clone. All layers reproducible.
- Validation: N_SUB=200k of sift-128-euclidean, 500 queries, ef=128 K=200.
- Result: fit=578s, presence=0.2446, raw=0.2410, **CONDITIONAL=0.9870**
  (assertion > 0.90; matches the 2026-10-04 container-ARM run at 0.985).
- `DOCKER VALIDATION PASSED`

## Engine 2 — int8 residual (docker_ctx/Dockerfile.int8)

- Build: two-stage rust:1.98-slim engine + python:3.11-slim runtime; cargo release
  of `build_int8_graph` + `int8_server` from the public clone. Module mounted at
  runtime per harness convention.
- Validation: N_SUB=100k (the 200k size OOM'd the 2GB droplet on 2026-09-29;
  100k is the known-good size), 500 queries, ef=128 K=200, --memory=3g.
- Result: fit=174s (codec 60s + graph 114s), presence=0.1424,
  **CONDITIONAL=0.9939** (assertion > 0.90).
- `INT8 DOCKER VALIDATION PASSED`

## Notes

- 6.5ms/q in the smoke run is the capped 4-CPU/3g VM, not a leaderboard number;
  1M M3 Pro single-thread measurement for this config: 0.9919 @ ~940 QPS.
- Docker Hub was unreachable from the VM at test time (transient); base images
  pulled via docker.m.daocloud.io mirror and tagged locally. The PR image builds
  on the harness runner (GitHub Actions, US network) where this does not apply.
- h5py is provided by the harness base image; the int8 runtime image correctly
  needs none (the dry-run script installed it at runtime for validation only).
