# YP 10M Master Plan

**Created:** 2026-07-31  
**Principle:** Additive only. Nothing deleted. Nothing rewired. Feature-flag everything.

## Phases

1. **Multi-Probe ITQ** (`scripts/scale_engine_pid.py`)
   - Each vector gets 3 hashes inserted (primary + 2 perturbed).
   - Increases Hamming recall without touching Rust HNSW code.

2. **Shard Router** (`scripts/shard_router.py`)
   - 10 shards × ~1M vectors.
   - Routes queries to top-2 shards by 64-bit hash-prefix distance.

3. **Geometric Routing Gate** (`scripts/routing_gate.py`)
   - Fast path vs deep path decision using direct-hash, spectral, and cascade signals.
   - Stubs in `ScaleEnginePID` to be wired to existing FFI modules.

4. **PQ Re-Rank** (`scripts/pq_reranker.py`)
   - 8-byte PQ codes per vector.
   - Asymmetric distance re-rank for deep-path candidates.

5. **10M Build + Burn-In** (`scripts/benchmark_10m.py`)
   - Ties all phases together and tunes gate thresholds.

## Validation

Run the per-phase gates in `scripts/benchmark_10m.py` comments.

## Files Added

- `scripts/shard_router.py`
- `scripts/routing_gate.py`
- `scripts/pq_reranker.py`
- `scripts/benchmark_10m.py`
- `docs/YP_10M_Master_Plan_2026-07-31.md`

## Files Modified

- `scripts/scale_engine_pid.py` — `_multi_probe_itq`, `_build_hybrid_mesh384` multi-probe support, routing stubs.
