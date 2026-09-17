# M3.8 Validation Certificate

**Date:** 2026-07-26  
**Commit:** 2990419f  
**Status:** ✅ TARGET MET

## Results (10,000 held-out title queries)

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| R@1 | **99.57%** | ≥ 97% | ✅ PASS |
| R@5 | **99.57%** | — | — |
| R@10 | **99.57%** | — | — |
| P50 latency | **3.29 µs** | — | — |
| P99 latency | **8.75 µs** | — | — |

## Cascade Configuration

| Layer | Speed | Role |
|-------|-------|------|
| L0: Exact title hash | 4 µs | Instant exact match |
| L1: Prefix map (20-80 chars) | 4 µs | Partial title recovery |
| L2: Keyword geometric mesh | 1.2 ms | Word-deletion recovery (69% R@1) |
| L3: TF-IDF confidence gate | 0.8 ms | Skips MiniLM when >0.95 |
| L4: MiniLM semantic | 5 ms | Final fallback |

## Sign-off

Engine validated for production deployment.  
Next: Scale validation (40M-100M), ArXiv submission, Phoenix Bench release.
