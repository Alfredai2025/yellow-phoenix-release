# TGE / Geometric Brain Session Summary

**Date:** 2026-07-30

## What we tested

| Experiment | Result | Verdict |
|---|---|---|
| PID controller for HNSW `ef` | 24k cumulative latency vs 29k static/bang-bang, 0 violations | ✅ Viable — should ship |
| YPER binary-hash cascade (threshold early stop) | 1.56× speedup, 2.4% R@1 | ❌ Broken (hash ties) |
| YPER margin-based early stop | 2.4× speedup, 42% R@1 | ❌ Poor recall |
| Cl(5) geometric brain, PCA W | 72% grade-2 ground truth, 0% R@1 vs cosine | ❌ No retrieval signal |
| TGE 5-D end-to-end (unconstrained weights) | w2=1.782, 0% R@1, score peaks at 47° | ⚠️ Serendipity engine, not NN retrieval |
| TGE 5-D constrained (w2=0.3·w0) | w2=0.300, loss stuck, 0% R@1 | ❌ 5-D projection too crushed |
| **TGE 32-D PyTorch** | 🔄 Running now | TBD |

## Key insight

For unit vectors, the bivector norm is a deterministic function of the dot product:
`||q∧p|| = sqrt(1 - <q,p>²)` (in 5-D). A score that adds positive scalar and positive bivector terms can peak at an intermediate angle, producing a "related but not identical" ranking rather than nearest-neighbor ranking.

## Files

- `scripts/tge_prototype.py` — 5-D PyTorch TGE (unconstrained)
- `scripts/tge_constrained.py` — 5-D PyTorch TGE (w2 = 0.3·w0)
- `scripts/tge_pytorch_32d.py` — 32-D PyTorch TGE (running)
- `data/tge/tge_phase0.pt` — 5-D unconstrained checkpoint (serendipity ghost)
- `data/tge/tge_constrained.pt` — 5-D constrained checkpoint
- `data/tge/tge_32d.pt` — 32-D checkpoint (will exist after run)
- `docs/TGE_Spec_v1.md` — full spec

## Next decision

- If 32-D TGE gets R@1 > 50%: proceed to Phase 2 (more data / hard negatives).
- If 32-D TGE fails: geometric brain is MUSEUM for retrieval, but the 5-D unconstrained "serendipity ghost" may be useful as a complementary-discovery feature.
