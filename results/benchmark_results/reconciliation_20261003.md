# RECONCILIATION EXPERIMENT — target definition vs structural absence — 2026-10-03
Question (from GLM review F2): payload targets are pre-fix ARBITRARY top-10 members; §4.1 claims rank-1;
is the 86.5% headline deflated/inflated, and why is scan 86.5 vs ceilings 96.9-98?

METHOD: dual-target benches (bench_identical_1m/bench_rerank_sweep patched for member+gt[0] ids);
3.9M flat hash top-10 from judge detail npz; presence = target corpus position < prefix.
Panel check: RECONCILE_*.md (3 SiliconFlow models): Finding 1 SOUND (report both conditionings),
Finding 2 underpowered (n=32), Finding 3 must be demonstrated per judge (DONE below).

## 3.9M flat scan (n=695)
| target | presence in slice | containment all | containment conditioned |
|---|---|---|---|
| payload member | 100% | 86.0% | 86.0% (n=695) |
| MiniLM gt[0] | 87.6% | 77.0% | 87.8% (n=609) |
| BGE judge pick | 90.4% | 76.7% | 84.9% (n=628) |
| Qwen3 judge pick | 91.1% | 72.5% | 79.6% (n=633) |

## 1M HNSW (n=175; gt[0] present only 32/175 = 18.3%)
- member 66.9-69.1% (ef50-400) | gt0 all 11.4% FLAT across ef (structural absence, not exploration)
- conditioned (n=32): member = gt0 = 62.5% [CI wide; underpowered, secondary evidence only]

## 4.9M graph (n=853)
- raw ef50: member 81.5% | gt0 77.4% | +1024 rerank K200: member 94.0% | gt0 87.3% (ef400: 94.6/87.9; K1000: 95.8/89.0)

## SELF-OCCUPANCY (verified)
Query's own record: distance 0, rank 1 (7/8 sampled; rest tie-ordered among distance-0 group).
Device top-10 = self + 9 candidate slots. All S@10 numbers include self. EXPLAINS strict R@1 = 0.0%
(previously attributed to twin pileup). Must be disclosed in paper protocol.

## CONCLUSION (panel-verified)
- The binary index is ~86-88% faithful to MiniLM-reference targets that EXIST in the slice; the
  raw 86-vs-77 gap was structural absence (old survivorship filter used old target, not gt[0]).
- Conditioned judge fidelity: MiniLM 87.8 / BGE 84.9 / Qwen3 79.6 — residual spread = genuine
  model disagreement (index compresses MiniLM's structure; Qwen3's neighborhood is hardest).
- Kill-test reinterpretation: the "dissent" was ~half structural absence, half genuine; the
  re-scope to index fidelity STANDS, now with per-judge conditioned numbers as the honest table.
- REPORT BOTH CONDITIONINGS (collider caveat from panel): own-presence (member 86.0 n=695) AND
  common-subset (87.8=87.8 n=609).
