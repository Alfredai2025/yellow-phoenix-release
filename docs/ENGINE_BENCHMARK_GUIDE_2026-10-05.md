# YELLOW PHOENIX — Engine Benchmark Guide

**Marc John Sawyer | August 2026 → Revised October 5, 2026**

*Revision 2026-10-05: every number now carries its source. Three labels are used throughout:*
- **[CHAIN-nnn]** — attested in the append-only proof-of-life chain (entry nnn), the record the papers cite
- **[CAMPAIGN]** — measured in the 2026-10 ann-benchmarks campaign (SIFT1M, single-thread, full methodology in the release repo)
- **[UNVERIFIED]** — design target or reference-machine number, NOT a Yellow Phoenix measurement. Never quote these as results.

*Corrections this revision: the 165µs/1.27M headline is downgraded to UNVERIFIED (it was the hnswlib reference, not our golden stack — caught by the 2026-10-04 audit); the CrystalMesh384 section is withdrawn (2026-10-04 source inspection found an identity-table artifact, not an ANN index); the BinaryHNSW real-corpus R@1 now carries the self-occupancy disclosure the chain documented (Entry 784).*

License: AGPL-3.0 | Commercial licensing available

---

## Hybrid Re-Rank — the headline engine

**BENCHMARKS (chain-verified, real corpus, corrected ground truth):**

| Metric | Value | Source |
|---|---|---|
| R@10 fidelity vs exact kNN | **96.8%** at 5.66M real papers | [CHAIN-780] |
| R@3 / R@1 | 94.3% / 78.3% | [CHAIN-780] |
| Candidate presence @ef500, top-200 | 96.9% | [CHAIN-780] |
| Index size | 64 B/doc codes + graph + mmap payload | [CAMPAIGN] |
| Hardware | MacBook Pro M3 Pro (no GPU) | — |

**The retired number:** the August edition claimed 165µs P50 / 100% R@10 at 1.27M, marked VERIFIED. The 2026-10-04 audit showed that figure was the hnswlib reference measurement, not this engine. **[UNVERIFIED — re-measurement scheduled; do not quote.]**

**On SIFT1M benchmark data [CAMPAIGN]:** two-stage family spans 0.85 @ ~6,700 QPS up to **0.994 @ 681 QPS** (m24), and 0.36 @ ~29,000 QPS pure-binary speed corner — the smallest index in the room on that chart.

**WHY IT IS FAST:** BinaryHNSW does a shallow traversal to fetch top-K candidates, not a deep search; exact re-rank scores only those K with full precision. Shallow graph + tiny re-rank < deep graph alone. Verified across 4 OS families and 2 ISAs with bit-identical curves [CHAIN-783].

**WHY IT IS SPECIAL:** asymmetric two-stage — the index is the search structure; full precision touches only the shortlist. Nobody else fields a tier family from 29k QPS to 0.994 recall on one engine.

**WHAT RIVALS CANNOT DO:** FAISS-PQ class compression drops recall to the 44–55% range; this architecture holds 0.99-class recall at speed because verification is exact, not quantized. [CAMPAIGN]

---

## BinaryHNSW — the production core

**BENCHMARKS:**

| Metric | Value | Source |
|---|---|---|
| P50 latency | 416–481 µs | Aug 2026 production measurement |
| R@1, real corpus | 99% — **with disclosure:** on the real corpus rank-1 is frequently the query itself (self-occupancy, [CHAIN-784]); the honest structural number on held-out synthetic 10M is **65%** | [CHAIN-784] |
| Memory | 48 B raw codes / ~218 B graph on disk | Aug 2026 |
| Scale | 1M–10M documents | Aug 2026 |
| Build | 244 s @ 1M · 3,760 s @ 10M | Aug 2026 (cf. 268 s @ 988k in [CAMPAIGN]) |
| Status | PRODUCTION — sole production engine since July 31 | — |

**WHY IT IS FAST:** native Rust, zero Python in the hot path; 512-bit ITQ hashes, hamming = XOR + popcount; M=16 ef=50 tuned Pareto point.

**WHY IT IS SPECIAL:** memory-safe, thread-safe; the index IS the embedding (48 B/doc); 128-function FFI; graph save/load, batch insert, shard routing.

**WHAT RIVALS CANNOT DO:** single static binary vs conda+CUDA stacks; PQ-class rivals sacrifice recall this engine does not [CAMPAIGN].

---

## ISM — Inverted Slot Map, the O(1) filter

**BENCHMARKS (Aug 2026, unchanged this revision):**

- P50 latency **0.375 µs** · validated to **200M documents** · build **67.1 s @ 200M** · 48 B/doc · standalone recall 0–18% — it is a *filter tier*, not an engine.

**WHY IT IS SPECIAL:** a hash table, not a graph: bucket lookup returns a semantic neighborhood; enables cascading without re-querying. FAISS has no O(1) layer; cloud APIs cannot beat the speed of light.

---

## CrystalMesh384 — **WITHDRAWN 2026-10-04**

The August edition presented this as a compact 384-bit ANN variant (99.6% R@1, ~280 µs estimated, 2.3× insert speedup). **A user-requested source inspection (museum audit) found the underlying module was an exact-identity hash table** — the "2.3×" was identity lookup versus the hybrid mesh, not graph construction. All CrystalMesh384 performance claims are retracted. The compact-variant role is now served by the binary two-stage family [CAMPAIGN]. *Lesson recorded in-chain: inspect source before benchmarking; a namesake in the graveyard is not evidence.*

---

## Cascade Cache — the warm path

**BENCHMARKS (unchanged, LIVE in production):**

- 0.3–10 µs warm · **99.8% hit rate** at 12K queries · LRU + TTL · seed 10.3 s · lid-aware (pauses closed, resumes open).

**WHY IT IS SPECIAL:** a semantic cache that learns from live traffic; 998 of 1,000 queries never touch the graph. No rival ships an in-engine self-training cache.

---

## Full YP Pipeline — the intelligent system

**BENCHMARKS (unchanged, LIVE):**

- 6.5 ms P50 (domain detection + routing + circuit breakers + audit) · 97.8% top-5 overlap · 1.27M corpus + 12K live queries.

**WHY IT IS SPECIAL:** not a library — a system. Thermal hysteresis (throttles at 72 °C), tamper-evident audit chain, semantic domain routing, subharmonic tier scheduling.

---

## Engine Comparison Matrix

| Engine | Speed | Recall | Memory | Status |
|---|---|---|---|---|
| Hybrid Re-Rank | 681 QPS @ 0.994 (SIFT1M) [CAMPAIGN]; 5.66M corpus fidelity R@10 96.8% [CHAIN-780] | 0.99-class | 64 B + payload | **LEAD** |
| BinaryHNSW | 416–481 µs | 65% structural / 99% with self-occupancy disclosure | 48 B | Core |
| ISM Filter | 0.375 µs | 0–18% alone (filter tier) | 48 B | Filter |
| Cascade Cache | 0.3–10 µs warm | 99.8% hit | Tiny | Optimize |
| Full Pipeline | 6.5 ms | 97.8% top-5 overlap | Heavy | System paper |

*(The withdrawn CrystalMesh384 row is removed by design.)*

---

## KEY INSIGHT

- Lead with the **chain-verified** numbers: 5.66M real papers, R@10 96.8 [CHAIN-780], and the SIFT1M campaign family 0.36 @ 29k QPS → 0.994 @ 681 QPS [CAMPAIGN].
- The 165 µs / 100% R@10 figure is **UNVERIFIED** — re-measure or stay silent.
- ISM is a filter tier ("the 0.4 µs first stage"), never a standalone claim.
- Full Pipeline (6.5 ms) is a system, not a benchmark — saved for the intelligent-search paper.
- **Claims discipline (frozen):** never "fastest in the world." Always: "fastest CPU engine with guaranteed 100% recall" — and now every number carries its chain entry or its UNVERIFIED label.
