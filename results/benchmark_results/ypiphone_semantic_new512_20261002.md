# YP-Bench iPhone 17 Pro Max — NEW 512 SEMANTIC (4.9M fast + 3.9M scan)
Date: 2026-10-02 19:12-13 · Device: iPhone 17 Pro Max (iPhone18,2), iOS 18.7.8
Corpus: yp-corpus-20261001-3e9a2336 · hash index_10m_hashes.bin ITQ-512 (2026-10-01)
## 4.9M FAST (graph_49m_new512_v5, m12/efC200 v5)
Memory gate OK (need ~855MB, avail 3525MB) · graph loaded 1.32s
HNSW exclude-self (853 semantic queries, E-space gt):
  ef50  R@1 14.9%  R@10 81.5%  p50 4274us   <- cold (first run pays flash page-ins)
  ef100 R@1 16.2%  R@10 81.9%  p50 1329us
  ef200 R@1 16.4%  R@10 82.6%  p50 831us
  ef400 R@1 15.8%  R@10 82.4%  p50 543us    <- warm steady state ~0.5-0.8ms
## 3.9M SCAN (Tier C brute force, 695 queries)
FLAT exclude-self: R@10 86.5% · p50 7472us (~1.9ns/record — 4.7x faster per doc than watch)
Memory: clean exit both runs (~3.5GB avail).
NOTES: (1) ladder p50 inverted because ef=50 runs first on cold mmap; warm latency
is the last rows. (2) R@1 here is TRUE strict top-1 (HNSW section computes it
properly) — raw graph top-1 semantic ~16%; the exam gate would lift it. (3) No
delivery problems on iPhone: cable + 1.8GB app installed in 57s; device auto-
registered in provisioning profile on rebuild with destination attached.
Screenshots IMG_9758/9759.

## WARM REPEATS (user iPhone, 19:18) — canonical steady-state
3.9M scan x3: R@10 86.5% all runs (deterministic); p50 7472/7410/7433us (~7.4ms stable)
4.9M FAST warm ladder (853q):
  ef50  R@10 81.5%  p50 90us    <- headline: 4.9M semantic in 90 MICROseconds
  ef100 R@10 81.9%  p50 124us
  ef200 R@10 82.6%  p50 210us
  ef400 R@10 82.4%  p50 374us
(First-ever run cold: ef50 4274us -> warm 90us; ladder ordering correct once warm.
Graph load: 1.32s cold -> 0.16s warm. Runs deterministic on recall.)
Screenshots IMG_9761-9766.

## WIFE'S iPhone 16 (iPhone17,3, iOS 26.2.1) 19:23-24 — CROSS-DEVICE REPRODUCTION
3.9M scan x3: R@10 86.5% (identical to 17 Pro Max), p50 7343/7375/7366us
4.9M FAST: cold ef50 2500us -> warm ef50 79-81us; ef200 172-173us
Recall ladder IDENTICAL to the other iPhone (81.5/81.9/82.6/82.4) — deterministic
yardstick reproduced across two different devices. Screenshots IMG_9734-9739.

## MAC — same device protocol (Rust engine via yp_bridge + set_ef, 853q)
Graph loaded 0.70s. ef50 R@10 81.5% / ef100 81.9% / ef200 82.6% / ef400 82.4%
R@1 14.9/16.2/16.4/15.8 — BIT-IDENTICAL to both iPhones (three-platform
determinism: same engine+graph+yardstick -> same numbers). Latencies carry
Python FFI overhead (594us-3.4ms); recall is the apples-to-apples metric.

## TABLE-1 identical-N (iPhone 16, 22:13, gate build) — 1M graph m8, 175q
R@10: ef50 66.9 / ef100 66.9 / ef200 68.0 / ef400 69.1%; R@1 strict 0.0 (twin pileup
in first-1M slice); p50 warm 86us. +gate: +1.1pt at ef50/100, null after —
INDEPENDENT DEVICE REPRODUCTION of the Mac null-gate finding. Screenshot IMG_9774.

## 🏆 FINAL HEADLINE — 1024-RERANK ON PHYSICAL iPhone 16 (22:42-43), 3 runs
4.9M docs, graph_49m_new512_v5 + sidecar_1k_49m (666MB, 16B-aligned header):
  HNSW+1024 R@10: ef50 94.0 / ef100 94.0 / ef200 94.0 / ef400 94.6% — BIT-IDENTICAL
  to Mac prediction. Raw graph 81.5-82.6% @ 86us warm; rerank adds ~30us.
  Device-side bugs fixed tonight: 13B->16B header alignment (2 crashes),
  payload staging (h1k field), count-read alignment. Screenshots IMG_9778-80.

## WIFE'S iPhone 17 Pro Max — 1024-RERANK REPRODUCTION (22:55-56), 3 runs
HNSW+1024 R@10: 94.0/94.0/94.0/94.6 at ef50-400 — BIT-IDENTICAL to iPhone 16 and Mac.
Three platforms, two phones, six+ runs, zero deviation. Screenshots IMG_9743-45.
