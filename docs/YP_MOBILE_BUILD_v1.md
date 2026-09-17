# YP MOBILE PRODUCT BUILD — DUAL-TIER iPHONE + WATCH PACK + MAC
## v1 — based on pre-registered measurements (Entries 61–64, 2026-09-16)

## 0. MEASURED FACTS THIS BUILD RELIES ON (do not re-derive)

| Fact | Value | Source |
|---|---|---|
| Float16 re-rank quality | R@10 0.906 @ 768 B/doc | Entry 64 matched-bytes |
| PQ @ 192 B | R@10 0.872, R@1 ≈ 1.0 | Entry 64 |
| PQ @ 64 B | R@10 0.714 | Entry 64 |
| Factors @ any budget | loses to PQ where compression matters | Entry 64 (museum'd) |
| ISM filter | 0.375μs, 48 B/doc, O(1) | bench guide, verified |
| Cascade Cache | 0.31μs warm, semantic, lid-aware | production |
| iPhone memory wall | 3.44 GB per-process kill | arXiv paper §3 |
| pread vs mmap | float store must use pread (ENOMEM on map) | arXiv paper §6.1 |
| macOS wall | ~7.1GB (iOS mmap lesson is iOS-specific) | Entry 708 notes |

Kill chain 61/62/63/64 closed. Float line closed. Custom encoder (QAT/IB) =
separate pre-registered program, NOT part of this build.

## 1. OPERATING RULES

1. Snapshot first: `git tag pre-mobile-v1-$(date +%Y%m%d-%H%M%S)`; backup `data/`; backup `yp_bridge.py` if touched.
2. Exam quarantine: frozen exam (now `data/frozen_exam_hybrid5m_20260916/`, rescued
   from /tmp 2026-09-17; was /tmp/hybrid5m) + frozen 106k droplets = validation-only. Never training input.
3. `git add` named files only. Never `-A`. `.git` is 16GB.
4. Manual/foreground runs only. No launchd, no unattended caffeinate (thermal rule).
5. Debug protocol: OBSERVE → evidence → PASS/FAIL gate per phase. No symptom-patching.
6. New code under `yp_mobile/`. No edits to engine hot path until Phase 3 and only behind flags.

## 2. PHASE 1 — iPHONE DUAL/Triple-TIER OPTION (ship first)

One app, one active dataset at a time, user picks tier at onboarding; switchable
in Settings with re-download. Build order rationale: iPhone dual-tier is the product.

| Tier | Re-rank data | Total | R@10 | Copy |
|---|---|---|---|---|
| Full | float16 store | ~5GB | 0.906 | "Best possible ordering" |
| Compact | PQ@192 | ~2.5GB | 0.872 | "Same top answer; list tail may shuffle" |
| Lite | PQ@64 | ~1.7GB | 0.714 | "Top answer still right; lower half reorders" |

Build steps:
1. `yp_mobile/build_tiers.py` (one script, three outputs): OPQ/PCA-rotate →
   m subq x 2-dim (Compact m=192; Lite m=64) x 256-centroid codebooks. Train on
   train embeddings only; assert exam-id disjointness. Each output: store bin +
   codebooks.npz + meta.json (sha1 of inputs, git hash, Entry-64 numbers for
   cross-check).
2. Sanity: decode sample docs, cosine vs float16 originals, assert within Entry-64 envelope.
3. Search path `--tier {full|compact|lite}` swaps only the re-rank reader
   (float16 pread vs PQ decode). ISM→HNSW→re-rank pipeline identical. Default = Compact.
4. Switch mechanics: onboarding picker (three cards, measured numbers as copy);
   Settings → Storage → switch = download new, verify sha1, delete old (never both resident).
5. Gates (per tier): frozen-exam delta within Entry-64 envelope; R@1 delta ≈ 0;
   100-query device soak under 3.44GB wall, no kill; median latency within 10% of Full.

PASS: all three tiers pass → commit, tag, proof-of-life entry.

## 3. PHASE 3 — MAC APP (the "pro" tier; build second — mostly a shell)

Mac has no 3.44GB wall — no-compromise edition and the dev/reference build.
- Corpus: full 5.1M default; headroom toward 20M+ (PQ tiers scale; ISM validated to 200M).
- Storage: default Full (mmap fine on macOS); Compact/Lite in Settings for small installs.
- Features: batch search, result export, keyboard-first UI, corpus builder tools
  (builds watch packs / phone tiers), benchmark mode (harness lives here).
- Gates: frozen-exam envelope; bench harness regression-clean.

## 4. PHASE 2 — APPLE WATCH PACK ("your library on your wrist"; build last — the demo)

Self-contained pack, NO phone dependency (WatchConnectivity is unreliable).
**100k curated docs max** (favorites/reading list/recent).
- Data: ISM 100k x 48 B = 4.8MB + mini graph (M=12, 384-bit codes) ~20MB + titles ~5MB ≈ 30MB
- Encoder: int8 MiniLM ~25–30MB via CoreML → total ~60MB (limit ~80MB)
- Query: explicit trigger only (crown tap) → speech (WhisperKit watchOS 10+) or scribble
  → encode burst → ISM lookup (0.375μs, negligible energy — the only FLOP-free retrieval
  primitive that fits the watch thermal budget) → top-5 titles
- Reuse cache + thermal governance as-is
- Build: `build_watch_pack.py` on phone → WatchConnectivity transfer with on-watch fallback
- Gates: pack ≤ 80MB, query < 200ms on S9, 50 spot-checks correct, thermal-clean soak

Watch use-case notes (researched): 92% of watch use is health/fitness; the binding
constraint is thermal (short INT8 bursts only), not compute. Candidate products:
voice food logging, medication/clinical reference pack (~50k entries ≈ 2.4MB),
"ask my library" papers demo, notification triage by meaning, glanceable lookups.
Scope discipline: favorites-only; the full corpus never fits and never will.

## 5. PHASE 3-conditional — FLOAT TAIL HYBRID (do NOT build yet)

Only if product demands the last ~3.4pp back on Compact: residual-check at build
time, keep float16 for the ~5% worst-reconstructing docs (+0.2GB → ~1.2GB total).
Pre-register before building; needs an explicit product decision.

## 6. DEFINITION OF DONE (v1)

- iPhone: 3 tiers validated + soak clean
- Watch: pack demo working, thermal-clean, <80MB
- Mac: full-corpus app + tier switch + bench mode
- All artifacts hashed, Entry 64 cross-referenced in every meta.json, commits tagged,
  proof-of-life entries appended
- Marketing lines drawn only from measured numbers

**Build order: Phase 1 → Phase 3 (Mac) → Phase 2 (Watch).**
Paper section "Float Store Floor" runs in parallel with Phase 1 — fully measured already.
