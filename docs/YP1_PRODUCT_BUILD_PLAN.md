# Yellow Phoenix 1.0 — Full Product Build Plan
**Goal:** turn the 5M-on-iPhone benchmark into a world-class offline paper-search product.
**Status:** DRAFT for review — nothing below is final until you approve it.

---

## 1. The Product

An iPhone app holding **5,107,508 real papers (arXiv + PubMed)** fully offline:
semantic search (1.8ms warm), author search, title keyword search, year/category
facets, abstract reading — in ~6GB, under the jetsam ceiling, with online
full-text as an optional extra.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  iPhone (Documents/Yellow Phoenix/)                    RAM budget    │
│                                                                      │
│  SEMANTIC LAYER                                                      │
│  ├─ real_5m.ism          351 MB   512-bit fingerprints (exact scan)  │
│  ├─ real_5m_hnsw.bin   3.16 GB   v5 mmap graph (m=32) ←───┐          │
│  │                            read-only, OS page-cached    │          │
│  LEXICAL LAYER                                                     │
│  ├─ papers_5m.db         1.8 GB   sqlite: meta + FTS5       │          │
│  │   meta(id PK, ext_id, title, authors, year, cat, src)    │          │
│  │   fts(title,authors) contentless detail=none unicode61   │          │
│  │                            ↑ joined to semantic by id ───┘          │
│  TEXT LAYER                                                          │
│  └─ abstracts_5m.bin    ~1.5 GB   zstd-dict blocks, id→abstract      │
│                                                                      │
│  ONLINE-OPTIONAL: ext_id → arxiv.org/abs/…, pubmed URL               │
│                                                                      │
│  Pinned heap ≈ 1.2 GB · Soft page cache ≈ 1.5 GB · Ceiling ≈ 3 GB   │
└──────────────────────────────────────────────────────────────────────┘
```

**Design rule:** every big thing is a separate mmap'd file; layers join on the
ISM row id; nothing new touches the search hot path.

---

## 3. File Formats (specs)

### 3.1 papers_5m.db — metadata + lexical index ✅ BUILT (awaiting validation)
```sql
PRAGMA page_size=16384;                       -- fewer IOPS (research-backed)
PRAGMA mmap_size=268435456;                   -- 256MB, OS-purgeable (app)
PRAGMA cache_size=-65536;                     -- 64MB heap cap (app)

CREATE TABLE meta(
  id INTEGER PRIMARY KEY,        -- ISM/HNSW row id (the join key, 0..5,107,507)
  ext_id TEXT,                   -- "arxiv1m:cs/0301001" / pmid / droplet id
  title TEXT, authors TEXT, year INTEGER, categories TEXT, source TEXT);
CREATE INDEX idx_year ON meta(year);

CREATE VIRTUAL TABLE fts USING fts5(
  title, authors,
  tokenize='unicode61',          -- NO porter: names must not be stemmed
  detail='none',                 -- 5.5× smaller index (sqlite.org measurement)
  content='');                   -- contentless: no text duplication
-- query: SELECT id FROM fts WHERE fts MATCH ?  → join meta for display
```
Built by `scripts/build_papers_5m_db.py` (replay dedup → per-source join).
**Gate:** `scripts/validate_papers_5m_db.py` — 30 random rows re-embedded
(MiniLM, same recipe as original pipeline), ITQ hash must land at the same id.
≥29/30 required or the mapping is wrong and we fix before pushing.

### 3.2 abstracts_5m.bin — offline reading store (TO BUILD)
```
Header: magic "YPA1", version, count (u64), block_size, dict_size
Dictionary: zstd dictionary trained on ~100k sample abstracts (ZDICT_train)
Blocks: N × [u32 uncomp_len][u32 comp_len][zstd frame of ≤64KB of records]
Record stream per block: varint id_delta + varint title_len + title +
                        varint abs_len + abstract   (id deltas → sorted ids)
Offset index: u64[N_blocks] absolute offsets
```
- Decode one abstract: binary search block table (log2 90k ≈ 17 steps) +
  one 64KB zstd decode ≈ 20–50µs.
- mmap'd; RAM = touched pages only (same philosophy as v5).
- Builder streams phoenix db + droplet db in ISM id order.
- **Decision recorded:** text store inside sqlite as blobs was rejected —
  per-row zstd frames inflate sqlite pages and force a zstd dep into SQL
  queries; sidecar binary keeps layers independent (and is the seed of the
  Option-C research artifact).

### 3.3 3M graph v5 (30-second conversion)
`convert_hnsw_v5 data/hnsw_3m.bin data/hnsw_3m_v5.bin` → pushed as
`hnsw_3m.bin`. Completes the benchmark ladder 106K→1M→3M→5M→10M.

### 3.4 Existing files — unchanged
v5 loader (sniffs v4/v5), v4 originals kept on Mac as insert base,
copy script maps `*_v5.bin` → legacy names on phone.

---

## 4. Data Pipeline (Mac)

| Step | Script | Time | Gate |
|---|---|---|---|
| 1. metadata DB | build_papers_5m_db.py | ~2 min | validate_papers_5m_db.py ≥29/30 |
| 2. abstract store | build_abstracts_5m_bin.py | ~15–30 min | spot-decode 1000 abstracts, verify text |
| 3. 3M v5 | convert_hnsw_v5 | 30 s | equivalence probe ≥499/500 |
| 4. push | copy_rebuilt_to_iphone.py (updated FILES) | ~8 min | sha256 round-trip verify |

---

## 5. Rust Engine Changes

### 5.1 Cold-start warm-up (Phase 2)
Problem: first query pays mmap page faults (116ms @5M, 604ms @10M).
```
load_v5():
  ...existing...
  spawn background thread:
    madvise(edge_blob, WILLNEED)          -- kernel prefetch, async
    run W=50 self-hash queries at ef=400  -- touches hot graph pages
    advise_dontneed(cold regions)
```
- Target: first real query <10ms. Cost: +2–4s background CPU after load,
  zero added peak RAM (pages are shared with the cache).
- Flag-gated so benchmarks can measure cold vs warm separately.

STATUS (2026-09-03): DONE. `BinaryHNSW::warm_up(n_queries, ef)` in
`src/binary_hnsw.rs` (madvise WILLNEED + 50 sampled self-hash probes at
ef=400), FFI `yp_hnsw_warm_up` in `src/flat_ffi.rs`, called from
NativeEngine.swift after both `yp_load_index_chunk` sites before
`ready = true`. Mac results (bench_warm_up, MADV_DONTNEED to force cold):
- 5M v5: cold first query 20.7ms → warm 0.24ms; warm-up cost 1.1s
- 10M v5: cold first query 73.4ms → warm 0.62ms; warm-up cost 2.8s
`load()` itself is untouched; shootout/bench cold numbers stay valid.

### 5.2 Not changing
Search algorithm, v5 format, FFI surface (only additions).

---

## 6. App Changes (Phase 3) — YPPhone

### 6.1 Search modes (Search tab)
```
[ search box ]        [ Semantic | Author | Title ]   <- segmented control,
                                                        auto-detect: "author:"
                                                        prefix forces Author
Semantic:  query text → MiniLM → ITQ → HNSW/ISM → ids
Author:    FTS MATCH on authors → ids (join meta for display)
Title:     FTS MATCH on title → ids
All modes: results = [title, authors, year, category] from papers_5m.db
```

### 6.2 Result detail sheet
- Title, authors, year, category, source badge (arXiv/PubMed/Droplet)
- Abstract (decode from abstracts_5m.bin, ~50µs)
- Buttons: [Find Similar] (vector search seeded by this paper's hash — the
  author→semantic bridge), [Full Text] (ext_id → Safari; online, marked as such)

### 6.3 Facets
- Year range slider + category chips, applied to current result set
  (post-filter on top-k for semantic; WHERE-clause for lexical) — <1ms

### 6.4 Files touched
- `NativeEngine.swift`: papers_5m.db handle (mmap pragmas), FTS query fn,
  abstract decode fn, ext_id URL builder, titlesName wiring for real_5m/3M
- `YPViews.swift`: Search tab UI (segmented control, result cells, detail
  sheet, facets) — new `PaperSearchView`, keeps benchmark UI untouched
- `CompleteTestRunner.swift`: no changes (benchmarks stay independent)

### 6.5 UX copy for offline honesty
Results show source + year prominently; "Full Text" button labeled
"Requires internet" when offline (NWPathMonitor).

---

## 7. Benchmarks for the Paper (Phase 4)

| # | Measurement | Why |
|---|---|---|
| 1 | 10M full shootout modes (now possible) | complete ladder |
| 2 | 3M shootout on device | complete ladder |
| 3 | 5M n≥100 (was n=12) | statistical validity |
| 4 | 5M float-GT semantic recall (Mac-side GT from embeddings) | honest quality: ITQ ceiling vs graph quality |
| 5 | Cold vs warm first-query after warm-up fix | the <10ms claim |
| 6 | Full-system run with all product files installed | shipping-config numbers |

Deliverable: updated tables in `arxiv_update/benchmark_reconciliation.md`.

---

## 8. Memory Budget (rechecked after every phase)

| Layer | Pinned | Soft (purgeable) |
|---|---|---|
| App + runtime + MiniLM | 0.3 GB | — |
| HNSW v5 records | 0.58 GB | — |
| ISM (when loaded) | 0.35 GB | — |
| Hot graph pages | — | 1.0–1.5 GB |
| papers_5m.db pages | — | ≤0.32 GB (256MB mmap cap) |
| abstracts pages | — | ≤0.2 GB |
| **Total** | **≈1.2 GB** | **≈2.0 GB worst** |

Rules: mmap everything big; bound sqlite cache; unload ISM↔HNSW between modes.

---

## 9. Build Order & Effort

| Phase | Item | Est. | Depends |
|---|---|---|---|
| 1a | papers_5m.db ✅ built | done | — |
| 1b | validation gate | 5 min | 1a |
| 1c | abstracts_5m.bin | ~2 hrs | — |
| 1d | 3M v5 + push + sha256 | ~30 min | 1b |
| 2 | Rust warm-up + XCFramework | ~half day | — |
| 3a | App: search modes + result cells | ~1.5 days | 1d, 2 |
| 3b | App: detail sheet + abstracts + similar | ~1 day | 3a, 1c |
| 3c | App: facets + full-text button | ~0.5 day | 3b |
| 4 | Benchmark suite re-run + paper tables | ~1 day | 3a (run on final build) |

**Total: ~4–5 working days to product + paper-ready numbers.**

---

## 10. Explicit Non-Goals (this build)

- Citation counts, affiliations, journal names (need new harvests — roadmap)
- Option-C custom succinct engine (research artifact — after product ships;
  abstracts_5m.bin format is designed to migrate into it)
- Sidecar on-device incremental add (Mac rebuild + push is the update path)
- 20M graph, Spotlight, Android, iCloud sync

---

## 11. Risks & Open Questions

| Risk | Mitigation |
|---|---|
| MiniLM validation mismatch | script reports per-row; if <29/30, bisect per source segment |
| papers_5m.db too big at 1.8GB (est was ~1GB) | acceptable now (34GB free); v2: drop FTS to contentless+prefix='2,3' or zstd meta |
| Abstract store bigger than ~1.5GB est | measure at build; fallback: first-500-chars store |
| Warm-up RAM regression on device | Phase-4 memory gate re-run; advise_dontneed on cold regions |
| FTS ranking quality on author names | unicode61 tokenizes "Smith, John" fine; test "lastname firstname" permutations in validation |
| droplet rowid-order assumption | validation script samples src4 rows specifically (seed covers all segments) |

---

## 12. Decisions Already Made (recorded)

1. Metadata in sqlite FTS5, **not** in the semantic hash — identity ≠ meaning
2. `detail='none'` contentless FTS — research-backed 5.5× index cut
3. Abstracts as sidecar binary, not sqlite blobs — layer independence
4. ext_id link-out only for full text — offline boundary = abstract
5. v5 stays read-only; updates via Mac rebuild + push
6. Benchmarks never depend on product files — product additions must not
   change 1.77ms (verified by Phase-4 re-run)
