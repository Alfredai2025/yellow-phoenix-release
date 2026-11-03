# Mac Paper Verification Bundle

Verification artifacts for **"Yellow Phoenix: Semantic Search on Apple Silicon at
the Candidate Ceiling"** — Marc John Sawyer, CC BY 4.0, © 2026.

Paper record (Zenodo): **DOI 10.5281/zenodo.23920218**

This bundle lets anyone verify the paper's headline recall numbers **without
the private retrieval engine**. It contains only measurements, embeddings,
ids, and scores — **no corpus text and no engine code**.

## Contents

- `frozen_exam_hybrid5m_20260916/` — the frozen exam (seed 20260916, n=1000
  queries over a 5,107,508-paper corpus): query embeddings (`qemb.npy`),
  query hashes (`qhash.npy`), released candidate ids/scores (`cand_ids.npy`,
  `cand_scores.npy`), exact ground-truth top-100 (`gt_top100_ids.npy`,
  `gt_top100_sims.npy`), sampled doc ids (`pids.npy`), the exam's own
  measured metrics (`metrics.json`), calibration reports, and prep metadata.
  Copied verbatim from the measurement archive; nothing modified.
- `benchmarks/` — machine-readable records behind the paper's tables:
  - `claims_layer.json` — the paper's embedded machine-readable claims layer,
    extracted from the deposited PDF with `tools/extract_claims_layer.py`
    (SHA-256 verified, see below).
  - `ef_sweep_k100_20260919_073535.json`, `ef_r10_ci_p95.json`,
    `r10_confidence_intervals.json` — the ef-ladder behind Table 2.
  - `mac_5m_golden_bench_20260918_154254.json`,
    `mac_10m_bench_20260918_153949.json`,
    `mac_10m_bench_20260919_212637.json`,
    `mac_10m_ef400_warm_20260918_154254.json` — the 5M/10M sweep records.
  - `tier_meta.json` — compact-tier (PQ) construction metadata behind the
    tier table.
  - `rerun_*_20261029.json` — the 2026-10-29/30 same-parameter rebuild and
    frozen-exam re-run records (the paper's provenance note reports these).
- `tools/check_exam.py` — clean-room checker (numpy only, no engine imports).
- `tools/extract_claims_layer.py` — PDF attachment extractor + SHA-256 gate.

## Data sources and privacy

The underlying corpus is assembled from **public scholarly sources**: the
arXiv bulk dataset (export.arxiv.org / arXiv AWS Open Data) and PubMed
(via NCBI). To rebuild a comparable corpus, fetch those public datasets and
embed titles+abstracts with a MiniLM-class 384-dim sentence encoder.

The exam ships **embeddings, ids, and scores only — no document text**. The
corpus vector store (one float16 384-dim row per doc, 3.9 GB for 5.1M docs)
and the corpus hash store (64 bytes/doc) are **not** part of this bundle;
they are read locally, read-only, at check time.

## Running the checker

Requirements: Python 3 with numpy (only numpy). The corpus vector store is
required; the hash store is optional (enables an auxiliary depth-200 number).

```sh
python3 tools/check_exam.py \
    --exam frozen_exam_hybrid5m_20260916 \
    --vectors /path/to/floats_5m.f16 \
    --hashes /path/to/real_5m.ism \
    --claims benchmarks/claims_layer.json
```

`floats_5m.f16` layout: 32-byte header, then 5,107,508 rows × 384 float16
(little-endian), row index = document id. `real_5m.ism` layout: 8-byte
little-endian doc count, then 5,107,508 rows × 64 bytes of packed 512-bit
hash, row index = document id.

## Expected output

Mac paper Table 1 (candidate-depth effect, exact re-rank, n=1000):

| candidate depth | published R@10 | check result (2026-11-03, Apple Silicon) |
|---|---|---|
| 50  | 0.9049 | **reproduced exactly** |
| 100 | 0.9538 | **reproduced exactly** |
| 200 | 0.9802 | **not reproducible from released artifacts** |

The depth-200 row is disclosed in the paper itself: the released exam ships
only 100 candidate ids per query, and the depth-200 candidate set came from a
graph index (ef=400) that is not part of this bundle. The checker therefore
reports FAIL for that row rather than pretending. With `--hashes`, the checker
additionally prints an **auxiliary scout** number — exact brute-force Hamming
top-200 from the public hash store, re-ranked identically — which reads
R@10 = 0.9804 (ceiling 0.9806). It is close to, but not equal to, the
published 0.9802, and is labeled in-line as a different protocol.

## Verifying the claims layer

The deposited PDF embeds `claims_layer.json` as a PDF attachment. To extract
and validate it (SHA-256 must equal
`f272acd3d34d043adff96c53ba3de2c4705a6ce1edcaa9b817443944b4710364`):

```sh
python3 tools/extract_claims_layer.py yellow_phoenix_mac_paper.pdf \
    --out claims_layer.json
```

## License

CC BY 4.0, © 2026 Marc John Sawyer. Cite the paper DOI
10.5281/zenodo.23920218 when reusing these artifacts.
