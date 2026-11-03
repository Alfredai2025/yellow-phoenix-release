# Yellow Phoenix — Duel / Measurement Harness

The measurement code behind the "beat Google" duel claim of the Yellow
Phoenix **recall–bytes map** paper (DOI `10.5281/zenodo.23920187`). At
byte-matched widths, a plain per-dimension Lloyd-Max codec — the "plain
recipe" in `codec.py` — out-recalls Google's published **RSLM** reference
implementation (vendored under `vendor/rslm/`, Apache-2.0, unmodified) on
every paired cell of the published map.

Everything here is pure Python (numpy + scikit-learn). No engine, no index
structure, no timing claims: this harness measures **codec recall only**.

## What this is (and is not)

* **Is:** the codec; the two rulers (flat-recall evaluator, engine-ruler
  evaluator); the duel runner against the public RSLM code; the map-cell
  scripts that regenerate every published duel cell; and a fixture of the
  published numbers for self-checking.
* **Is not:** any part of the Yellow Phoenix engine (graph index, SIMD
  kernels, layouts, speed campaign). Those are out of scope for this
  release and appear nowhere in this directory.

**SCOUT / positional framing.** Campaign convention: a *scout* is a
single-run, not-yet-reproduced measurement; nothing here is one. The recall
numbers this harness emits are deterministic functions of (codes, books,
corpus, queries, ground truth) — re-running it reproduces the published
cells exactly (below). Timing columns in the CSV outputs, where present,
are informational only and are scouts by that convention; no claim in the
map paper rests on them. *Positional* framing, from the paper: the map is
terrain, not dots — the value of a cell is the kind of ground it probes
(pool size × bytes × ruler), and a codec's position on both rulers is what
characterizes it.

## The protocol (matches the map paper)

* 1000 queries per cell; ground truth = the bundle's first-10 neighbors
  (Yandex: the distributed top-100, first-10; audited 24/24 exact-match).
* Two rulers per cell:
  * **flat** — recall@10 of code scores alone over the full pool
    ("the field's standard ruler": what a codec-only brute-force scan gets).
  * **w10k** (engine ruler) — recall@10 after an exact re-rank of the
    code-score top-10,000 shortlist against the original vectors
    ("the deployment semantics": what a real engine gets, since engines
    re-rank their shortlists exactly).
* cosine datasets: corpus and queries unit-normalized; code scores are
  dots of unit-normalized reconstructions. L2 datasets (SIFT): scores are
  negated squared L2. Byte widths verified against packed bytes; the duel
  is byte-matched (e.g. our 64B vs RSLM's 64B, not nearest configuration).
* The streaming top-k merge (chunked argpartition with sentinel, worst
  carried across chunks) is the campaign convention, triple-verified
  against brute force; per-chunk score expressions preserve the campaign
  operation order, so cached codes reproduce published cells bit-for-bit.

## Files

| path | what |
|---|---|
| `codec.py` | `PerDimCodec` — per-dimension Lloyd-Max (k-means) scalar quantizer: mean-center, per-dim K=2^b levels, nearest-level codes, decode = books[code]+mu |
| `rulers.py` | `flat_recall`, `engine_ruler`, `dual_ruler` — the two evaluators + the shared streaming merge |
| `duel.py` | duel runner: vendored RSLM codecs under our rulers (`glove` / `sift` / `yandex10m`) |
| `cells/sift_glove_ladder.py` | our-codec ladder, SIFT-1M (L2) + GloVe-1M (cosine), 1..8 bits/dim, both rulers |
| `cells/yandex_10m_ladder.py` | our-codec ladder, Yandex T2I-10M (cosine), 1..8 bits/dim, both rulers |
| `cells/yahoo384_sweep.py` | PCA + per-dim 4-bit ledge sweep at 384 dims (flat ruler) |
| `vendor/rslm/` | Google's RSLM reference codec, **vendored unmodified with its Apache-2.0 LICENSE** (see its README for provenance + fetch/diff instructions) |
| `expected_outputs/published_cells.csv` | the published duel table (ours + vendor rows + Yahoo-384) to self-check against |
| `LICENSE` `CLA.md` `COMMERCIAL.md` | AGPL-3.0-or-later + project CLA/commercial terms (copies of the repo root files) |

## Fetching the public datasets

No data is distributed with this harness. The published cells use:

* **SIFT-1M** and **GloVe-1M**: the ann-benchmarks hdf5 bundles
  (`sift-128-euclidean.hdf5`, `glove-100-angular.hdf5`), public mirrors at
  `http://ann-benchmarks.com/<name>.hdf5`; ground truth included. Used as
  distributed.
* **Yandex T2I-10M** (CC-BY): Yandex Research, "Benchmarks for Billion-Scale
  Similarity Search" (`research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search`).
  Download the 10M base subset, the 100K query set, and the top-100 ground
  truth; lay out as `base_10m.f32` (10,000,000×200 f32 rows), `queries_100k.fbin`
  (uint32 nq, uint32 dim, then nq×200 f32), `gt.npy`.
* **Yahoo-384**: the published numbers used the campaign's 384-d Yahoo
  Answers embedding payload (671,164 vectors, cosine, MiniLM-class encoder)
  — a campaign artifact that is **not redistributed**. The script
  (`cells/yahoo384_sweep.py`) takes any payload/gt/queries of the same
  shape; the protocol is the released artifact for this cell. Its rows in
  the fixture are flat-ruler only and are matched to ~3e-4 (see below).

## Running

```bash
pip install -r requirements.txt

# our-codec side
python cells/sift_glove_ladder.py --data-dir DATA --work-dir WORK --out-dir OUT
python cells/yandex_10m_ladder.py  --data-dir DATA --work-dir WORK --out-dir OUT

# vendor side (the duel opponent)
python duel.py glove     --data-dir DATA --out-dir OUT
python duel.py sift      --data-dir DATA --out-dir OUT
python duel.py yandex10m --data-dir DATA --out-dir OUT --work-dir WORK
```

## Self-check: reproducing a published cell exactly

The published SIFT/GloVe/Yandex cells are reproduced by re-running the
ladder scripts end-to-end from fetched data (training is seeded; ~hours on
a laptop). The cheaper exact check — and the one performed before this
release — evaluates **banked codes** (the campaign's cached
`books`/`codes` arrays) through this harness's `rulers.py`:

> SIFT-1M, 64 bytes (4 bits/dim): flat **0.8776**, w10k **0.9992**
> (RSLM Rslm4Lite at 64B: flat 0.7967 — the published duel margin).
> GloVe-1M, 25 bytes (2 bits/dim): flat **0.5639**, w10k **1.0000**
> (RSLM Rslm2 at 27B: flat 0.5555).

An evaluators-against-banked-codes run must match `expected_outputs/
published_cells.csv` to all four decimals on every row of those ladders.
(Yahoo-384 rows: the campaign scored with torch/MPS; the numpy port can
differ in last-ulp tie order within the paper's documented ±3e-4
boundary-instability band, so those rows are anchors to ~3e-4, not
bit-exact.)

## Licensing

Harness code: **AGPL-3.0-or-later** (see `LICENSE`). `vendor/rslm/` is
Google's code under **Apache-2.0** and keeps its own `LICENSE`; it is
vendored verbatim, with attribution — no notices stripped. Dataset files
keep their own licenses (ann-benchmarks bundles; Yandex T2I CC-BY) and are
fetched by you from the sources above.
