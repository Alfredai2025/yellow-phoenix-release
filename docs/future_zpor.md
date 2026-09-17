# ZPOR: Zero-Point Observer Retrieval (Future Work)

## Idea

Use the 64-dimensional spectral basis to warm-start HNSW graph search.
Instead of a random or fixed entry point, teleport the greedy walk to the
spectrally-nearest top-layer node.

## Pilot Result

On 100K papers, spectral landmarks are **98.2% closer** to the true nearest
neighbor than random landmarks:

- Random landmark → true NN cosine distance: 0.0415
- Spectral landmark → true NN cosine distance: 0.0167

## Practical Recall Test

Adding the spectral landmark to the HNSW candidate pool (ef=25) produces
no recall gain:

- Baseline HNSW R@10: **99.83%**
- ZPOR (landmark injected) R@10: **99.83%**
- Delta: **0.00 pp**

## Why It Didn't Translate

HNSW with `ef=25` already finds the true neighbors in its candidate pool on
100K vectors. The spectral landmark is redundant because HNSW already explored
that neighborhood during the greedy walk.

## Why It Might Still Matter

The pilot distance result suggests spectral coordinates contain useful
long-range structure. A real ZPOR implementation would need to bypass
hnswlib's fixed entry point and start the internal greedy walk from the
spectral landmark. hnswlib does not expose custom entry points, so this
requires modifying the C++ graph search code.

## Path Forward

- **Short term:** production path stays HNSW + exact cosine re-rank on 384-d.
- **Long term:** fork hnswlib (or implement a custom graph search) to support
  a warm-started entry point. Benchmark query latency and recall at 1M+ scale
  where entry-point quality may matter more.
