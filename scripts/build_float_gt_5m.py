#!/usr/bin/env python3
"""5M float-space ground truth (exclude-self) + Hamming-vs-float recall.

Measures the ITQ ceiling: for Q sampled papers (spread across all 5 sources,
ISM id space), compute
  - float GT  : top-10 cosine neighbors of the paper's 384-dim embedding
                (exact, streaming single pass over the kept corpus)
  - hamming   : exact top-10 over ITQ 512-bit hashes (same pass)
Then recall@k of the hamming top-k against the float GT (self excluded).

No MiniLM needed: this measures neighbor-preservation of the ITQ code,
device-independent. Combined with the on-device 100% index-space R@1
(Hamming-vs-Hamming graph correctness) it closes the float-GT gap.

Output: benchmark_results/float_gt_5m_<date>.json
"""
import os, json, time
import numpy as np

DATA = os.path.expanduser("~/yellow_phoenix/data")
WORK = os.path.join(DATA, "_build_papers5m")
SOURCES = [
    ("data/embeddings_3m.npy", "raw"),
    ("data/embeddings_oai439k.npy", "npy"),
    ("data/embeddings_pubmed_new.npy", "npy"),
    ("data/embeddings_pubmed_new2.npy", "npy"),
    ("data/droplet_embeddings_aligned.npy", "npy"),
]
CHUNK = 50_000
Q = 256
SEED = 20260903
TOPK = 10
POPCNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def open_source(rel, kind):
    path = os.path.join(os.path.expanduser("~/yellow_phoenix"), rel)
    if kind == "raw":
        mm = np.memmap(path, dtype=np.float32, mode="r").reshape(-1, 384)
        assert mm.shape[0] == 3_025_752, f"{rel}: {mm.shape}"
        return mm
    arr = np.load(path)
    assert arr.dtype == np.float32 and arr.shape[1] == 384, f"{rel}: {arr.shape}"
    return arr


def merge_topk(best_s, best_i, cand_s, cand_i):
    """Keep top-TOPK smallest scores of (running best, chunk candidates) per row.
    For similarities (float cosine) pass negated scores so smallest == best."""
    ms = np.concatenate([best_s, cand_s], axis=1)
    mi = np.concatenate([best_i, cand_i], axis=1)
    top = np.argpartition(ms, TOPK, axis=1)[:, :TOPK]
    best_s[:] = np.take_along_axis(ms, top, 1)
    best_i[:] = np.take_along_axis(mi, top, 1)
    return best_s, best_i


def main():
    t0 = time.time()
    m = np.load(os.path.join(DATA, "itq_model_512_fixed.npz"))
    mean = m["mean"].astype(np.float32)
    proj = m["proj"].astype(np.float32)

    kept = [np.load(os.path.join(WORK, f"kept_src{i}.npy")) for i in range(5)]
    bounds = np.cumsum([0] + [len(a) for a in kept])
    total = int(bounds[-1])
    print(f"corpus: {total:,} kept papers", flush=True)

    # --- sample Q ism ids uniformly, resolve to (source, npy_row) ---
    rng = np.random.default_rng(SEED)
    q_ids = np.sort(rng.choice(total, size=Q, replace=False))
    q_src = np.searchsorted(bounds, q_ids, side="right") - 1
    q_row = np.array([int(kept[s][i - bounds[s]]) for s, i in zip(q_src, q_ids)])

    # --- load query vectors once (raw for cosine, centered for ITQ bits) ---
    qraw = np.empty((Q, 384), dtype=np.float32)
    for s in range(5):
        mask = q_src == s
        if not mask.any():
            continue
        arr = open_source(*SOURCES[s])
        qraw[mask] = np.asarray(arr[q_row[mask]], dtype=np.float32)
        del arr
    qnorm = qraw / (np.linalg.norm(qraw, axis=1, keepdims=True) + 1e-12)
    qbits = np.packbits(((qraw - mean) @ proj) > 0, axis=1)  # Q x 64
    print(f"queries loaded in {time.time()-t0:.0f}s", flush=True)

    # --- streaming pass: running top-TOPK per query for float + hamming ---
    # best_f stores NEGATED cosine (smallest == most similar)
    best_f = np.full((Q, TOPK), np.inf, dtype=np.float32)
    best_fid = np.full((Q, TOPK), -1, dtype=np.int64)
    best_h = np.full((Q, TOPK), 65535, dtype=np.uint16)
    best_hid = np.full((Q, TOPK), -1, dtype=np.int64)

    for si, (rel, kind) in enumerate(SOURCES):
        arr = open_source(rel, kind)
        rows = kept[si]
        n = len(rows)
        for lo in range(0, n, CHUNK):
            hi = min(lo + CHUNK, n)
            x = np.ascontiguousarray(arr[rows[lo:hi]], dtype=np.float32)  # k x 384
            ids = bounds[si] + lo + np.arange(hi - lo, dtype=np.int64)

            # float cosine top-10 of this chunk (negated), merged into best
            xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
            sn = -(qnorm @ xn.T)                               # Q x k
            idx = np.argpartition(sn, TOPK, axis=1)[:, :TOPK]
            cs = np.take_along_axis(sn, idx, 1)
            merge_topk(best_f, best_fid, cs, ids[idx])

            # hamming top-10 of this chunk via LUT popcount (per-row merge)
            bits = np.packbits(((x - mean) @ proj) > 0, axis=1)   # k x 64
            for q in range(Q):
                d = POPCNT[np.bitwise_xor(bits, qbits[q])].sum(
                    axis=1, dtype=np.uint16)
                j = np.argpartition(d, TOPK)[:TOPK]
                ms = np.concatenate([best_h[q], d[j]])
                mi = np.concatenate([best_hid[q], ids[j]])
                top = np.argpartition(ms, TOPK)[:TOPK]
                best_h[q] = ms[top]
                best_hid[q] = mi[top]
        print(f"  src{si}: {n:,} rows | {time.time()-t0:.0f}s", flush=True)
        del arr

    # --- recall@k: hamming top-k vs float GT top-10, self excluded ---
    order = np.argsort(best_f, axis=1)               # ascending: best (neg) first
    gt = np.take_along_axis(best_fid, order, 1)      # Q x 10 float GT ids
    horder = np.argsort(best_h, axis=1)              # ascending: closest first
    ham = np.take_along_axis(best_hid, horder, 1)    # Q x 10 ham ids

    def recall(k, mask):
        g = gt[mask][:, :k]
        h = ham[mask][:, :k]
        hits = 0
        for i in range(len(g)):
            hits += len(set(h[i].tolist()) & set(g[i].tolist()))
        return hits / (len(g) * k)

    def gt5_in_ham10(mask):
        """Operational metric: fraction of the float top-5 present in the
        hamming top-10 (what a user sees when we show 10 results)."""
        g = gt[mask][:, :5]
        h = ham[mask]
        hits = 0
        for i in range(len(g)):
            hits += len(set(h[i].tolist()) & set(g[i].tolist()))
        return hits / (len(g) * 5)

    result = {
        "method": "exclude-self float kNN (cosine, exact) vs exact ITQ-512 hamming top-k",
        "queries": Q, "topk": TOPK, "seed": SEED,
        "total_corpus": total,
        "overall": {f"recall@{k}": round(recall(k, np.ones(Q, bool)), 4) for k in (1, 5, 10)}
                   | {"gt5_in_ham10": round(gt5_in_ham10(np.ones(Q, bool)), 4)},
        "per_source": {},
        "runtime_s": round(time.time() - t0, 1),
    }
    for s in range(5):
        mask = q_src == s
        if mask.any():
            result["per_source"][f"src{s}"] = {
                "n_queries": int(mask.sum()),
                **{f"recall@{k}": round(recall(k, mask), 4) for k in (1, 5, 10)},
                "gt5_in_ham10": round(gt5_in_ham10(mask), 4),
            }

    out = os.path.join(os.path.expanduser("~/yellow_phoenix"),
                       "benchmark_results", f"float_gt_5m_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
