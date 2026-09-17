# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
Measure recall@k for static ITQ vs MiniLM+ITQ on the FULL 1M corpus.

This is the honest metric for a fast encoder used as a candidate generator
before MiniLM float32 re-rank. Hash-only R@1 is the wrong metric.

Reports:
  - recall@100, recall@500, recall@1000
  - encode latency
  - candidate pool size (full 1M)
"""
import sys, os, pickle, sqlite3, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

ITQ_MODEL = "data/itq_model_minilm_static_512.npz"
TABLE_PATH = "data/itq_token_table_minilm_static_512_f32.bin"
META_PATH = "data/paper_meta_arxiv_1m.pkl"
QUERY_PATH = "data/paraphrase_queries_500.pkl"
HASH_PATH = "data/itq_hashes_1m.npy"
DB_PATH = "data/papers_arxiv_1m.db"
RESULTS_PATH = "data/benchmark_static_itq_recall_at_k_results.json"

K_LIST = [100, 500, 1000]
DOC_BATCH = 4096


def main():
    print("Loading static ITQ table and tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained("models/all-MiniLM-L6-v2", local_files_only=True)
    T = np.fromfile(TABLE_PATH, dtype=np.float32).reshape(-1, 512)
    print(f"Table: {T.shape}")

    itq_model = np.load(ITQ_MODEL)
    # We don't strictly need mean/W here because T already encodes them.

    print("Loading metadata, hashes, and queries ...")
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    pids = meta["pids"]
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}

    full_hashes = np.load(HASH_PATH, mmap_mode="r")  # N x 64
    N = full_hashes.shape[0]

    with open(QUERY_PATH, "rb") as f:
        queries = pickle.load(f)[:200]
    true_indices = [pid_to_idx[pid] for pid, _, _ in queries]

    # Compute static hashes for all 1M docs
    print(f"Computing static ITQ hashes for {N:,} docs ...")
    static_hashes = np.zeros((N, 64), dtype=np.uint8)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, title, abstract FROM papers")

    t0 = time.time()
    batch_texts = []
    batch_pids = []
    processed = 0

    def flush_batch():
        nonlocal batch_texts, batch_pids
        if not batch_texts:
            return
        enc = tokenizer(
            batch_texts,
            add_special_tokens=False,
            truncation=True,
            max_length=512,
            padding=False,
        )
        ids = enc["input_ids"]
        lengths = np.array([len(seq) for seq in ids], dtype=np.int64)
        if lengths.sum() == 0:
            batch_texts, batch_pids = [], []
            return

        flat_ids = np.concatenate(ids).astype(np.int64)
        # Sum T rows per document
        offsets = np.r_[0, np.cumsum(lengths[:-1])]
        sums = np.add.reduceat(T[flat_ids], offsets)
        bits = (sums > 0).astype(np.uint8)
        packed = np.packbits(bits, axis=1)

        for pid, h in zip(batch_pids, packed):
            idx = pid_to_idx.get(pid)
            if idx is not None:
                static_hashes[idx] = h

        batch_texts, batch_pids = [], []

    for doc_id, title, abstract in cur:
        text = f"{title or ''} {abstract or ''}".strip()
        batch_texts.append(text)
        batch_pids.append(doc_id)
        if len(batch_texts) >= DOC_BATCH:
            flush_batch()
            processed += DOC_BATCH
            if processed % 100000 == 0:
                print(f"  ... {processed:,}/{N:,} docs")
    flush_batch()
    processed += len(batch_texts)
    print(f"Static hashes computed in {time.time() - t0:.1f}s")

    # Benchmark queries
    print(f"Benchmarking {len(queries)} queries ...")
    hits_static = {k: 0 for k in K_LIST}
    hits_full = {k: 0 for k in K_LIST}
    static_times = []
    full_times = []

    for i, (true_pid, title, query_text) in enumerate(queries):
        true_idx = true_indices[i]

        # Static encode
        t0 = time.perf_counter()
        ids = tokenizer.encode(query_text, add_special_tokens=False, truncation=True, max_length=512)
        if len(ids) == 0:
            h_static = np.zeros(64, dtype=np.uint8)
        else:
            h_static = np.packbits((T[ids].sum(axis=0) > 0).astype(np.uint8))
        static_times.append((time.perf_counter() - t0) * 1e6)

        # MiniLM+ITQ encode (we use precomputed ground-truth hash for true paper,
        # but for query hash we need to encode. For speed, approximate by using
        # the stored ITQ hash of the true paper is NOT the query hash. We must
        # compute the query MiniLM+ITQ hash. Use existing full MiniLM pipeline.)
        t0 = time.perf_counter()
        # Compute full MiniLM+ITQ query hash for fair comparison.
        h_full = _encode_full_itq(query_text)
        full_times.append((time.perf_counter() - t0) * 1e6)

        # Hamming distances
        d_static = np.bitwise_count(h_static ^ static_hashes).sum(axis=1)
        d_full = np.bitwise_count(h_full ^ full_hashes).sum(axis=1)

        for k in K_LIST:
            if true_idx in np.argpartition(d_static, k - 1)[:k]:
                hits_static[k] += 1
            if true_idx in np.argpartition(d_full, k - 1)[:k]:
                hits_full[k] += 1

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(queries)}")

    n = len(queries)

    print("\n" + "=" * 70)
    print("RECALL@K: Static ITQ vs MiniLM+ITQ (FULL 1M CORPUS)")
    print("=" * 70)

    print(f"\n{'K':>5} | {'MiniLM+ITQ':>12} | {'Static ITQ':>12} | {'Delta':>10}")
    print("-" * 55)
    for k in K_LIST:
        r_full = hits_full[k] / n * 100
        r_static = hits_static[k] / n * 100
        print(f"{k:>5} | {r_full:>11.1f}% | {r_static:>11.1f}% | {r_static - r_full:>+9.1f}%")

    print(f"\nLatency:")
    print(f"  Static ITQ: {np.mean(static_times):.1f} us (P50: {np.percentile(static_times, 50):.1f})")
    print(f"  MiniLM+ITQ: {np.mean(full_times):.1f} us (P50: {np.percentile(full_times, 50):.1f})")
    print(f"  Speedup:    {np.mean(full_times) / np.mean(static_times):.1f}x")

    print("\n" + "=" * 70)
    r1000 = hits_static[1000] / n * 100
    if r1000 >= 90:
        print("VERDICT: VIABLE as Tier-0 candidate generator.")
        print("         Static ITQ retrieves >90% of true papers in top-1000.")
        print("         Build hybrid: static top-1000 → MiniLM re-rank.")
    elif r1000 >= 70:
        print("VERDICT: MARGINAL.")
        print("         Recall@1000 is 70–90%. Hybrid may work with quality tolerance.")
    else:
        print("VERDICT: MUSEUM.")
        print("         Even at top-1000, static ITQ misses too many true papers.")
        print("         Re-rank cannot recover what candidate generation loses.")
    print("=" * 70)

    results = {
        "n_queries": n,
        "corpus_size": int(N),
        "k_list": K_LIST,
        "recall_at_k_minilm_itq": {str(k): hits_full[k] / n * 100 for k in K_LIST},
        "recall_at_k_static_itq": {str(k): hits_static[k] / n * 100 for k in K_LIST},
        "static_encode_us_mean": float(np.mean(static_times)),
        "static_encode_us_p50": float(np.percentile(static_times, 50)),
        "minilm_itq_encode_us_mean": float(np.mean(full_times)),
        "speedup": float(np.mean(full_times) / np.mean(static_times)),
        "verdict": "viable" if r1000 >= 90 else ("marginal" if r1000 >= 70 else "museum"),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {RESULTS_PATH}")


_full_minilm = None
_full_itq_mean = None
_full_itq_W = None


def _encode_full_itq(text: str) -> np.ndarray:
    global _full_minilm, _full_itq_mean, _full_itq_W
    if _full_minilm is None:
        from sentence_transformers import SentenceTransformer
        _full_minilm = SentenceTransformer("models/all-MiniLM-L6-v2", local_files_only=True)
        itq = np.load("itq_model_512.npz")
        _full_itq_mean = itq["mean"].astype(np.float32)
        _full_itq_W = itq["proj"].astype(np.float32)
    emb = _full_minilm.encode(text, convert_to_numpy=True, show_progress_bar=False)
    x = (emb.astype(np.float32) - _full_itq_mean) @ _full_itq_W
    return np.packbits((x > 0).astype(np.uint8))


if __name__ == "__main__":
    main()
