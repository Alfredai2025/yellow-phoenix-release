# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
Benchmark bigram cheat-sheet hash vs MiniLM+ITQ on real queries.

Uses:
  - data/bigram_cheat_sheet_1m.bin
  - data/paraphrase_queries_500.pkl
  - data/itq_hashes_1m.npy
  - data/paper_meta_arxiv_1m.pkl
"""
import sys, os, pickle, json, time, struct
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

CHEAT_BIN = "data/bigram_cheat_sheet_1m.bin"
QUERY_PATH = "data/paraphrase_queries_500.pkl"
HASH_PATH = "data/itq_hashes_1m.npy"
META_PATH = "data/paper_meta_arxiv_1m.pkl"
ITQ_PATH = "itq_model_512.npz"
RESULTS_PATH = "data/benchmark_bigram_cheat_results.json"
MINILM_PATH = "models/all-MiniLM-L6-v2"

K_LIST = [1, 10, 50, 100]


def load_cheat_sheet(path: str, vocab_size: int):
    """Return dict key -> 64-byte hash array."""
    cheat = {}
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            t1, t2 = struct.unpack("<II", header)
            h = np.frombuffer(f.read(64), dtype=np.uint8)
            cheat[t1 * vocab_size + t2] = h
    return cheat


def encode_bigram_cheat(query_text: str, tokenizer, cheat: dict, vocab_size: int) -> bytes:
    ids = tokenizer.encode(query_text, add_special_tokens=False,
                           truncation=True, max_length=512)
    if len(ids) < 2:
        return bytes(64), 0, 0
    accum = np.zeros(64, dtype=np.uint16)
    count = 0
    found = 0
    for i in range(len(ids) - 1):
        key = ids[i] * vocab_size + ids[i + 1]
        count += 1
        h = cheat.get(key)
        if h is not None:
            accum += h.astype(np.uint16)
            found += 1
    # majority vote over *all* query bigrams (missing count as zero)
    result = (accum > (count // 2)).astype(np.uint8)
    return result.tobytes(), found, count


def main():
    print("Loading tokenizer, cheat sheet, and corpus ...")
    tokenizer = AutoTokenizer.from_pretrained(MINILM_PATH, local_files_only=True)
    vocab_size = tokenizer.vocab_size
    cheat = load_cheat_sheet(CHEAT_BIN, vocab_size)
    print(f"Loaded {len(cheat):,} bigram cheat hashes")

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    pids = meta["pids"]
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}

    hashes = np.load(HASH_PATH, mmap_mode="r")  # N x 64
    N = hashes.shape[0]

    with open(QUERY_PATH, "rb") as f:
        queries = pickle.load(f)[:200]

    true_indices = [pid_to_idx[pid] for pid, _, _ in queries]
    true_set = set(true_indices)
    other = [i for i in range(N) if i not in true_set]
    other = np.random.choice(other, 100000 - len(true_indices), replace=False).tolist()
    cand_indices = np.array(list(true_set) + other, dtype=np.int64)
    np.random.shuffle(cand_indices)
    cand_hashes = hashes[cand_indices]
    print(f"Candidate pool: {len(cand_indices):,}")

    itq = np.load(ITQ_PATH)
    mean = itq["mean"].astype(np.float32)
    W = itq["proj"].astype(np.float32)
    model = SentenceTransformer(MINILM_PATH, local_files_only=True)

    def encode_minilm_itq(text: str) -> bytes:
        emb = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        x = (emb.astype(np.float32) - mean) @ W
        bits = (x > 0).astype(np.uint8)
        return np.packbits(bits).tobytes()

    hits_itq = {k: 0 for k in K_LIST}
    hits_bi = {k: 0 for k in K_LIST}
    overlap_10 = []
    itq_times = []
    bi_times = []
    coverage_found = []
    coverage_total = []

    print(f"Benchmarking {len(queries)} queries ...")
    for i, (true_pid, title, query_text) in enumerate(queries):
        true_idx_in_cand = np.where(cand_indices == true_indices[i])[0][0]

        t0 = time.perf_counter()
        h_itq = encode_minilm_itq(query_text)
        itq_times.append((time.perf_counter() - t0) * 1e6)

        t0 = time.perf_counter()
        h_bi, found, total = encode_bigram_cheat(query_text, tokenizer, cheat, vocab_size)
        bi_times.append((time.perf_counter() - t0) * 1e6)
        coverage_found.append(found)
        coverage_total.append(total)

        d_itq = np.bitwise_count(
            np.frombuffer(h_itq, dtype=np.uint8) ^ cand_hashes.astype(np.uint8)
        ).sum(axis=1)
        d_bi = np.bitwise_count(
            np.frombuffer(h_bi, dtype=np.uint8) ^ cand_hashes.astype(np.uint8)
        ).sum(axis=1)

        top_itq_10 = set(np.argpartition(d_itq, 10 - 1)[:10])
        top_bi_10 = set(np.argpartition(d_bi, 10 - 1)[:10])
        overlap_10.append(len(top_itq_10 & top_bi_10))

        for k in K_LIST:
            if true_idx_in_cand in np.argpartition(d_itq, k - 1)[:k]:
                hits_itq[k] += 1
            if true_idx_in_cand in np.argpartition(d_bi, k - 1)[:k]:
                hits_bi[k] += 1

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(queries)}")

    n = len(queries)

    print("\n" + "=" * 70)
    print("BIGRAM CHEAT SHEET vs MiniLM+ITQ BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\n{'K':>5} | {'ITQ R@K':>10} | {'Bigram R@K':>12} | {'Delta':>10}")
    print("-" * 50)
    for k in K_LIST:
        r_itq = hits_itq[k] / n * 100
        r_bi = hits_bi[k] / n * 100
        print(f"{k:>5} | {r_itq:>9.1f}% | {r_bi:>11.1f}% | {r_bi - r_itq:>+9.1f}%")

    print(f"\nBigram coverage per query:")
    print(f"  Avg found / total: {np.mean(coverage_found):.1f} / {np.mean(coverage_total):.1f}")
    print(f"  Queries with zero covered bigrams: {sum(1 for x in coverage_found if x == 0) / n * 100:.1f}%")

    print(f"\nTop-10 overlap (same candidate indices):")
    print(f"  Avg: {np.mean(overlap_10):.1f} / 10")
    print(f"  >0:  {sum(1 for x in overlap_10 if x > 0) / n * 100:.1f}% of queries")

    print(f"\nLatency:")
    print(f"  Bigram cheat: {np.mean(bi_times):.1f} us (P50: {np.percentile(bi_times, 50):.1f})")
    print(f"  MiniLM+ITQ:   {np.mean(itq_times):.1f} us (P50: {np.percentile(itq_times, 50):.1f})")
    print(f"  Speedup:      {np.mean(itq_times) / np.mean(bi_times):.1f}x")

    r1_bi = hits_bi[1] / n * 100
    print("\n" + "=" * 70)
    if r1_bi >= 30:
        print("VERDICT: VIABLE.")
        print("         Bigram cheat captures real semantic signal.")
        print("         Next: build trigram layer and ensemble voting.")
    elif r1_bi >= 10:
        print("VERDICT: BORDERLINE.")
        print("         Some signal, but not strong enough alone.")
        print("         Ensemble + re-rank might rescue it.")
    else:
        print("VERDICT: MUSEUM.")
        print("         Corpus-averaged bigram hashes are too noisy.")
    print("=" * 70)

    results = {
        "n_queries": n,
        "candidate_pool": int(len(cand_indices)),
        "k_list": K_LIST,
        "r_at_k_itq": {str(k): hits_itq[k] / n * 100 for k in K_LIST},
        "r_at_k_bigram": {str(k): hits_bi[k] / n * 100 for k in K_LIST},
        "avg_bigrams_found": float(np.mean(coverage_found)),
        "avg_bigrams_total": float(np.mean(coverage_total)),
        "queries_zero_coverage_pct": float(sum(1 for x in coverage_found if x == 0) / n * 100),
        "avg_top10_overlap": float(np.mean(overlap_10)),
        "bigram_encode_us_mean": float(np.mean(bi_times)),
        "bigram_encode_us_p50": float(np.percentile(bi_times, 50)),
        "itq_encode_us_mean": float(np.mean(itq_times)),
        "speedup": float(np.mean(itq_times) / np.mean(bi_times)),
        "verdict": "viable" if r1_bi >= 30 else ("borderline" if r1_bi >= 10 else "museum"),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
