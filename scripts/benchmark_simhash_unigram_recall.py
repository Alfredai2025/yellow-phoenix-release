# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
Benchmark a deterministic SimHash unigram hash vs MiniLM+ITQ on real queries.

This tests the core claim of the multi-layer independent-hash design:
  > Can a deterministic unigram hash (no MiniLM, no ITQ) achieve useful R@K?

Uses:
  - data/paraphrase_queries_500.pkl  (true_pid, title, paraphrase)
  - data/itq_hashes_1m.npy          (MiniLM+ITQ ground-truth hashes)
  - data/paper_meta_arxiv_1m.pkl    (pids, titles)
  - data/papers_arxiv_1m.db         (title + abstract text)
  - data/simhash_unigram_30k_256_f32.bin

The SimHash table is consumed via the existing Rust `encode_tabulated` FFI
(rows are summed and thresholded at zero), so this is a fair test of the
fast-path speed as well.
"""
import sys, os, pickle, sqlite3, json, time, ctypes
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

HASH_PATH = "data/itq_hashes_1m.npy"
META_PATH = "data/paper_meta_arxiv_1m.pkl"
QUERY_PATH = "data/paraphrase_queries_500.pkl"
ITQ_PATH = "itq_model_512.npz"
SIMHASH_TABLE = "data/simhash_unigram_30k_256_f32.bin"
DB_PATH = "data/papers_arxiv_1m.db"
DYLIB_PATH = "target/release/libpams.dylib"
RESULTS_PATH = "data/benchmark_simhash_unigram_results.json"

K_LIST = [1, 10, 50, 100]
DOC_BATCH = 1000
QUERY_BATCH = 32


def load_lib_and_init(table_path: str):
    """Load libpams and point the tabulated encoder at the SimHash table."""
    lib = ctypes.CDLL(DYLIB_PATH)
    table_arr = np.ascontiguousarray(np.fromfile(table_path, dtype=np.uint8))
    rows = len(table_arr) // (256 * 4)
    print(f"[SimHash] Loaded table: {rows} tokens, {len(table_arr)/1e6:.2f} MB")

    lib.tabulated_init.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_size_t
    ]
    lib.tabulated_init(
        table_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        rows, 256
    )

    lib.encode_tabulated.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8)
    ]
    lib.encode_tabulated_batch.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint8)
    ]
    return lib, table_arr


def encode_texts(lib, tokenizer, texts: list[str]) -> np.ndarray:
    """Return Nx32 uint8 SimHash for a list of raw texts."""
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=512,
        padding=False,
    )["input_ids"]

    flat_ids = []
    counts = []
    for ids in encoded:
        flat_ids.extend(ids)
        counts.append(len(ids))

    n_docs = len(texts)
    if not flat_ids:
        return np.zeros((n_docs, 32), dtype=np.uint8)

    ids_arr = np.ascontiguousarray(np.array(flat_ids, dtype=np.uint32))
    counts_arr = np.ascontiguousarray(np.array(counts, dtype=np.uint64))
    out = ctypes.create_string_buffer(64 * n_docs)

    lib.encode_tabulated_batch(
        ids_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        counts_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        n_docs,
        ctypes.cast(out, ctypes.POINTER(ctypes.c_uint8))
    )

    raw = np.frombuffer(out.raw, dtype=np.uint8).reshape(n_docs, 64)
    return np.ascontiguousarray(raw[:, :32])  # first 256 bits are meaningful


def encode_one(lib, tokenizer, text: str) -> bytes:
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=512)
    if not ids:
        return bytes(32)
    ids_arr = np.ascontiguousarray(np.array(ids, dtype=np.uint32))
    out = ctypes.create_string_buffer(64)
    lib.encode_tabulated(
        ids_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        len(ids),
        ctypes.cast(out, ctypes.POINTER(ctypes.c_uint8))
    )
    return out.raw[:32]


def main():
    print("Loading metadata, hashes, and queries ...")
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    pids = meta["pids"]
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}

    itq_hashes = np.load(HASH_PATH, mmap_mode="r")  # N x 64
    N = itq_hashes.shape[0]

    with open(QUERY_PATH, "rb") as f:
        queries = pickle.load(f)[:200]

    true_indices = [pid_to_idx[pid] for pid, _, _ in queries]
    true_set = set(true_indices)
    other = [i for i in range(N) if i not in true_set]
    other = np.random.choice(other, 100000 - len(true_indices), replace=False).tolist()
    cand_indices = np.array(list(true_set) + other, dtype=np.int64)
    np.random.shuffle(cand_indices)
    cand_pids = [pids[i] for i in cand_indices]
    print(f"Candidate pool: {len(cand_indices):,} docs")

    # Ground-truth ITQ hashes for the candidate pool
    cand_itq = itq_hashes[cand_indices]

    # Tokenizer + Rust encoder
    print("Loading tokenizer and Rust encoder ...")
    tokenizer = AutoTokenizer.from_pretrained("models/all-MiniLM-L6-v2", local_files_only=True)
    lib, _ = load_lib_and_init(SIMHASH_TABLE)

    # Fetch title+abstract from SQLite and encode SimHash in batches
    print("Fetching corpus texts and encoding SimHash unigram hashes ...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cand_simhash = np.zeros((len(cand_indices), 32), dtype=np.uint8)

    t0 = time.time()
    for batch_start in range(0, len(cand_pids), DOC_BATCH):
        batch_pids = cand_pids[batch_start:batch_start + DOC_BATCH]
        placeholders = ",".join("?" * len(batch_pids))
        cur.execute(
            f"SELECT id, title, abstract FROM papers WHERE id IN ({placeholders})",
            batch_pids,
        )
        rows = {row[0]: (row[1] or "") + " " + (row[2] or "") for row in cur.fetchall()}

        texts = [rows.get(pid, "") for pid in batch_pids]
        hashes = encode_texts(lib, tokenizer, texts)
        cand_simhash[batch_start:batch_start + len(hashes)] = hashes

        if (batch_start + len(batch_pids)) % 10000 < DOC_BATCH:
            print(f"  ... encoded {min(batch_start + len(batch_pids), len(cand_pids)):,}/{len(cand_pids):,}")
    print(f"SimHash encoding done in {time.time() - t0:.1f}s")

    # Load ITQ model + MiniLM for query encoding
    print("Loading MiniLM + ITQ ...")
    itq = np.load(ITQ_PATH)
    mean = itq["mean"].astype(np.float32)
    W = itq["proj"].astype(np.float32)
    model = SentenceTransformer("models/all-MiniLM-L6-v2", local_files_only=True)

    def encode_minilm_itq(text: str) -> bytes:
        emb = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        x = (emb.astype(np.float32) - mean) @ W
        bits = (x > 0).astype(np.uint8)
        return np.packbits(bits).tobytes()

    hits_itq = {k: 0 for k in K_LIST}
    hits_sim = {k: 0 for k in K_LIST}
    overlap_10 = []
    sim_times = []
    itq_times = []

    print(f"Benchmarking {len(queries)} queries ...")
    for i, (true_pid, title, query_text) in enumerate(queries):
        true_idx_in_cand = np.where(cand_indices == true_indices[i])[0][0]

        t0 = time.perf_counter()
        h_itq = encode_minilm_itq(query_text)
        itq_times.append((time.perf_counter() - t0) * 1e6)

        t0 = time.perf_counter()
        h_sim = encode_one(lib, tokenizer, query_text)
        sim_times.append((time.perf_counter() - t0) * 1e6)

        d_itq = np.bitwise_count(
            np.frombuffer(h_itq, dtype=np.uint8) ^ cand_itq.astype(np.uint8)
        ).sum(axis=1)
        d_sim = np.bitwise_count(
            np.frombuffer(h_sim, dtype=np.uint8) ^ cand_simhash.astype(np.uint8)
        ).sum(axis=1)

        top_itq_10 = set(np.argpartition(d_itq, 10 - 1)[:10])
        top_sim_10 = set(np.argpartition(d_sim, 10 - 1)[:10])
        overlap_10.append(len(top_itq_10 & top_sim_10))

        for k in K_LIST:
            if true_idx_in_cand in np.argpartition(d_itq, k - 1)[:k]:
                hits_itq[k] += 1
            if true_idx_in_cand in np.argpartition(d_sim, k - 1)[:k]:
                hits_sim[k] += 1

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(queries)}")

    n = len(queries)

    print("\n" + "=" * 70)
    print("SIMHASH UNIGRAM vs MiniLM+ITQ BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\n{'K':>5} | {'ITQ R@K':>10} | {'SimHash R@K':>12} | {'Delta':>10}")
    print("-" * 50)
    for k in K_LIST:
        r_itq = hits_itq[k] / n * 100
        r_sim = hits_sim[k] / n * 100
        print(f"{k:>5} | {r_itq:>9.1f}% | {r_sim:>11.1f}% | {r_sim - r_itq:>+9.1f}%")

    print(f"\nTop-10 overlap (same candidate indices):")
    print(f"  Avg: {np.mean(overlap_10):.1f} / 10")
    print(f"  >0:  {sum(1 for x in overlap_10 if x > 0) / n * 100:.1f}% of queries")

    print(f"\nLatency:")
    print(f"  SimHash unigram: {np.mean(sim_times):.1f} us (P50: {np.percentile(sim_times, 50):.1f})")
    print(f"  MiniLM+ITQ:      {np.mean(itq_times):.1f} us (P50: {np.percentile(itq_times, 50):.1f})")
    print(f"  Speedup:         {np.mean(itq_times) / np.mean(sim_times):.1f}x")

    r1_sim = hits_sim[1] / n * 100
    print("\n" + "=" * 70)
    if r1_sim >= 40:
        print("VERDICT: VIABLE.")
        print("         Deterministic unigram SimHash preserves enough signal.")
        print("         Next: add bigram/trigram layers and ensemble voting.")
    elif r1_sim >= 20:
        print("VERDICT: BORDERLINE.")
        print("         Weak signal. Bigram/trigram layers might rescue it.")
        print("         Build the ensemble before deciding.")
    else:
        print("VERDICT: MUSEUM.")
        print("         Deterministic unigram hashing is too weak on this corpus.")
        print("         Stick with learned hashes (ITQ / PQ / MiniLM).")
    print("=" * 70)

    results = {
        "n_queries": n,
        "candidate_pool": int(len(cand_indices)),
        "k_list": K_LIST,
        "r_at_k_itq": {str(k): hits_itq[k] / n * 100 for k in K_LIST},
        "r_at_k_simhash": {str(k): hits_sim[k] / n * 100 for k in K_LIST},
        "avg_top10_overlap": float(np.mean(overlap_10)),
        "queries_with_overlap_pct": float(sum(1 for x in overlap_10 if x > 0) / n * 100),
        "simhash_encode_us_mean": float(np.mean(sim_times)),
        "simhash_encode_us_p50": float(np.percentile(sim_times, 50)),
        "itq_encode_us_mean": float(np.mean(itq_times)),
        "itq_encode_us_p50": float(np.percentile(itq_times, 50)),
        "speedup": float(np.mean(itq_times) / np.mean(sim_times)),
        "verdict": "viable" if r1_sim >= 40 else ("borderline" if r1_sim >= 20 else "museum"),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
