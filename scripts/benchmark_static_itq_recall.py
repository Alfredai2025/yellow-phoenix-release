# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
Benchmark static MiniLM-input ITQ vs full MiniLM+ITQ on real queries.

Static encoder:
  text → MiniLM tokenizer → mean(word embeddings) → ITQ projection → 512-bit hash

This is mathematically identical to the tabulated sum, so it is the honest
fast-encoder test.
"""
import sys, os, pickle, time, json, sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

ITQ_MODEL = "data/itq_model_minilm_static_512.npz"
MINILM_PATH = "models/all-MiniLM-L6-v2"
HASH_PATH = "data/itq_hashes_1m.npy"
META_PATH = "data/paper_meta_arxiv_1m.pkl"
QUERY_PATH = "data/paraphrase_queries_500.pkl"
DB_PATH = "data/papers_arxiv_1m.db"
RESULTS_PATH = "data/benchmark_static_itq_results.json"

K_LIST = [1, 10, 50, 100]
DOC_BATCH = 512


def main():
    print("Loading MiniLM tokenizer, model, and static ITQ ...")
    tokenizer = AutoTokenizer.from_pretrained(MINILM_PATH, local_files_only=True)
    minilm = AutoModel.from_pretrained(MINILM_PATH, local_files_only=True)
    emb_matrix = minilm.embeddings.word_embeddings.weight.detach().cpu()
    vocab_size, dim = emb_matrix.shape

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    emb_bag = torch.nn.EmbeddingBag.from_pretrained(emb_matrix, mode="mean", freeze=True).to(device)

    itq = np.load(ITQ_MODEL)
    mean_doc = itq["mean"].astype(np.float32)
    W = itq["W"].astype(np.float32)

    # Full MiniLM+ITQ model for ground truth
    full_model = SentenceTransformer(MINILM_PATH, local_files_only=True)
    itq_full = np.load("itq_model_512.npz")
    mean_full = itq_full["mean"].astype(np.float32)
    W_full = itq_full["proj"].astype(np.float32)

    print("Loading corpus metadata and hashes ...")
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    pids = meta["pids"]
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}
    hashes_full = np.load(HASH_PATH, mmap_mode="r")  # N x 64

    with open(QUERY_PATH, "rb") as f:
        queries = pickle.load(f)[:200]

    true_indices = [pid_to_idx[pid] for pid, _, _ in queries]
    true_set = set(true_indices)
    other = [i for i in range(hashes_full.shape[0]) if i not in true_set]
    other = np.random.choice(other, 100000 - len(true_indices), replace=False).tolist()
    cand_indices = np.array(list(true_set) + other, dtype=np.int64)
    np.random.shuffle(cand_indices)
    cand_pids = [pids[i] for i in cand_indices]
    cand_hashes_full = hashes_full[cand_indices]
    print(f"Candidate pool: {len(cand_indices):,}")

    # Fetch candidate texts from DB
    print("Fetching candidate texts ...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    texts = [""] * len(cand_pids)
    for i in range(0, len(cand_pids), 1000):
        batch_pids = cand_pids[i:i+1000]
        placeholders = ",".join("?" * len(batch_pids))
        cur.execute(f"SELECT id, title, abstract FROM papers WHERE id IN ({placeholders})", batch_pids)
        for pid, title, abstract in cur:
            idx = cand_pids.index(pid)
            texts[idx] = f"{title or ''} {abstract or ''}".strip()

    # Compute static hashes for candidates
    print("Computing static ITQ hashes for candidates ...")
    cand_hashes_static = np.zeros((len(cand_indices), 64), dtype=np.uint8)
    for i in range(0, len(texts), DOC_BATCH):
        batch_texts = texts[i:i+DOC_BATCH]
        enc = tokenizer(
            batch_texts,
            add_special_tokens=False,
            truncation=True,
            max_length=512,
            padding=False,
        )
        ids = enc["input_ids"]
        lengths = [len(seq) for seq in ids]
        offsets = [0] + list(np.cumsum(lengths[:-1]))
        if sum(lengths) == 0:
            continue
        flat_ids = torch.tensor([t for seq in ids for t in seq], dtype=torch.long, device=device)
        offsets_t = torch.tensor(offsets, dtype=torch.long, device=device)
        with torch.no_grad():
            mean_embs = emb_bag(flat_ids, offsets_t).cpu().numpy()  # (n, 384)
        x = (mean_embs - mean_doc) @ W  # (n, 512)
        bits = (x > 0).astype(np.uint8)
        cand_hashes_static[i:i+len(bits)] = np.packbits(bits, axis=1)
        if (i + len(batch_texts)) % 10000 < DOC_BATCH:
            print(f"  ... {min(i + len(batch_texts), len(texts)):,}/{len(texts):,}")

    # Benchmark queries
    hits_full = {k: 0 for k in K_LIST}
    hits_static = {k: 0 for k in K_LIST}
    overlap_10 = []
    full_times = []
    static_times = []

    print(f"Benchmarking {len(queries)} queries ...")
    for i, (true_pid, title, query_text) in enumerate(queries):
        true_idx_in_cand = np.where(cand_indices == true_indices[i])[0][0]

        t0 = time.perf_counter()
        emb = full_model.encode(query_text, convert_to_numpy=True, show_progress_bar=False)
        x_full = (emb.astype(np.float32) - mean_full) @ W_full
        h_full = np.packbits((x_full > 0).astype(np.uint8)).tobytes()
        full_times.append((time.perf_counter() - t0) * 1e6)

        t0 = time.perf_counter()
        ids = tokenizer.encode(query_text, add_special_tokens=False, truncation=True, max_length=512)
        if len(ids) == 0:
            h_static = bytes(64)
        else:
            mean_emb = emb_matrix[ids].mean(dim=0).numpy()
            x_static = (mean_emb - mean_doc) @ W
            h_static = np.packbits((x_static > 0).astype(np.uint8)).tobytes()
        static_times.append((time.perf_counter() - t0) * 1e6)

        d_full = np.bitwise_count(
            np.frombuffer(h_full, dtype=np.uint8) ^ cand_hashes_full.astype(np.uint8)
        ).sum(axis=1)
        d_static = np.bitwise_count(
            np.frombuffer(h_static, dtype=np.uint8) ^ cand_hashes_static.astype(np.uint8)
        ).sum(axis=1)

        top_full_10 = set(np.argpartition(d_full, 10 - 1)[:10])
        top_static_10 = set(np.argpartition(d_static, 10 - 1)[:10])
        overlap_10.append(len(top_full_10 & top_static_10))

        for k in K_LIST:
            if true_idx_in_cand in np.argpartition(d_full, k - 1)[:k]:
                hits_full[k] += 1
            if true_idx_in_cand in np.argpartition(d_static, k - 1)[:k]:
                hits_static[k] += 1

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(queries)}")

    n = len(queries)

    print("\n" + "=" * 70)
    print("STATIC MiniLM-INPUT ITQ vs MiniLM+ITQ BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\n{'K':>5} | {'MiniLM+ITQ':>12} | {'Static ITQ':>12} | {'Delta':>10}")
    print("-" * 55)
    for k in K_LIST:
        r_full = hits_full[k] / n * 100
        r_static = hits_static[k] / n * 100
        print(f"{k:>5} | {r_full:>11.1f}% | {r_static:>11.1f}% | {r_static - r_full:>+9.1f}%")

    print(f"\nTop-10 overlap:")
    print(f"  Avg: {np.mean(overlap_10):.1f} / 10")
    print(f"  >0:  {sum(1 for x in overlap_10 if x > 0) / n * 100:.1f}% of queries")

    print(f"\nLatency:")
    print(f"  Static ITQ: {np.mean(static_times):.1f} us (P50: {np.percentile(static_times, 50):.1f})")
    print(f"  MiniLM+ITQ: {np.mean(full_times):.1f} us (P50: {np.percentile(full_times, 50):.1f})")
    print(f"  Speedup:    {np.mean(full_times) / np.mean(static_times):.1f}x")

    r1_static = hits_static[1] / n * 100
    print("\n" + "=" * 70)
    if r1_static >= 50:
        print("VERDICT: VIABLE as Tier-0 fast path.")
        print("         Static ITQ preserves strong signal.")
    elif r1_static >= 30:
        print("VERDICT: VIABLE as shadow encoder.")
        print("         Useful speedup with acceptable recall drop.")
    elif r1_static >= 20:
        print("VERDICT: BORDERLINE.")
        print("         Weak signal. Might work as coarse filter only.")
    else:
        print("VERDICT: MUSEUM.")
        print("         Static ITQ is too far from MiniLM+ITQ.")
    print("=" * 70)

    results = {
        "n_queries": n,
        "candidate_pool": int(len(cand_indices)),
        "k_list": K_LIST,
        "r_at_k_minilm_itq": {str(k): hits_full[k] / n * 100 for k in K_LIST},
        "r_at_k_static_itq": {str(k): hits_static[k] / n * 100 for k in K_LIST},
        "avg_top10_overlap": float(np.mean(overlap_10)),
        "queries_with_overlap_pct": float(sum(1 for x in overlap_10 if x > 0) / n * 100),
        "static_encode_us_mean": float(np.mean(static_times)),
        "static_encode_us_p50": float(np.percentile(static_times, 50)),
        "minilm_itq_encode_us_mean": float(np.mean(full_times)),
        "speedup": float(np.mean(full_times) / np.mean(static_times)),
        "verdict": "viable" if r1_static >= 30 else ("borderline" if r1_static >= 20 else "museum"),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
