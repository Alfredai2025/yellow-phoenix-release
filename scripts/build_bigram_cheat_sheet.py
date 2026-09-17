# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
Build a bigram cheat-sheet hash table from the 1M corpus.

For every bigram that appears at least MIN_COUNT times, the cheat hash is the
majority-vote (bitwise) average of the 512-bit ITQ hashes of all documents
containing that bigram.

Output:
  data/bigram_cheat_sheet_1m.bin   packed (t1:u32, t2:u32, hash:64 bytes)
  data/bigram_cheat_sheet_meta.npz (min_count, n_bigrams, vocab_size)

Usage:
  python scripts/build_bigram_cheat_sheet.py
  python scripts/build_bigram_cheat_sheet.py --sample 50000
"""
import sys, os, sqlite3, argparse, time, struct
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer

DB_PATH = "data/papers_arxiv_1m.db"
HASH_PATH = "data/itq_hashes_1m.npy"
META_PATH = "data/paper_meta_arxiv_1m.pkl"
MINILM_PATH = "models/all-MiniLM-L6-v2"
OUT_BIN = Path("data/bigram_cheat_sheet_1m.bin")
OUT_META = Path("data/bigram_cheat_sheet_meta.npz")

DEFAULT_MIN_COUNT = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="Number of docs to sample (0 = all)")
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT,
                        help="Minimum bigram occurrences to keep")
    parser.add_argument("--max-bigrams", type=int, default=0,
                        help="Keep only top-N frequent bigrams (0 = no limit)")
    args = parser.parse_args()

    print("Loading tokenizer and ITQ hashes ...")
    tokenizer = AutoTokenizer.from_pretrained(MINILM_PATH, local_files_only=True)
    vocab_size = tokenizer.vocab_size
    hashes = np.load(HASH_PATH, mmap_mode="r")  # N x 64

    # Map DB row order to hash index via meta pids
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    pids = meta["pids"]
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM papers")
    total_docs = cur.fetchone()[0]
    n_docs = args.sample if 0 < args.sample < total_docs else total_docs
    print(f"Corpus: {n_docs:,} / {total_docs:,} docs")

    # ── Pass 1: count bigrams ──
    print(f"Pass 1: counting bigrams (min_count will be {args.min_count}) ...")
    counter = defaultdict(int)
    t0 = time.time()

    cur.execute("SELECT id, title, abstract FROM papers LIMIT ?", (n_docs,))
    processed = 0
    for pid, title, abstract in cur:
        text = (title or "") + " " + (abstract or "")
        ids = tokenizer.encode(text, add_special_tokens=False,
                               truncation=True, max_length=512)
        if len(ids) < 2:
            continue
        # integer key for speed
        V = vocab_size
        for i in range(len(ids) - 1):
            key = ids[i] * V + ids[i + 1]
            counter[key] += 1
        processed += 1
        if processed % 100000 == 0:
            print(f"  ... {processed:,} docs, {len(counter):,} unique bigrams")

    print(f"Pass 1 done in {time.time()-t0:.1f}s. {len(counter):,} unique bigrams.")

    # Keep frequent bigrams
    if args.max_bigrams > 0:
        frequent = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:args.max_bigrams]
        frequent = {k: c for k, c in frequent if c >= args.min_count}
    else:
        frequent = {k: c for k, c in counter.items() if c >= args.min_count}
    del counter
    print(f"Frequent bigrams (count >= {args.min_count}): {len(frequent):,}")

    # ── Pass 2: accumulate hashes ──
    print("Pass 2: accumulating ITQ hashes per bigram ...")
    t0 = time.time()
    acc = {k: np.zeros(64, dtype=np.uint16) for k in frequent}
    counts = {k: 0 for k in frequent}

    cur.execute("SELECT id, title, abstract FROM papers LIMIT ?", (n_docs,))
    processed = 0
    for pid, title, abstract in cur:
        idx = pid_to_idx.get(pid)
        if idx is None:
            continue
        hash_arr = hashes[idx].astype(np.uint16)

        text = (title or "") + " " + (abstract or "")
        ids = tokenizer.encode(text, add_special_tokens=False,
                               truncation=True, max_length=512)
        if len(ids) < 2:
            continue
        V = vocab_size
        for i in range(len(ids) - 1):
            key = ids[i] * V + ids[i + 1]
            if key in acc:
                acc[key] += hash_arr
                counts[key] += 1
        processed += 1
        if processed % 100000 == 0:
            print(f"  ... {processed:,} docs")

    print(f"Pass 2 done in {time.time()-t0:.1f}s.")

    # ── Threshold and save ──
    print("Saving cheat sheet ...")
    OUT_BIN.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_BIN, "wb") as f:
        for key in sorted(acc.keys()):
            t1 = key // vocab_size
            t2 = key % vocab_size
            mean_hash = acc[key]
            cnt = counts[key]
            binary = (mean_hash > (cnt // 2)).astype(np.uint8)
            f.write(struct.pack("<II", int(t1), int(t2)))
            f.write(binary.tobytes())

    np.savez(OUT_META,
             vocab_size=vocab_size,
             min_count=args.min_count,
             max_bigrams=args.max_bigrams,
             n_docs=n_docs,
             n_bigrams=len(acc),
             build_time_seconds=time.time()-t0)

    print(f"Saved: {OUT_BIN} ({OUT_BIN.stat().st_size / 1e6:.2f} MB)")
    print(f"Saved: {OUT_META}")
    print(f"Bigram cheat sheet complete: {len(acc):,} entries.")


if __name__ == "__main__":
    import pickle
    main()
