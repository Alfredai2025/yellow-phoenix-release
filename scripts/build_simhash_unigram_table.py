# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
Build a deterministic SimHash-style unigram table.

Each token t gets a 256-dimensional vector of +/-1 values derived from a
seeded pseudo-random number generator.  A document's hash is computed by
summing the token vectors and thresholding each dimension at zero.

This is mathematically equivalent to classic SimHash (random hyperplane
hashing over bag-of-token features).  The table is stored as float32 so it
can be consumed directly by the existing Rust `encode_tabulated` FFI.

Output:
  data/simhash_unigram_30k_256_f32.bin  (vocab_size x 256 x 4 bytes)
  data/simhash_unigram_meta.npz
"""
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

MINILM_PATH = "models/all-MiniLM-L6-v2"
OUT_TABLE = Path("data/simhash_unigram_30k_256_f32.bin")
OUT_META = Path("data/simhash_unigram_meta.npz")
SEED = 0x4C55434153  # "LUCAS" in hex
N_BITS = 256

def main():
    tokenizer = AutoTokenizer.from_pretrained(MINILM_PATH, local_files_only=True)
    vocab_size = tokenizer.vocab_size
    print(f"Building SimHash unigram table: {vocab_size} tokens x {N_BITS} bits")

    rng = np.random.default_rng(SEED)
    # +/-1 values as float32
    table = (rng.integers(0, 2, size=(vocab_size, N_BITS), dtype=np.int8) * 2 - 1).astype(np.float32)

    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.tofile(OUT_TABLE)

    np.savez(OUT_META,
             vocab_size=vocab_size,
             dim=N_BITS,
             dtype="float32",
             bytes_per_row=N_BITS * 4,
             seed=SEED,
             notes="SimHash random hyperplane unigram signatures. Sum token rows and threshold at 0.")

    print(f"Saved: {OUT_TABLE} ({OUT_TABLE.stat().st_size / 1e6:.2f} MB)")
    print(f"Saved: {OUT_META}")

if __name__ == "__main__":
    main()
