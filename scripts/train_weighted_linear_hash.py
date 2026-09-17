#!/usr/bin/env python3
"""Train a linear hash model with bit-weighted BCE.

Hypothesis: ITQ-style binary hashes may have bits that vary in importance for
neighborhood structure. Weighting early/coarse bits more heavily during training
might improve local geometry even if per-bit accuracy changes.

This script uses binary bag-of-words to match Rust inference, and validates with
a local-geometry median-rank metric rather than raw Hamming distance.
"""
import os
import sqlite3

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

DB_PATH = "data/phoenix_arxiv_1m.db"
ITQ_PATH = "data/itq_model_512.npz"
MODEL_PATH = "data/linear_hash_weighted.npz"
RUST_WEIGHTS_PATH = "data/linear_hash_weights_weighted.bin"

N_DOCS = 100_000
TRAIN_N = 50_000
VAL_N = 10_000
MAX_FEATURES = 50_000
LR = 0.05
EPOCHS = 30
SEED = 42


def bit_weights(dim: int = 512, scale: float = 3.0, decay: float = 128.0):
    """Higher weight for early bits, lower for later bits."""
    return 1.0 + scale * np.exp(-np.arange(dim) / decay)


def export_yplh(W, b, vocab, path):
    """Export to the Rust YPLH binary format used by ffi_linear_hash."""
    with open(path, "wb") as f:
        f.write(b"YPLH")
        f.write(np.uint32(1).tobytes())  # version
        f.write(np.uint32(len(vocab)).tobytes())
        f.write(np.uint32(512).tobytes())
        for val in b.astype(np.float32):
            f.write(val.tobytes())
        for tok, wi in zip(vocab, range(len(vocab))):
            tok_b = tok.encode("utf-8")
            f.write(np.uint16(len(tok_b)).tobytes())
            f.write(tok_b)
            for val in W[wi].astype(np.float32):
                f.write(val.tobytes())
    print(f"[+] Exported {path}")


def main():
    print("[+] Loading data...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT title, abstract FROM papers WHERE title IS NOT NULL OR abstract IS NOT NULL LIMIT ?",
        (N_DOCS,),
    )
    rows = cur.fetchall()
    conn.close()

    texts = [f"{t or ''} {a or ''}".strip() for t, a in rows]
    texts = [t for t in texts if t]
    print(f"    Loaded {len(texts)} non-empty docs")

    print("[+] Computing true ITQ hashes (local MiniLM)...")
    model = SentenceTransformer("models/all-MiniLM-L6-v2", local_files_only=True)
    itq = np.load(ITQ_PATH)
    mean = itq["mean"]
    M = itq["proj"]

    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    centered = embs - mean
    projected = centered @ M
    true_hashes = (projected > 0).astype(np.float32)

    print("[+] Building binary BOW (to match Rust tokenizer)...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=2,
        binary=True,
        tokenizer=lambda x: x.lower().split(),
        token_pattern=None,
    )
    X = vectorizer.fit_transform(texts).astype(np.float32)
    vocab = vectorizer.get_feature_names_out()
    vocab_size = X.shape[1]
    print(f"    Vocab size: {vocab_size}")

    weights = bit_weights()

    # Split
    X_train = X[:TRAIN_N]
    H_train = true_hashes[:TRAIN_N]
    X_val = X[TRAIN_N : TRAIN_N + VAL_N]
    H_val = true_hashes[TRAIN_N : TRAIN_N + VAL_N]

    W = np.zeros((vocab_size, 512), dtype=np.float32)
    b = np.zeros(512, dtype=np.float32)

    print(f"[+] Training weighted BCE for {EPOCHS} epochs...")
    rng = np.random.default_rng(SEED)

    for epoch in range(EPOCHS):
        perm = rng.permutation(TRAIN_N)
        total_loss = 0.0
        for i in perm:
            x = X_train[i].toarray().flatten()
            h_true = H_train[i]

            z = x @ W + b
            # stable sigmoid
            h_pred = np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

            # Weighted BCE loss
            eps = 1e-7
            loss = -np.sum(
                weights * (h_true * np.log(h_pred + eps) + (1 - h_true) * np.log(1 - h_pred + eps))
            )
            total_loss += loss

            # Gradient of weighted BCE w.r.t z
            grad = (h_pred - h_true) * weights

            W -= LR * np.outer(x, grad)
            b -= LR * grad

        # Validation: local geometry median rank
        val_pred = 1 / (1 + np.exp(-(X_val @ W + b)))
        val_bits = (val_pred > 0.5).astype(np.float32)
        train_pred = 1 / (1 + np.exp(-(X_train @ W + b)))
        train_bits = (train_pred > 0.5).astype(np.float32)

        # Sample 200 val docs; compare each to a 2K subset of train
        sample_idx = rng.choice(VAL_N, 200, replace=False)
        train_pool = rng.choice(TRAIN_N, 2000, replace=False)
        train_pool_bits = train_bits[train_pool]
        ranks = []
        for si in sample_idx:
            true_d = np.sum(H_train[train_pool] != H_val[si], axis=1)
            fast_d = np.sum(train_pool_bits != val_bits[si], axis=1)
            true_nn = int(np.argmin(true_d))
            fast_rank = int(np.argsort(fast_d)[true_nn])
            ranks.append(fast_rank)

        med_rank = float(np.median(ranks))
        mean_rank = float(np.mean(ranks))
        print(
            f"Epoch {epoch+1:2d}: loss={total_loss/TRAIN_N:.4f}, "
            f"val_median_rank={med_rank:.0f}, val_mean_rank={mean_rank:.0f} "
            f"(random={len(train_pool)/2:.0f})"
        )

    # Final save
    print("[+] Saving model...")
    np.savez(MODEL_PATH, W=W, b=b, vocab=vocab)
    print(f"    Saved {MODEL_PATH}")
    export_yplh(W, b, vocab, RUST_WEIGHTS_PATH)


if __name__ == "__main__":
    main()
