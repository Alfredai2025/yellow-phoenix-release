#!/usr/bin/env python3
"""Train BipolarEncoder(s) to replace MiniLM on ArXiv text.

Trains 4 independent 128-output BipolarEncoders, each predicting one
128-bit chunk of the 512-bit ITQ hash.  Uses perceptron updates in
pure NumPy (no PyTorch needed).

Usage:
    python3 scripts/train_bipolar_encoder.py

Outputs:
    data/bipolar_proj_0.f32  ... data/bipolar_proj_3.f32   (128x128 floats each)
    data/bipolar_bias_0.f32  ... data/bipolar_bias_3.f32   (128 floats each)
"""

import os
import sqlite3
import numpy as np

# ---------------------------------------------------------------------------
# Token hashing — exact match to Rust hash_token + splitmix64
# ---------------------------------------------------------------------------


def hash_token(seed: int, token: str) -> int:
    h = seed
    for b in token.encode("utf-8"):
        h = (h * 0x9E3779B97F4A7C15 + b) & 0xFFFFFFFFFFFFFFFF
    return h


def splitmix64(state: int) -> int:
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return int(z ^ (z >> 31))


def tokenize_to_rows(text: str, seed: int = 42) -> list:
    """Return list of projection-row indices (0..127) for each token."""
    rows = []
    for tok in text.split():
        h = hash_token(seed, tok)
        h = splitmix64(h)
        rows.append(int(h % 128))
    return rows


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_samples(db_path: str, n_train: int = 10_000, n_test: int = 1_000):
    """Load (id, text, itq_hash_512) from ArXiv DB."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, abstract, yp_hash512 FROM papers "
        "WHERE yp_hash512 IS NOT NULL AND length(yp_hash512)=128 "
        "ORDER BY RANDOM() LIMIT ?",
        (n_train + n_test,),
    )
    rows = cur.fetchall()
    conn.close()
    if len(rows) < n_train + n_test:
        raise ValueError(f"Only {len(rows)} papers with YP hashes found")

    out = []
    for pid, title, abstract, hhex in rows:
        text = f"{title or ''} {abstract or ''}".strip()
        hbytes = bytes.fromhex(hhex)
        # Convert 64 bytes -> 512 bits as numpy int8 {0,1}
        bits = np.unpackbits(np.frombuffer(hbytes, dtype=np.uint8))
        out.append((pid, text, bits))
    return out[:n_train], out[n_train:]


# ---------------------------------------------------------------------------
# BipolarEncoder (pure NumPy replica of Rust logic)
# ---------------------------------------------------------------------------


class BipolarEncoderPy:
    def __init__(self):
        self.proj = np.zeros((128, 128), dtype=np.float32)
        self.bias = np.zeros(128, dtype=np.float32)

    def encode(self, rows: list) -> np.ndarray:
        acc = np.zeros(128, dtype=np.float32)
        for r in rows:
            acc += self.proj[r]
        acc += self.bias
        return (acc > 0).astype(np.int32)

    def train_step(self, rows: list, target: np.ndarray, lr: float = 0.5):
        pred = self.encode(rows)
        for r in rows:
            self.proj[r] += lr * (target - pred).astype(np.float32)
        self.bias += lr * (target - pred).astype(np.float32)

    def save(self, prefix: str, idx: int):
        self.proj.tofile(f"{prefix}_proj_{idx}.f32")
        self.bias.tofile(f"{prefix}_bias_{idx}.f32")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train():
    db_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "phoenix_arxiv_1m.db"
    )
    db_path = os.path.abspath(db_path)
    print(f"[+] Loading samples from {db_path}")
    train_set, test_set = load_samples(db_path, n_train=10_000, n_test=1_000)
    print(f"[+] Train: {len(train_set)}  Test: {len(test_set)}")

    # 4 encoders, one per 128-bit chunk
    encoders = [BipolarEncoderPy() for _ in range(4)]

    epochs = 5
    lr = 0.3
    for epoch in range(epochs):
        correct_bits = 0
        total_bits = 0
        for pid, text, bits512 in train_set:
            rows = tokenize_to_rows(text)
            if not rows:
                continue
            for chunk_i in range(4):
                target = bits512[chunk_i * 128 : (chunk_i + 1) * 128]
                enc = encoders[chunk_i]
                pred = enc.encode(rows)
                enc.train_step(rows, target, lr=lr)
                correct_bits += np.sum(pred == target)
                total_bits += 128
        acc = correct_bits / total_bits
        print(f"  Epoch {epoch+1}/{epochs}  bit-acc: {acc:.4f}")
        lr *= 0.9

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, "bipolar")
    for i, enc in enumerate(encoders):
        enc.save(prefix, i)
    print(f"[+] Saved weights to {out_dir}/bipolar_*.f32")

    # Quick test-set correlation
    print("[+] Quick test-set Hamming correlation vs ITQ:")
    for chunk_i in range(4):
        enc = encoders[chunk_i]
        dists = []
        for pid, text, bits512 in test_set:
            rows = tokenize_to_rows(text)
            if not rows:
                continue
            pred = enc.encode(rows)
            target = bits512[chunk_i * 128 : (chunk_i + 1) * 128]
            hamming = np.sum(pred != target)
            dists.append(hamming)
        avg_ham = np.mean(dists)
        print(
            f"  Chunk {chunk_i}: avg Hamming = {avg_ham:.1f} / 128  "
            f"({avg_ham/128*100:.1f}% error)"
        )


if __name__ == "__main__":
    np.random.seed(42)
    train()
