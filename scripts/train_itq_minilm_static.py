# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
Train ITQ on static MiniLM input-embedding document vectors.

Document vector = mean of MiniLM word embeddings for the tokenized text.
No transformer layers, no position embeddings, no L2 normalization.
This makes the tabulated sum identity exact.

Outputs:
  data/itq_model_minilm_static_512.npz
    - mean: (384,) doc-mean vector
    - pca_components: (384, 384) PCA rotation
    - R: (512, 512) ITQ rotation
    - W: (384, 512) effective projection for tabulated table
"""
import sys, os, sqlite3, pickle, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA

DB_PATH = "data/papers_arxiv_1m.db"
QUERY_PATH = "data/paraphrase_queries_500.pkl"
MINILM_PATH = "models/all-MiniLM-L6-v2"
OUT_MODEL = "data/itq_model_minilm_static_512.npz"

N_DOCS_DEFAULT = 100000
ITERS = 50
N_BITS = 512
BATCH_SIZE = 512


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-docs", type=int, default=N_DOCS_DEFAULT)
    parser.add_argument("--exclude-true", action="store_true", default=True,
                        help="Exclude true query PIDs from training sample")
    args = parser.parse_args()

    print("Loading MiniLM tokenizer and input embeddings ...")
    tokenizer = AutoTokenizer.from_pretrained(MINILM_PATH, local_files_only=True)
    minilm = AutoModel.from_pretrained(MINILM_PATH, local_files_only=True)
    emb_matrix = minilm.embeddings.word_embeddings.weight.detach().cpu()  # (V, 384)
    vocab_size, dim = emb_matrix.shape
    print(f"Embedding matrix: {vocab_size} x {dim}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    emb_bag = torch.nn.EmbeddingBag.from_pretrained(emb_matrix, mode="mean", freeze=True).to(device)

    # PIDs to exclude from training sample (avoid leakage)
    exclude_pids = set()
    if args.exclude_true and os.path.exists(QUERY_PATH):
        with open(QUERY_PATH, "rb") as f:
            queries = pickle.load(f)
        exclude_pids = {pid for pid, _, _ in queries}
        print(f"Excluding {len(exclude_pids)} true query PIDs from training sample")

    print(f"Sampling {args.n_docs} docs from DB ...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM papers")
    total_docs = cur.fetchone()[0]

    # Stream rows and collect until we have n_docs valid, non-excluded docs
    cur.execute("SELECT id, title, abstract FROM papers")
    doc_vectors = []
    processed = 0
    collected = 0
    t0 = time.time()
    batch_texts = []

    def flush_batch():
        nonlocal batch_texts, collected
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
        if not any(ids):
            batch_texts = []
            return
        # Build offsets for EmbeddingBag
        lengths = [len(seq) for seq in ids]
        offsets = [0] + list(np.cumsum(lengths[:-1]))
        flat_ids = torch.tensor([t for seq in ids for t in seq], dtype=torch.long, device=device)
        offsets_t = torch.tensor(offsets, dtype=torch.long, device=device)
        with torch.no_grad():
            vecs = emb_bag(flat_ids, offsets_t).cpu().numpy()
        doc_vectors.append(vecs)
        collected += len(batch_texts)
        batch_texts = []

    for pid, title, abstract in cur:
        processed += 1
        if pid in exclude_pids:
            continue
        text = f"{title or ''} {abstract or ''}".strip()
        if not text:
            continue
        batch_texts.append(text)
        if len(batch_texts) >= BATCH_SIZE:
            flush_batch()
            if collected >= args.n_docs:
                break
    flush_batch()

    X = np.concatenate(doc_vectors, axis=0)[:args.n_docs]
    print(f"Collected {X.shape[0]} doc vectors in {time.time()-t0:.1f}s")

    # ── ITQ Training ──
    print("Training ITQ ...")
    t0 = time.time()
    mean = X.mean(axis=0)
    Xc = X - mean

    # PCA to full dim decorrelates; since dim < n_bits, pad afterwards
    pca = PCA(n_components=dim)
    X_pca = pca.fit_transform(Xc)
    print(f"PCA explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")

    # Pad to N_BITS
    if X_pca.shape[1] < N_BITS:
        pad = np.zeros((X_pca.shape[0], N_BITS - X_pca.shape[1]), dtype=np.float32)
        X_pad = np.concatenate([X_pca, pad], axis=1)
    else:
        X_pad = X_pca

    # Random rotation, then iterative quantization
    R = np.random.randn(N_BITS, N_BITS).astype(np.float32)
    R, _ = np.linalg.qr(R)

    for it in range(ITERS):
        V = X_pad @ R
        B = np.sign(V)
        B[B == 0] = 1
        U, _, Vt = np.linalg.svd(B.T @ X_pad)
        R = Vt.T @ U.T
        if (it + 1) % 10 == 0:
            loss = np.linalg.norm(V - B, "fro") ** 2
            print(f"  Iter {it+1}: loss={loss:.2e}")

    # Effective projection from original centered 384-d to 512 bits
    # W = P.T @ R[:dim, :]
    W = pca.components_.T.astype(np.float32) @ R[:dim, :].astype(np.float32)

    print(f"ITQ training done in {time.time()-t0:.1f}s")

    np.savez(
        OUT_MODEL,
        mean=mean.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        R=R.astype(np.float32),
        W=W.astype(np.float32),
        n_bits=N_BITS,
        dim=dim,
        n_docs=X.shape[0],
    )
    print(f"Saved: {OUT_MODEL}")


if __name__ == "__main__":
    main()
