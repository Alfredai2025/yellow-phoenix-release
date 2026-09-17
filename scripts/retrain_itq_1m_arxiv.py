#!/usr/bin/env python3
"""
Memory-efficient retraining of 512-bit ITQ on the 1M+ ArXiv embeddings.
Processes data in batches so it fits on memory-constrained machines.

Uses random orthonormal projection 384 -> 512, then ITQ rotation.
Outputs:
  data/itq_model_1m_512.npz  (mean, W, R)
  data/itq_hashes_1m_arxiv.npy
"""
import numpy as np
import os
import time

EMB_PATH = "data/paper_embeddings_arxiv_1m.npy"
MODEL_OUT = "data/itq_model_1m_512.npz"
HASH_OUT = "data/itq_hashes_1m_arxiv.npy"
MEAN_OUT = "data/itq_mean_1m_arxiv.npy"
N_BITS = 512
N_ITER = 50
BATCH_SIZE = 50_000


def random_projection_384_512(d_in=384, n_bits=512, seed=42):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d_in, n_bits))
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    W = U @ Vt  # (d_in, n_bits), rows orthonormal
    return W.astype(np.float32)


def main():
    if not os.path.exists(EMB_PATH):
        raise FileNotFoundError(EMB_PATH)

    print(f"[+] Loading embeddings from {EMB_PATH}")
    X = np.load(EMB_PATH, mmap_mode='r')
    n, d = X.shape
    print(f"    Shape: {n} x {d}, dtype: {X.dtype}")

    print("[+] Computing mean in batches...")
    mean = np.zeros(d, dtype=np.float64)
    for i in range(0, n, BATCH_SIZE):
        mean += X[i:i+BATCH_SIZE].astype(np.float32).sum(axis=0)
    mean = (mean / n).astype(np.float32)
    np.save(MEAN_OUT, mean)
    print(f"    Saved {MEAN_OUT}")

    print(f"[+] Random orthonormal projection {d} -> {N_BITS}...")
    W = random_projection_384_512(d, N_BITS)

    print(f"[+] ITQ rotation ({N_ITER} iterations, batch_size={BATCH_SIZE})...")
    R = np.random.randn(N_BITS, N_BITS).astype(np.float32)
    U, _, Vt = np.linalg.svd(R)
    R = (U @ Vt).astype(np.float32)

    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

    for it in range(N_ITER):
        t0 = time.time()
        C = np.zeros((N_BITS, N_BITS), dtype=np.float64)

        for b in range(n_batches):
            i = b * BATCH_SIZE
            j = min(i + BATCH_SIZE, n)
            X_batch = X[i:j].astype(np.float32) - mean
            X_proj_batch = X_batch @ W
            V_batch = X_proj_batch @ R
            B_batch = np.sign(V_batch)
            B_batch[B_batch == 0] = 1
            C += X_proj_batch.T @ B_batch

        U, _, Vt = np.linalg.svd(C)
        R = (U @ Vt).astype(np.float32)

        if (it + 1) % 5 == 0 or it == 0:
            # Sample semantic R@1 on first 10k
            sample = min(10_000, n)
            X_s = X[:sample].astype(np.float32)
            V_s = (X_s - mean) @ W @ R
            B_s = np.sign(V_s)
            B_s[B_s == 0] = 1
            loss = np.linalg.norm(V_s - B_s, 'fro') / sample
            bits = (V_s > 0).astype(np.uint8)
            packed = np.packbits(bits, axis=1)
            # Ground truth: top-1 cosine neighbor in original embeddings
            X_s_norm = X_s / (np.linalg.norm(X_s, axis=1, keepdims=True) + 1e-8)
            sim = X_s_norm @ X_s_norm.T
            np.fill_diagonal(sim, -2)
            gt_nn = sim.argmax(axis=1)
            # Hash nearest neighbor
            popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint16)
            xor = np.bitwise_xor(packed[:, None], packed[None, :])  # (sample, sample, 64)
            hamming = popcount_table[xor].sum(axis=2)
            np.fill_diagonal(hamming, 999999)
            hash_nn = hamming.argmin(axis=1)
            r1 = (gt_nn == hash_nn).mean()
            print(f"    Iter {it+1}/{N_ITER}: loss={loss:.4f}, sample semantic R@1={r1:.1%}, time={time.time()-t0:.1f}s")

    print("[+] Saving model...")
    np.savez(MODEL_OUT, mean=mean.astype(np.float32), W=W.astype(np.float32), R=R.astype(np.float32))
    print(f"    Saved {MODEL_OUT}")

    print("[+] Generating hashes in batches...")
    hashes = np.empty((n, N_BITS // 8), dtype=np.uint8)
    for i in range(0, n, BATCH_SIZE):
        j = min(i + BATCH_SIZE, n)
        V = (X[i:j].astype(np.float32) - mean) @ W @ R
        bits = (V > 0).astype(np.uint8)
        hashes[i:j] = np.packbits(bits, axis=1)
    np.save(HASH_OUT, hashes)
    print(f"    Saved {HASH_OUT} shape={hashes.shape}")

    # Final sample semantic R@1
    sample = min(10_000, n)
    X_s = X[:sample].astype(np.float32)
    X_s_norm = X_s / (np.linalg.norm(X_s, axis=1, keepdims=True) + 1e-8)
    sim = X_s_norm @ X_s_norm.T
    np.fill_diagonal(sim, -2)
    gt_nn = sim.argmax(axis=1)
    popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint16)
    xor = np.bitwise_xor(hashes[:sample][:, None], hashes[:sample][None, :])
    hamming = popcount_table[xor].sum(axis=2)
    np.fill_diagonal(hamming, 999999)
    hash_nn = hamming.argmin(axis=1)
    r1 = (gt_nn == hash_nn).mean()
    print(f"\n[+] Final sample semantic Hamming R@1: {r1:.1%}")


if __name__ == "__main__":
    main()
