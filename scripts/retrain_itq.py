#!/usr/bin/env python3
"""
Retrain ITQ on real embeddings. Evaluate Hamming R@1.
Uses 100K sample for fast training, evaluates on full set.
"""
import numpy as np
import os
import time
import glob

# =============================================================================
# CONFIG
# =============================================================================
N_BITS = 512
N_TRAIN = 100_000
N_EVAL = 1_000
N_ITER = 50

# =============================================================================
# FIND EMBEDDINGS
# =============================================================================
candidates = [
    os.path.expanduser("~/yellow_phoenix/data/paper_embeddings_1m_synthetic.npy"),
    os.path.expanduser("~/yellow_phoenix/data/paper_embeddings_arxiv_1m.npy"),
    os.path.expanduser("~/Documents/11 Aug Yellow Back up/paper_embeddings.npy"),
] + glob.glob(os.path.expanduser("~/yellow_phoenix/data/paper_embeddings_*.npy"))

EMB_PATH = None
for c in candidates:
    if os.path.exists(c):
        EMB_PATH = c
        break

if not EMB_PATH:
    print("ERROR: No embedding file found. Searched:")
    for c in candidates:
        print(f"  {c}")
    exit(1)

print(f"[+] Loading embeddings from: {EMB_PATH}")
X = np.load(EMB_PATH).astype(np.float32)
N, d = X.shape
print(f"    Shape: {N:,} vectors x {d} dims  ({X.nbytes / 1e9:.2f} GB)")

# =============================================================================
# POPCOUNT LUT
# =============================================================================
POPCOUNT = np.zeros(256, dtype=np.int32)
for i in range(256):
    POPCOUNT[i] = bin(i).count('1')

def hamming_all(codes, q):
    """codes: (N, n_bytes) uint8, q: (n_bytes,) uint8"""
    xor = np.bitwise_xor(codes, q)
    return POPCOUNT[xor].sum(axis=1)

# =============================================================================
# TRAIN ITQ
# =============================================================================
print(f"\n[+] Training ITQ ({N_BITS} bits, {N_ITER} iterations)...")
print(f"    Using {min(N_TRAIN, N):,} vectors for rotation learning")

# Center
mean = X[:min(N_TRAIN, N)].mean(axis=0)
Xc = X[:min(N_TRAIN, N)] - mean

# PCA to N_BITS
if d >= N_BITS:
    print("    PCA via SVD...")
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    W_pca = Vt[:N_BITS].T.astype(np.float32)
else:
    W_pca = np.eye(d, N_BITS, dtype=np.float32)

X_pca = Xc @ W_pca

# Iterative Quantization
R = np.eye(N_BITS, dtype=np.float32)
best_r1 = 0.0

for it in range(N_ITER):
    V = X_pca @ R
    B = np.sign(V)
    B[B == 0] = 1
    
    # SVD for optimal rotation
    C = B.T @ X_pca  # (N_BITS, N_BITS)
    U, _, Vt = np.linalg.svd(C)
    R_new = Vt.T @ U.T
    
    # Check convergence
    diff = np.linalg.norm(R - R_new)
    R = R_new
    
    if (it + 1) % 10 == 0 or diff < 1e-6:
        # Quick R@1 check on training sample (100 queries)
        V_s = X_pca[:100] @ R
        B_s = (V_s > 0).astype(np.uint8)
        codes_s = np.packbits(B_s.reshape(100, N_BITS // 8, 8), axis=2).reshape(100, N_BITS // 8)
        correct = 0
        for i in range(min(100, len(codes_s))):
            dists = hamming_all(codes_s, codes_s[i])
            dists[i] = 999999
            if dists.argmin() == i:
                correct += 1
        r1 = correct / min(100, len(codes_s))
        print(f"    Iter {it+1:2d}: R@1={r1:.1%}  rotation_diff={diff:.6f}")
        if r1 > best_r1:
            best_r1 = r1
        if diff < 1e-6:
            print("    Converged.")
            break

# =============================================================================
# GENERATE HASHES FOR FULL DATASET
# =============================================================================
print(f"\n[+] Generating {N_BITS}-bit hashes for all {N:,} docs...")
batch_size = 50_000
all_codes = np.zeros((N, N_BITS // 8), dtype=np.uint8)

for start in range(0, N, batch_size):
    end = min(start + batch_size, N)
    batch = X[start:end] - mean
    batch_pca = batch @ W_pca
    batch_rot = batch_pca @ R
    bits = (batch_rot > 0).astype(np.uint8)
    all_codes[start:end] = np.packbits(bits.reshape(end - start, N_BITS // 8, 8), axis=2).reshape(end - start, N_BITS // 8)
    if (end // batch_size) % 5 == 0:
        print(f"    ... {end:,} / {N:,}")

# =============================================================================
# EVALUATE HAMMING R@1
# =============================================================================
print(f"\n[+] Evaluating exact Hamming R@1 on {N_EVAL} queries...")

# Sample queries, ensure they're spread out
query_idx = np.linspace(0, N - 1, N_EVAL, dtype=np.int64)
correct = 0
total_dists = []

t0 = time.perf_counter()
for i, qi in enumerate(query_idx):
    q_code = all_codes[qi]
    dists = hamming_all(all_codes, q_code)
    dists[qi] = 999999  # exclude self
    nn = dists.argmin()
    if nn == qi:
        correct += 1
    total_dists.append(dists[nn])
    
    if (i + 1) % 100 == 0:
        print(f"    ... {i+1}/{N_EVAL}  running R@1: {correct/(i+1):.1%}")

elapsed = time.perf_counter() - t0
r1 = correct / N_EVAL
mean_nn_dist = np.mean(total_dists)

print(f"\n{'='*60}")
print(f"HAMMING R@1 RESULT")
print(f"{'='*60}")
print(f"Dataset:        {EMB_PATH}")
print(f"Vectors:        {N:,}")
print(f"Dimensions:     {d}")
print(f"Hash bits:      {N_BITS}")
print(f"Queries:        {N_EVAL}")
print(f"R@1:            {r1:.1%}")
print(f"Mean NN dist:   {mean_nn_dist:.1f} / {N_BITS} bits")
print(f"Eval time:      {elapsed:.1f}s ({elapsed/N_EVAL*1000:.1f} ms/query)")
print(f"{'='*60}")

# =============================================================================
# SAVE MODEL
# =============================================================================
out_path = "data/itq_retrained_20260825.npz"
np.savez(out_path,
    mean=mean.astype(np.float32),
    W_pca=W_pca.astype(np.float32),
    R=R.astype(np.float32),
    n_bits=N_BITS,
    r1=float(r1),
    dataset=EMB_PATH,
)
print(f"\n[+] Saved: {out_path}")

# Also save hashes for Rust
all_codes.tofile("data/paper_hashes_itq_retrained.bin")
print(f"[+] Saved: data/paper_hashes_itq_retrained.bin  ({all_codes.shape})")

# =============================================================================
# VERDICT
# =============================================================================
if r1 >= 0.85:
    print("\n*** EXCELLENT: R@1 >= 85%. Hash quality is production-grade. ***")
    print("    Action: Replace old ITQ model. No PQ needed.")
elif r1 >= 0.70:
    print("\n*** GOOD: R@1 >= 70%. Hash quality is usable. ***")
    print("    Action: Replace old ITQ model. Consider PQ re-rank for final 5%.")
else:
    print("\n*** POOR: R@1 < 70%. Hash space is too noisy for HNSW alone. ***")
    print("    Action: Wire PQ ADC as HNSW re-ranker (next script).")
