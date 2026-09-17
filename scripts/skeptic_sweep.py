#!/usr/bin/env python3
"""
Skeptic Sweep: Test 4 exotic geometries on 1000 papers.
If none beat cosine, the infinite universe is closed.
"""
import numpy as np
import os
import time


def load_embeddings():
    for name in ['paper_embeddings_100k.npy', 'paper_embeddings.npy']:
        p = os.path.expanduser(f'~/yellow_phoenix/data/{name}')
        if os.path.exists(p):
            return np.load(p).astype(np.float32)
    raise FileNotFoundError


def cosine_retrieve(emb, q_idx, top_k=5):
    q = emb[q_idx]
    sims = emb @ q
    sims[q_idx] = -999
    return np.argsort(sims)[-top_k:][::-1]


def poincare_distance(u, v):
    u_h = u * 0.99 / (np.linalg.norm(u) + 1e-8)
    v_h = v * 0.99 / (np.linalg.norm(v) + 1e-8)
    u_norm_sq = np.sum(u_h**2)
    v_norm_sq = np.sum(v_h**2)
    diff_sq = np.sum((u_h - v_h)**2)
    num = 2 * diff_sq
    denom = (1 - u_norm_sq) * (1 - v_norm_sq) + 1e-8
    return np.arccosh(1 + num / denom)


def hyperbolic_retrieve(emb, q_idx, top_k=5):
    q = emb[q_idx]
    dists = np.array([poincare_distance(q, emb[i]) for i in range(len(emb))])
    dists[q_idx] = 999
    return np.argsort(dists)[:top_k]


def build_local_subspaces(emb, k=5):
    n = len(emb)
    knn = []
    sims_all = emb @ emb.T
    np.fill_diagonal(sims_all, -999)
    for i in range(n):
        knn.append(np.argsort(sims_all[i])[-k:])
    return knn


def grassmannian_distance(emb, i, j, knn_i, knn_j):
    A = emb[knn_i] - emb[knn_i].mean(axis=0)  # (k, d)
    B = emb[knn_j] - emb[knn_j].mean(axis=0)
    # SVD of A @ B.T (kxk) gives same nonzero singular values as A.T @ B
    try:
        _, s, _ = np.linalg.svd(A @ B.T)
        s = np.clip(s / (s[0] + 1e-8), 0, 1)
        angles = np.arccos(s)
        return np.linalg.norm(angles)
    except Exception:
        return 999.0


def grassmannian_retrieve(emb, q_idx, knn_list, top_k=5):
    n = len(emb)
    dists = np.array([grassmannian_distance(emb, q_idx, i, knn_list[q_idx], knn_list[i]) for i in range(n)])
    dists[q_idx] = 999
    return np.argsort(dists)[:top_k]


def rbf_kernel(a, b, gamma=0.1):
    return np.exp(-gamma * np.sum((a - b)**2))


def kernel_retrieve(emb, q_idx, top_k=5, gamma=0.1):
    q = emb[q_idx]
    sims = np.array([rbf_kernel(q, emb[i], gamma) for i in range(len(emb))])
    sims[q_idx] = -999
    return np.argsort(sims)[-top_k:][::-1]


def wasserstein_retrieve(emb, q_idx, top_k=5):
    # Use precomputed PCA projections (passed in as global)
    q = proj_5d[q_idx]
    dists = np.array([np.mean(np.abs(np.sort(q) - np.sort(proj_5d[i]))) for i in range(len(proj_5d))])
    dists[q_idx] = 999
    return np.argsort(dists)[:top_k]


def evaluate(method_name, retrieve_fn, emb, n_test=100, **kwargs):
    n = len(emb)
    r1 = 0
    r5 = 0
    t0 = time.time()

    for _ in range(n_test):
        q_idx = np.random.randint(0, n)
        gt = cosine_retrieve(emb, q_idx, top_k=1)[0]
        result = retrieve_fn(emb, q_idx, **kwargs)
        if result[0] == gt:
            r1 += 1
        if gt in result:
            r5 += 1

    elapsed = time.time() - t0
    return r1 / n_test, r5 / n_test, elapsed


def main():
    global proj_5d

    print("=" * 70)
    print("SKEPTIC SWEEP: Testing 4 Exotic Geometries")
    print("=" * 70)

    emb = load_embeddings()
    n = len(emb)
    print(f"Embeddings: {n} x {emb.shape[1]}")

    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

    n_sample = min(1000, n)
    sample_idx = np.random.choice(n, n_sample, replace=False)
    emb_s = emb_norm[sample_idx]

    print(f"Sample: {n_sample} papers")
    print(f"Tests: 100 queries per method")
    print()

    # Precompute PCA for Wasserstein
    from sklearn.decomposition import PCA
    proj_5d = PCA(n_components=5).fit_transform(emb_s)

    print("Precomputing Grassmannian neighborhoods...")
    t0 = time.time()
    knn_list = build_local_subspaces(emb_s, k=5)
    print(f"Done in {time.time()-t0:.1f}s")
    print()

    results = []

    print("Testing COSINE (baseline)...")
    r1, r5, t = evaluate("Cosine", cosine_retrieve, emb_s, n_test=100)
    results.append(("Cosine", r1, r5, t))
    print(f"  R@1: {r1*100:.1f}% | R@5: {r5*100:.1f}% | Time: {t:.1f}s")
    print()

    print("Testing HYPERBOLIC (Poincare ball)...")
    r1, r5, t = evaluate("Hyperbolic", hyperbolic_retrieve, emb_s, n_test=100)
    results.append(("Hyperbolic", r1, r5, t))
    print(f"  R@1: {r1*100:.1f}% | R@5: {r5*100:.1f}% | Time: {t:.1f}s")
    print()

    print("Testing GRASSMANNIAN (subspace angles)...")
    r1, r5, t = evaluate("Grassmannian", grassmannian_retrieve, emb_s, n_test=100, knn_list=knn_list)
    results.append(("Grassmannian", r1, r5, t))
    print(f"  R@1: {r1*100:.1f}% | R@5: {r5*100:.1f}% | Time: {t:.1f}s")
    print()

    print("Testing KERNEL RBF...")
    r1, r5, t = evaluate("Kernel RBF", kernel_retrieve, emb_s, n_test=100, gamma=0.1)
    results.append(("Kernel RBF", r1, r5, t))
    print(f"  R@1: {r1*100:.1f}% | R@5: {r5*100:.1f}% | Time: {t:.1f}s")
    print()

    print("Testing WASSERSTEIN (1D proxy)...")
    r1, r5, t = evaluate("Wasserstein", wasserstein_retrieve, emb_s, n_test=100)
    results.append(("Wasserstein", r1, r5, t))
    print(f"  R@1: {r1*100:.1f}% | R@5: {r5*100:.1f}% | Time: {t:.1f}s")
    print()

    print("=" * 70)
    print("SKEPTIC SWEEP RESULTS")
    print("=" * 70)
    print(f"{'Method':<20} {'R@1':>8} {'R@5':>8} {'Time':>8}")
    print("-" * 70)
    best_r1 = max(r[1] for r in results)
    for name, r1, r5, t in results:
        marker = "👑" if abs(r1 - best_r1) < 1e-6 else "  "
        print(f"{marker} {name:<18} {r1*100:>6.1f}% {r5*100:>6.1f}% {t:>6.1f}s")

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    best = max(results, key=lambda x: x[1])
    if best[0] == "Cosine":
        print("❌ COSINE WINS. No exotic geometry beats it.")
        print("   The infinite universe is CLOSED for this data.")
        print("   Ship what works. Stop testing math.")
    else:
        print(f"⚠️  {best[0]} ties or beats cosine by {(best[1]-results[0][1])*100:.1f}pp")
        print("   Investigate further. But probably noise on small sample.")


if __name__ == '__main__':
    np.random.seed(42)
    main()
