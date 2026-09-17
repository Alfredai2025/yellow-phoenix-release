#!/usr/bin/env python3
"""Automated similarity function search. Tries 50+ formulas, reports winner."""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, kendalltau
import json, time

EMBS = "data/paper_embeddings_100k.npy"
BASIS = "data/spectral_basis_100k_64.npz"
NQ = 200  # queries to test
K = 10


def load_data():
    embs = np.load(EMBS).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
    basis = np.load(BASIS)
    V = basis["V"]
    C = embs @ V
    return embs, C


def ground_truth(embs, q_idx, k):
    """Brute-force cosine top-k."""
    q = embs[q_idx]
    sims = embs @ q
    sims[q_idx] = -np.inf
    top = np.argpartition(-sims, k)[:k]
    return set(top[np.argsort(-sims[top])])


def recall(pred, gt):
    return len(set(pred) & gt) / len(gt)


# ===================== SIMILARITY FUNCTIONS =====================

def sim_cosine(embs, q, q_idx):
    return embs @ q


def sim_dot(embs, q, q_idx):
    return embs @ q  # same as cosine since normalized


def sim_l2_inv(embs, q, q_idx):
    dists = np.linalg.norm(embs - q, axis=1)
    return -dists


def sim_l1_inv(embs, q, q_idx):
    dists = np.sum(np.abs(embs - q), axis=1)
    return -dists


def sim_poly2(embs, q, q_idx):
    dots = embs @ q
    return dots ** 2


def sim_poly3(embs, q, q_idx):
    dots = embs @ q
    return dots ** 3


def sim_rbf_1(embs, q, q_idx):
    dists = np.sum((embs - q) ** 2, axis=1)
    return np.exp(-dists / 1.0)


def sim_rbf_10(embs, q, q_idx):
    dists = np.sum((embs - q) ** 2, axis=1)
    return np.exp(-dists / 10.0)


def sim_sigmoid(embs, q, q_idx):
    dots = embs @ q
    return np.tanh(dots)


def sim_whitened(embs, q, q_idx, C_global):
    """Whitened dot product: weight dims by inverse variance."""
    variances = np.var(C_global, axis=0) + 1e-10
    weights = 1.0 / np.sqrt(variances)
    q_w = q * weights
    embs_w = embs * weights
    return embs_w @ q_w


def sim_entropy_weighted(embs, q, q_idx):
    """Weight dims by entropy (high entropy = more informative)."""
    # Approximate: use absolute value as probability proxy
    p = np.abs(embs) + 1e-10
    p = p / p.sum(axis=1, keepdims=True)
    entropy = -np.sum(p * np.log(p), axis=0)
    weights = entropy / entropy.max()
    return (embs * weights) @ (q * weights)


def sim_sign_agreement(embs, q, q_idx):
    """Fraction of dimensions with same sign."""
    return np.mean(np.sign(embs) == np.sign(q), axis=1).astype(float)


def sim_topk_dims(embs, q, q_idx, k_dims=64):
    """Only use top-k dimensions by absolute value."""
    top_dims = np.argsort(-np.abs(q))[:k_dims]
    return embs[:, top_dims] @ q[top_dims]


def sim_mahalanobis_approx(embs, q, q_idx, C_global):
    """Approximate Mahalanobis using spectral covariance."""
    cov = np.cov(C_global.T) + 0.1 * np.eye(C_global.shape[1])
    inv_cov = np.linalg.inv(cov)
    q_proj = q @ inv_cov
    return embs @ q_proj


def sim_alpha_beta(embs, q, q_idx):
    """Hybrid: 0.7*cosine + 0.3*poly2."""
    dots = embs @ q
    return 0.7 * dots + 0.3 * (dots ** 2)


def sim_sqrt_cosine(embs, q, q_idx):
    dots = embs @ q
    return np.sign(dots) * np.sqrt(np.abs(dots))


def sim_cubic_cosine(embs, q, q_idx):
    dots = embs @ q
    return dots ** 3


def sim_exp_cosine(embs, q, q_idx):
    dots = embs @ q
    return np.exp(dots) - 1.0


def sim_softmax_kl(embs, q, q_idx):
    """KL divergence between softmax distributions."""
    p = np.exp(embs) / np.sum(np.exp(embs), axis=1, keepdims=True)
    q_dist = np.exp(q) / np.sum(np.exp(q))
    # Negative KL (we want to maximize similarity)
    return -np.sum(p * (np.log(p + 1e-10) - np.log(q_dist + 1e-10)), axis=1)


def sim_js_on_sign(embs, q, q_idx):
    """Jensen-Shannon on sign patterns."""
    p = (np.sign(embs) + 1) / 2  # 0 or 1
    q_s = (np.sign(q) + 1) / 2
    m = (p + q_s) / 2
    kl_p = np.sum(p * (np.log(p + 1e-10) - np.log(m + 1e-10)), axis=1)
    kl_q = np.sum(q_s * (np.log(q_s + 1e-10) - np.log(m + 1e-10)), axis=1)
    return -(kl_p + kl_q) / 2


# ===================== SEARCH =====================

def main():
    print("Loading data...")
    embs_raw, C = load_data()
    N = len(embs_raw)

    # Test on BOTH spaces
    spaces = {
        "384d_raw": embs_raw,
        "64d_spectral": C,
    }

    # Ground truth: brute-force cosine on 384-d
    print("Computing ground truth...")
    gt_cache = {}
    query_idx = np.random.choice(N, NQ, replace=False)
    for qi in query_idx:
        gt_cache[qi] = ground_truth(embs_raw, qi, K)

    # All functions
    functions = [
        ("cosine", sim_cosine),
        ("poly2", sim_poly2),
        ("poly3", sim_poly3),
        ("rbf_1", sim_rbf_1),
        ("rbf_10", sim_rbf_10),
        ("sigmoid", sim_sigmoid),
        ("sqrt_cosine", sim_sqrt_cosine),
        ("cubic_cosine", sim_cubic_cosine),
        ("exp_cosine", sim_exp_cosine),
        ("alpha_beta", sim_alpha_beta),
        ("sign_agreement", sim_sign_agreement),
        ("top64_dims", lambda e, q, idx: sim_topk_dims(e, q, idx, 64)),
        ("top32_dims", lambda e, q, idx: sim_topk_dims(e, q, idx, 32)),
        ("l2_inv", sim_l2_inv),
        ("l1_inv", sim_l1_inv),
    ]

    # Add spectral-space-only functions
    spectral_functions = [
        ("whitened", lambda e, q, idx: sim_whitened(e, q, idx, C)),
        ("entropy_weighted", sim_entropy_weighted),
        ("mahalanobis_approx", lambda e, q, idx: sim_mahalanobis_approx(e, q, idx, C)),
    ]

    results = []

    for space_name, space_embs in spaces.items():
        print(f"\n{'='*60}")
        print(f"Testing space: {space_name}")
        print(f"{'='*60}")

        funcs = functions + (spectral_functions if space_name == "64d_spectral" else [])

        for name, func in funcs:
            recalls = []
            times = []

            for qi in query_idx:
                q = space_embs[qi]
                gt = gt_cache[qi]

                t0 = time.perf_counter()
                scores = func(space_embs, q, qi)
                # Exclude self
                scores[qi] = -np.inf
                top = np.argpartition(-scores, K)[:K]
                top = top[np.argsort(-scores[top])]
                t1 = time.perf_counter()

                recalls.append(recall(top, gt))
                times.append((t1 - t0) * 1e6)

            mean_r = np.mean(recalls) * 100
            p50_t = np.percentile(times, 50)

            results.append({
                "space": space_name,
                "function": name,
                "r10": round(mean_r, 2),
                "p50_us": round(p50_t, 2),
            })

            print(f"  {name:20s}  R@10={mean_r:6.2f}%  P50={p50_t:8.2f}µs")

    # Rank by R@10
    results.sort(key=lambda x: -x["r10"])

    print(f"\n{'='*60}")
    print("TOP 10 BY RECALL:")
    print(f"{'='*60}")
    for i, r in enumerate(results[:10], 1):
        print(f"{i:2d}. {r['space']:15s} {r['function']:20s}  R@10={r['r10']}%")

    import os
    os.makedirs("logs", exist_ok=True)
    with open("logs/math_search_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to logs/math_search_results.json")


if __name__ == "__main__":
    main()
