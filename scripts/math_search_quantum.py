#!/usr/bin/env python3
"""
Quantum & Advanced Math Search for Yellow Phoenix Retrieval.
Tests 40+ similarity functions including quantum formalism,
information geometry, spectral measures, and geometric algebra.
Run: python scripts/math_search_quantum.py
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import sqrtm, logm
from scipy.stats import entropy
import json, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

EMBS = "data/paper_embeddings_100k.npy"
BASIS = "data/spectral_basis_100k_64.npz"
NQ = 200
K = 10

# ── Load ──
def load_data():
    embs = np.load(EMBS).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
    basis = np.load(BASIS)
    V = basis["V"]
    C = embs @ V
    return embs, C

# ── Ground truth ──
def ground_truth(embs, q_idx, k):
    q = embs[q_idx]
    sims = embs @ q
    sims[q_idx] = -np.inf
    top = np.argpartition(-sims, k)[:k]
    return set(top[np.argsort(-sims[top])])

def recall(pred, gt):
    return len(set(pred) & gt) / len(gt)

# ═══════════════════════════════════════════════════════════════
#  CLASSICAL SIMILARITIES
# ═══════════════════════════════════════════════════════════════

def sim_cosine(embs, q, qi):
    return embs @ q

def sim_dot(embs, q, qi):
    return embs @ q

def sim_l2_inv(embs, q, qi):
    return -np.linalg.norm(embs - q, axis=1)

def sim_l1_inv(embs, q, qi):
    return -np.sum(np.abs(embs - q), axis=1)

def sim_poly2(embs, q, qi):
    d = embs @ q
    return d * d

def sim_poly3(embs, q, qi):
    d = embs @ q
    return d ** 3

def sim_rbf_1(embs, q, qi):
    return np.exp(-np.sum((embs - q)**2, axis=1))

def sim_rbf_10(embs, q, qi):
    return np.exp(-np.sum((embs - q)**2, axis=1) / 10.0)

def sim_sigmoid(embs, q, qi):
    return np.tanh(embs @ q)

def sim_sqrt_cosine(embs, q, qi):
    d = embs @ q
    return np.sign(d) * np.sqrt(np.abs(d))

def sim_exp_cosine(embs, q, qi):
    return np.exp(embs @ q) - 1.0

def sim_alpha_beta(embs, q, qi):
    d = embs @ q
    return 0.7 * d + 0.3 * d * d

def sim_sign_agreement(embs, q, qi):
    return np.mean(np.sign(embs) == np.sign(q), axis=1).astype(float)

def sim_topk_dims(embs, q, qi, kdims):
    top = np.argsort(-np.abs(q))[:kdims]
    return embs[:, top] @ q[top]

# ═══════════════════════════════════════════════════════════════
#  QUANTUM-INSPIRED SIMILARITIES
# ═══════════════════════════════════════════════════════════════

def sim_quantum_fidelity(embs, q, qi):
    """
    Fidelity between pure quantum states |psi_i> and |psi_q>.
    For normalized vectors: F = |<psi_i|psi_q>|^2 = (cosine)^2.
    """
    d = embs @ q
    return d * d

def sim_quantum_kernel(embs, q, qi, sigma=1.0):
    """
    Quantum kernel: K(x,y) = |<phi(x)|phi(y)>|^2 where phi is a
    feature map. Approximated by RBF on a lifted space.
    """
    dists = np.sum((embs - q)**2, axis=1)
    return np.exp(-dists / (2 * sigma * sigma))

def sim_quantum_js(embs, q, qi):
    """
    Quantum Jensen-Shannon divergence between density matrices.
    rho = |psi><psi|. For pure states: QJS = 1 - |<psi_i|psi_q>|^2.
    We return negative QJS as similarity.
    """
    d = embs @ q
    return -(1.0 - d * d)

def sim_quantum_holevo(embs, q, qi):
    """
    Holevo-inspired: distinguishability of ensembles.
    Use Bhattacharyya coefficient on positive/negative subspaces.
    """
    pos_i = np.maximum(embs, 0)
    pos_q = np.maximum(q, 0)
    neg_i = np.maximum(-embs, 0)
    neg_q = np.maximum(-q, 0)
    bc_pos = np.sum(np.sqrt(pos_i * pos_q), axis=1)
    bc_neg = np.sum(np.sqrt(neg_i * neg_q), axis=1)
    return bc_pos + bc_neg

def sim_quantum_bures(embs, q, qi):
    """
    Bures metric for quantum states: d_B^2 = 2 - 2*sqrt(F).
    Similarity = -d_B^2 = 2*sqrt(F) - 2.
    """
    d = np.clip(embs @ q, -1.0, 1.0)
    F = d * d
    return 2.0 * np.sqrt(F) - 2.0

def sim_quantum_helstrom(embs, q, qi):
    """
    Helstrom bound: probability of distinguishing two states.
    P_dist = 0.5 * (1 + sqrt(1 - F)). Similarity = -P_dist.
    """
    d = np.clip(embs @ q, -1.0, 1.0)
    F = d * d
    P_dist = 0.5 * (1.0 + np.sqrt(np.clip(1.0 - F, 0, 1)))
    return -P_dist

def sim_quantum_renyi_2(embs, q, qi):
    """
    Renyi-2 entropy of the overlap: S_2 = -log(tr(rho_i rho_q)).
    For pure states: tr(rho_i rho_q) = |<i|q>|^2 = F.
    Similarity = -S_2 = log(F).
    """
    d = np.clip(embs @ q, -1.0, 1.0)
    F = d * d
    F = np.clip(F, 1e-10, 1.0)
    return np.log(F)

def sim_quantum_tsallis(embs, q, qi, q_param=2.0):
    """
    Tsallis entropy: S_q = (1 - tr(rho^q)) / (q - 1).
    For overlap: similarity = tr(rho_i rho_q) = F.
    """
    d = np.clip(embs @ q, -1.0, 1.0)
    return d * d

def sim_quantum_swap(embs, q, qi):
    """
    Swap test: probability of |0> after swap = (1 + |<i|q>|^2)/2.
    Similarity = P(|0>) - 0.5 = F/2.
    """
    d = embs @ q
    return 0.5 * d * d

# ═══════════════════════════════════════════════════════════════
#  INFORMATION GEOMETRY
# ═══════════════════════════════════════════════════════════════

def sim_kl_symmetric(embs, q, qi):
    """Symmetric KL on softmax distributions."""
    p = np.exp(embs)
    p = p / p.sum(axis=1, keepdims=True)
    qd = np.exp(q)
    qd = qd / qd.sum()
    kl_pq = np.sum(p * (np.log(p + 1e-10) - np.log(qd + 1e-10)), axis=1)
    kl_qp = np.sum(qd * (np.log(qd + 1e-10) - np.log(p + 1e-10)), axis=1)
    return -(kl_pq + kl_qp) / 2.0

def sim_js_softmax(embs, q, qi):
    """Jensen-Shannon on softmax."""
    p = np.exp(embs)
    p = p / p.sum(axis=1, keepdims=True)
    qd = np.exp(q)
    qd = qd / qd.sum()
    m = (p + qd) / 2.0
    kl_pm = np.sum(p * (np.log(p + 1e-10) - np.log(m + 1e-10)), axis=1)
    kl_qm = np.sum(qd * (np.log(qd + 1e-10) - np.log(m + 1e-10)), axis=1)
    return -(kl_pm + kl_qm) / 2.0

def sim_hellinger(embs, q, qi):
    """Hellinger distance on sign patterns."""
    p = (np.sign(embs) + 1) / 2.0 + 1e-10
    qd = (np.sign(q) + 1) / 2.0 + 1e-10
    bc = np.sum(np.sqrt(p * qd), axis=1)
    return bc

def sim_itakura_saito(embs, q, qi):
    """Itakura-Saito divergence (spectral distance)."""
    ratio = embs / (q + 1e-10)
    return -(ratio - np.log(ratio) - 1.0).sum(axis=1)

def sim_chi_square(embs, q, qi):
    """Chi-square statistic as similarity."""
    return -np.sum((embs - q)**2 / (np.abs(q) + 1e-10), axis=1)

# ═══════════════════════════════════════════════════════════════
#  SPECTRAL / MATRIX MEASURES
# ═══════════════════════════════════════════════════════════════

def sim_frobenius_inner(embs, q, qi):
    """Treat vectors as diagonal matrices, inner product."""
    return np.sum(embs * q, axis=1)

def sim_hadamard_ratio(embs, q, qi):
    """Hadamard product then sum ratio."""
    h = embs * q
    return h.sum(axis=1) / (np.abs(h).sum(axis=1) + 1e-10)

def sim_determinant(embs, q, qi):
    """Determinant-inspired: product of matching-sign dims."""
    mask = np.sign(embs) == np.sign(q)
    prod = np.prod(np.where(mask, np.abs(embs * q) + 1e-10, 1.0), axis=1)
    return prod ** (1.0 / embs.shape[1])

def sim_spectral_angle(embs, q, qi):
    """Spectral angle mapper (SAM) from remote sensing."""
    num = embs @ q
    den = np.linalg.norm(embs, axis=1) * np.linalg.norm(q)
    return np.arccos(np.clip(num / (den + 1e-10), -1, 1))

# ═══════════════════════════════════════════════════════════════
#  GEOMETRIC ALGEBRA (on 64-d spectral)
# ═══════════════════════════════════════════════════════════════

def sim_wedge_norm(embs, q, qi):
    """
    Wedge product magnitude: |a ∧ b| = |a||b|sin(theta).
    For normalized: sin(arccos(cosine)).
    """
    d = np.clip(embs @ q, -1.0, 1.0)
    theta = np.arccos(d)
    return np.sin(theta)

def sim_inner_plus_wedge(embs, q, qi):
    """
    Geometric product: a*b = a·b + a∧b.
    Score = |a·b| + 0.1*|a∧b|.
    """
    d = np.clip(embs @ q, -1.0, 1.0)
    theta = np.arccos(d)
    inner = np.abs(d)
    wedge = np.sin(theta)
    return inner + 0.1 * wedge

def sim_rotor_angle(embs, q, qi):
    """
    Rotor: angle between subspaces. Use arccos(cosine) directly.
    """
    d = np.clip(embs @ q, -1.0, 1.0)
    return -np.arccos(d)  # negative because smaller angle = more similar

# ═══════════════════════════════════════════════════════════════
#  CHAOS / ENTROPY MEASURES
# ═══════════════════════════════════════════════════════════════

def sim_tsallis_entropy(embs, q, qi, q_t=2.0):
    """Tsallis entropy of the element-wise product distribution."""
    joint = np.abs(embs * q) + 1e-10
    joint = joint / joint.sum(axis=1, keepdims=True)
    if q_t == 1.0:
        return -np.sum(joint * np.log(joint), axis=1)
    else:
        return (1.0 - np.sum(joint ** q_t, axis=1)) / (q_t - 1.0)

def sim_permutation_entropy(embs, q, qi):
    """Compare rank orderings."""
    ranks_i = np.argsort(np.argsort(embs, axis=1), axis=1)
    ranks_q = np.argsort(np.argsort(q))
    return -np.abs(ranks_i - ranks_q).sum(axis=1).astype(float)

def sim_hurst_exponent(embs, q, qi):
    """Approximate Hurst: variance of cumulative sum differences."""
    cum_i = np.cumsum(embs, axis=1)
    cum_q = np.cumsum(q)
    R = np.max(cum_i, axis=1) - np.min(cum_i, axis=1)
    S = np.std(embs, axis=1) + 1e-10
    return R / S

# ═══════════════════════════════════════════════════════════════
#  HYBRID / ENSEMBLE
# ═══════════════════════════════════════════════════════════════

def sim_ensemble_vote(embs, q, qi):
    """
    Ensemble: cosine + quantum_fidelity + rbf_1.
    Weighted vote.
    """
    c = embs @ q
    qf = c * c
    rbf = np.exp(-np.sum((embs - q)**2, axis=1))
    # Normalize each to [0,1]
    c_n = (c + 1) / 2
    qf_n = qf
    rbf_n = rbf
    return 0.5 * c_n + 0.3 * qf_n + 0.2 * rbf_n

# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("Loading data...")
    embs_raw, C = load_data()
    N = len(embs_raw)

    spaces = {
        "384d_raw": embs_raw,
        "64d_spectral": C,
    }

    print("Computing ground truth...")
    gt_cache = {}
    np.random.seed(42)
    query_idx = np.random.choice(N, NQ, replace=False)
    for qi in query_idx:
        gt_cache[qi] = ground_truth(embs_raw, qi, K)

    functions = [
        ("cosine", sim_cosine),
        ("poly2", sim_poly2),
        ("poly3", sim_poly3),
        ("rbf_1", sim_rbf_1),
        ("rbf_10", sim_rbf_10),
        ("sigmoid", sim_sigmoid),
        ("sqrt_cosine", sim_sqrt_cosine),
        ("exp_cosine", sim_exp_cosine),
        ("alpha_beta", sim_alpha_beta),
        ("sign_agreement", sim_sign_agreement),
        ("top64_dims", lambda e,q,idx: sim_topk_dims(e,q,idx,64)),
        ("top32_dims", lambda e,q,idx: sim_topk_dims(e,q,idx,32)),
        ("l2_inv", sim_l2_inv),
        ("l1_inv", sim_l1_inv),
        # Quantum
        ("quantum_fidelity", sim_quantum_fidelity),
        ("quantum_kernel_1", lambda e,q,idx: sim_quantum_kernel(e,q,idx,1.0)),
        ("quantum_kernel_10", lambda e,q,idx: sim_quantum_kernel(e,q,idx,10.0)),
        ("quantum_js", sim_quantum_js),
        ("quantum_holevo", sim_quantum_holevo),
        ("quantum_bures", sim_quantum_bures),
        ("quantum_helstrom", sim_quantum_helstrom),
        ("quantum_renyi2", sim_quantum_renyi_2),
        ("quantum_tsallis", sim_quantum_tsallis),
        ("quantum_swap", sim_quantum_swap),
        # Information geometry
        ("kl_symmetric", sim_kl_symmetric),
        ("js_softmax", sim_js_softmax),
        ("hellinger", sim_hellinger),
        ("itakura_saito", sim_itakura_saito),
        ("chi_square", sim_chi_square),
        # Spectral / matrix
        ("frobenius_inner", sim_frobenius_inner),
        ("hadamard_ratio", sim_hadamard_ratio),
        ("determinant", sim_determinant),
        ("spectral_angle", sim_spectral_angle),
        # Geometric algebra
        ("wedge_norm", sim_wedge_norm),
        ("inner_plus_wedge", sim_inner_plus_wedge),
        ("rotor_angle", sim_rotor_angle),
        # Chaos / entropy
        ("tsallis_entropy", sim_tsallis_entropy),
        ("permutation_entropy", sim_permutation_entropy),
        ("hurst_exponent", sim_hurst_exponent),
        # Hybrid
        ("ensemble_vote", sim_ensemble_vote),
    ]

    results = []

    for space_name, space_embs in spaces.items():
        print(f"\n{'='*70}")
        print(f"SPACE: {space_name}  (shape: {space_embs.shape})")
        print(f"{'='*70}")

        for name, func in functions:
            recalls = []
            times = []

            for qi in query_idx:
                q = space_embs[qi]
                gt = gt_cache[qi]

                t0 = time.perf_counter()
                scores = func(space_embs, q, qi)
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

            print(f"  {name:22s}  R@10={mean_r:6.2f}%  P50={p50_t:8.2f}µs")

    # Rank
    results.sort(key=lambda x: (-x["r10"], x["p50_us"]))

    print(f"\n{'='*70}")
    print("TOP 15 BY RECALL:")
    print(f"{'='*70}")
    for i, r in enumerate(results[:15], 1):
        print(f"{i:2d}. {r['space']:15s} {r['function']:22s}  R@10={r['r10']:6.2f}%  P50={r['p50_us']:8.2f}µs")

    with open("logs/math_search_quantum.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to logs/math_search_quantum.json")

    # Best per space
    print(f"\n{'='*70}")
    print("BEST PER SPACE:")
    print(f"{'='*70}")
    for space in spaces:
        best = max([r for r in results if r["space"] == space], key=lambda x: x["r10"])
        print(f"  {space:15s}  {best['function']:22s}  R@10={best['r10']}%")

if __name__ == "__main__":
    main()
