#!/usr/bin/env python3
"""
MDTC: Math Decision Tree Cascade
Each gate uses a different geometry to Accept/Reject/Pass.
Reverse math (reconstruction) skips gates when confident.
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


def gate_hyperbolic(q, p):
    u = q * 0.99 / (np.linalg.norm(q) + 1e-8)
    v = p * 0.99 / (np.linalg.norm(p) + 1e-8)
    u_sq = np.sum(u**2)
    v_sq = np.sum(v**2)
    diff_sq = np.sum((u - v)**2)
    num = 2 * diff_sq
    denom = (1 - u_sq) * (1 - v_sq) + 1e-8
    return np.arccosh(1 + num / denom)


def gate_grassmannian(q, p):
    dot = np.abs(np.dot(q, p))
    return np.arccos(np.clip(dot, 0, 1))


def gate_rbf(q, p, gamma=0.1):
    return np.exp(-gamma * np.sum((q - p)**2))


def gate_cosine(q, p):
    return np.dot(q, p)


def reconstruct_from_hyperbolic(d):
    return max(0.0, min(1.0, 1.0 - d * 0.45))


def reconstruct_from_grassmannian(angle):
    return max(0.0, min(1.0, 1.0 - 2.0 * angle / np.pi))


def reconstruct_from_rbf(sim):
    return sim


class MDTC:
    def __init__(self, thresholds=None, confidence=0.90):
        if thresholds is None:
            thresholds = {
                'hyperbolic': (0.05, 1.50),
                'grassmannian': (0.15, 1.20),
                'rbf': (0.92, 0.40),
            }
        self.thresholds = thresholds
        self.confidence = confidence
        self.stats = {k: 0 for k in [
            'hyperbolic_accept', 'hyperbolic_reject', 'hyperbolic_pass',
            'grassmannian_accept', 'grassmannian_reject', 'grassmannian_pass',
            'rbf_accept', 'rbf_reject', 'rbf_pass', 'cosine_final'
        ]}

    def reset_stats(self):
        for k in self.stats:
            self.stats[k] = 0

    def query_one(self, q_emb, p_emb):
        path = []

        # Gate 0: Hyperbolic
        d_hyp = gate_hyperbolic(q_emb, p_emb)
        path.append(f"H:{d_hyp:.3f}")

        low, high = self.thresholds['hyperbolic']
        if d_hyp < low:
            recon = reconstruct_from_hyperbolic(d_hyp)
            if recon > self.confidence:
                self.stats['hyperbolic_accept'] += 1
                return recon, 1, path + ["FAST_ACCEPT"]
        if d_hyp > high:
            recon = reconstruct_from_hyperbolic(d_hyp)
            if recon < (1.0 - self.confidence):
                self.stats['hyperbolic_reject'] += 1
                return recon, 1, path + ["FAST_REJECT"]
        self.stats['hyperbolic_pass'] += 1

        # Gate 1: Grassmannian
        ang_grass = gate_grassmannian(q_emb, p_emb)
        path.append(f"G:{ang_grass:.3f}")

        low, high = self.thresholds['grassmannian']
        if ang_grass < low:
            recon = reconstruct_from_grassmannian(ang_grass)
            if recon > self.confidence:
                self.stats['grassmannian_accept'] += 1
                return recon, 2, path + ["FAST_ACCEPT"]
        if ang_grass > high:
            recon = reconstruct_from_grassmannian(ang_grass)
            if recon < (1.0 - self.confidence):
                self.stats['grassmannian_reject'] += 1
                return recon, 2, path + ["FAST_REJECT"]
        self.stats['grassmannian_pass'] += 1

        # Gate 2: RBF
        sim_rbf = gate_rbf(q_emb, p_emb)
        path.append(f"R:{sim_rbf:.3f}")

        low, high = self.thresholds['rbf']  # (accept_above, reject_below)
        if sim_rbf > low:
            self.stats['rbf_accept'] += 1
            return sim_rbf, 3, path + ["ACCEPT"]
        if sim_rbf < high:
            self.stats['rbf_reject'] += 1
            return sim_rbf, 3, path + ["REJECT"]
        self.stats['rbf_pass'] += 1

        # Gate 3: Full Cosine
        sim_cos = gate_cosine(q_emb, p_emb)
        self.stats['cosine_final'] += 1
        return sim_cos, 4, path + [f"C:{sim_cos:.3f}"]

    def retrieve(self, q_idx, embeddings, top_k=5):
        q = embeddings[q_idx]
        n = len(embeddings)
        scores = []
        for i in range(n):
            if i == q_idx:
                continue
            score, gates, path = self.query_one(q, embeddings[i])
            scores.append((score, i, gates, path))
        scores.sort(reverse=True)
        return scores[:top_k]


def evaluate_mdtc(embeddings, n_test=200, top_k=5):
    n = len(embeddings)
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    mdtc = MDTC()

    cos_r1 = mdtc_r1 = cos_r5 = mdtc_r5 = 0
    total_gates_mdtc = 0
    stats_acc = {k: 0 for k in mdtc.stats}

    t0 = time.time()
    for _ in range(n_test):
        q_idx = np.random.randint(0, n)

        # Ground truth
        cos_sims = emb_norm @ emb_norm[q_idx]
        cos_sims[q_idx] = -999
        gt_ranking = np.argsort(cos_sims)[::-1]
        gt_top1 = gt_ranking[0]
        gt_top5 = set(gt_ranking[:5])

        # MDTC retrieve
        mdtc.reset_stats()
        results = mdtc.retrieve(q_idx, emb_norm, top_k=top_k)
        mdtc_top1 = results[0][1]
        mdtc_top5 = set(r[1] for r in results)

        if mdtc_top1 == gt_top1:
            mdtc_r1 += 1
        if gt_top1 in mdtc_top5:
            mdtc_r5 += 1
        if gt_top1 == gt_top1:
            cos_r1 += 1
        if gt_top1 in gt_top5:
            cos_r5 += 1

        total_gates_mdtc += np.mean([r[2] for r in results])
        for k in mdtc.stats:
            stats_acc[k] += mdtc.stats[k]

    elapsed = time.time() - t0

    return {
        'cos_r1': cos_r1 / n_test,
        'mdtc_r1': mdtc_r1 / n_test,
        'cos_r5': cos_r5 / n_test,
        'mdtc_r5': mdtc_r5 / n_test,
        'avg_gates_mdtc': total_gates_mdtc / n_test,
        'time': elapsed,
        'stats': stats_acc,
    }


def main():
    print("=" * 70)
    print("MDTC: Math Decision Tree Cascade")
    print("=" * 70)

    emb = load_embeddings()
    n = len(emb)
    print(f"Embeddings: {n} x {emb.shape[1]}")

    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    n_sample = min(2000, n)
    sample_idx = np.random.choice(n, n_sample, replace=False)
    emb_s = emb_norm[sample_idx]
    print(f"Sample: {n_sample} papers")

    print("\nTraining thresholds on 100 random pairs...")
    pairs = []
    for _ in range(100):
        i, j = np.random.randint(0, n_sample, 2)
        if i != j:
            pairs.append((i, j))

    hyp_dists = [gate_hyperbolic(emb_s[i], emb_s[j]) for i, j in pairs]
    grass_angles = [gate_grassmannian(emb_s[i], emb_s[j]) for i, j in pairs]
    rbf_sims = [gate_rbf(emb_s[i], emb_s[j]) for i, j in pairs]
    cos_sims = [gate_cosine(emb_s[i], emb_s[j]) for i, j in pairs]

    sorted_idx = np.argsort(cos_sims)
    n_pairs = len(sorted_idx)
    bad_idx = sorted_idx[:n_pairs // 5]
    good_idx = sorted_idx[-n_pairs // 5:]

    hyp_good = max([hyp_dists[i] for i in good_idx])
    hyp_bad = min([hyp_dists[i] for i in bad_idx])
    grass_good = max([grass_angles[i] for i in good_idx])
    grass_bad = min([grass_angles[i] for i in bad_idx])
    rbf_good = min([rbf_sims[i] for i in good_idx])
    rbf_bad = max([rbf_sims[i] for i in bad_idx])

    print(f"  Hyperbolic:   accept < {hyp_good:.3f}, reject > {hyp_bad:.3f}")
    print(f"  Grassmannian: accept < {grass_good:.3f}, reject > {grass_bad:.3f}")
    print(f"  RBF:          accept > {rbf_good:.3f}, reject < {rbf_bad:.3f}")

    mdtc = MDTC(thresholds={
        'hyperbolic': (hyp_good, hyp_bad),
        'grassmannian': (grass_good, grass_bad),
        'rbf': (rbf_good, rbf_bad),
    })

    print("\nEvaluating on 200 queries...")
    results = evaluate_mdtc(emb_s, n_test=200, top_k=5)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Cosine R@1:    {results['cos_r1']*100:.1f}%")
    print(f"MDTC R@1:      {results['mdtc_r1']*100:.1f}%")
    print(f"Cosine R@5:    {results['cos_r5']*100:.1f}%")
    print(f"MDTC R@5:      {results['mdtc_r5']*100:.1f}%")
    print(f"\nAvg gates per top-5 candidate: {results['avg_gates_mdtc']:.2f}")
    print(f"Time: {results['time']:.1f}s")

    print(f"\n{'='*70}")
    print("GATE STATISTICS (accumulated)")
    print(f"{'='*70}")
    for k, v in results['stats'].items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    np.random.seed(42)
    main()
