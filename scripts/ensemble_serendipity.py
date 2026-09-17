#!/usr/bin/env python3
"""
Final test: Serendipity (47° peak) + Cosine ensemble.
Serendipity expands candidate pool; cosine re-ranks.
"""
import numpy as np
import os


def load_embeddings():
    for name in ['paper_embeddings_100k.npy', 'paper_embeddings.npy']:
        p = os.path.expanduser(f'~/yellow_phoenix/data/{name}')
        if os.path.exists(p):
            return np.load(p).astype(np.float32)
    raise FileNotFoundError


def serendipity_score(q5, P5, w0, w2):
    """Vectorized 47° peak score over all candidates P5 (n,5)."""
    dots = P5 @ q5  # (n,)
    bivectors = np.sqrt(np.maximum(0.0, 1.0 - dots**2))
    return w0 * np.abs(dots) + w2 * bivectors


def main():
    print("=" * 60)
    print("ENSEMBLE: Serendipity (47°) + Cosine")
    print("=" * 60)

    emb = load_embeddings()
    n = len(emb)
    print(f"Embeddings: {n}")

    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=5, svd_solver='randomized')
    pca.fit(emb_norm[:5000])
    W = pca.components_  # (5, 384)

    # Project all to 5-D and normalize
    emb5 = emb_norm @ W.T
    emb5 = emb5 / (np.linalg.norm(emb5, axis=1, keepdims=True) + 1e-8)

    # Unconstrained weights from tge_prototype.py run
    w0, w2 = 1.639, 1.782

    np.random.seed(42)
    n_test = 200

    cos_r1 = cos_r5 = cos_r100 = 0
    ens_r1 = ens_r5 = ens_r100 = 0
    ser_r1 = ser_r5 = 0

    for _ in range(n_test):
        q_idx = np.random.randint(0, n)
        q = emb_norm[q_idx]
        q5 = emb5[q_idx]

        # Ground truth: cosine nearest neighbor (excluding self)
        cos_sims = emb_norm @ q
        cos_sims[q_idx] = -999
        gt = np.argsort(cos_sims)[::-1][0]

        # Baseline cosine top-100
        cos_top100 = np.argsort(cos_sims)[::-1][:100]
        if cos_top100[0] == gt:
            cos_r1 += 1
        if gt in cos_top100[:5]:
            cos_r5 += 1
        if gt in cos_top100:
            cos_r100 += 1

        # Serendipity top-100
        ser_scores = serendipity_score(q5, emb5, w0, w2)
        ser_scores[q_idx] = -999
        ser_top100 = np.argsort(ser_scores)[::-1][:100]

        if ser_top100[0] == gt:
            ser_r1 += 1
        if gt in ser_top100[:5]:
            ser_r5 += 1

        # Ensemble: cosine re-rank serendipity top-100
        ens_candidates = emb_norm[ser_top100]
        ens_sims = ens_candidates @ q
        ens_ranked = ser_top100[np.argsort(ens_sims)[::-1]]

        if ens_ranked[0] == gt:
            ens_r1 += 1
        if gt in ens_ranked[:5]:
            ens_r5 += 1
        if gt in ens_ranked:
            ens_r100 += 1

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'R@1':>8} {'R@5':>8} {'R@100':>8}")
    print("-" * 60)
    print(f"{'Cosine only':<20} {cos_r1/n_test*100:>7.1f}% {cos_r5/n_test*100:>7.1f}% {cos_r100/n_test*100:>7.1f}%")
    print(f"{'Serendipity only':<20} {ser_r1/n_test*100:>7.1f}% {ser_r5/n_test*100:>7.1f}% {'N/A':>8}")
    print(f"{'Ensemble (S+C)':<20} {ens_r1/n_test*100:>7.1f}% {ens_r5/n_test*100:>7.1f}% {ens_r100/n_test*100:>7.1f}%")

    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    if ens_r100 > cos_r100:
        print("✅ ENSEMBLE WINS — Serendipity finds candidates cosine misses!")
        print("   The 47° peak has value as a recall expander.")
    elif ens_r100 == cos_r100:
        print("⚠️  TIE — Serendipity finds nothing new.")
    else:
        print("❌ COSINE WINS — Serendipity pollutes the candidate pool.")
        print("   The 47° peak is noise, not signal.")

    if ens_r1 > cos_r1:
        print("   (And ensemble improves precision too!)")
    elif ens_r1 < cos_r1:
        print("   (But ensemble hurts precision — serendipity adds false positives.)")


if __name__ == '__main__':
    main()
