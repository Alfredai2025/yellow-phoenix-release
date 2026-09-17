#!/usr/bin/env python3
"""ZPOR: Zero-Point Observer Retrieval — spectral warm-start experiment."""

import numpy as np
import hnswlib
import json, time

EMBS = "data/paper_embeddings_100k.npy"
BASIS = "data/spectral_basis_100k_64.npz"
NQ = 1000
K = 10


def load():
    embs = np.load(EMBS).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
    basis = np.load(BASIS)
    V = basis["V"]
    C = embs @ V
    return embs, C


def brute_force_nn(embs, q_idx):
    q = embs[q_idx]
    sims = embs @ q
    sims[q_idx] = -np.inf
    return int(np.argmax(sims))


def main():
    print("Loading...")
    embs, C = load()
    N = len(embs)

    print("Building HNSW...")
    index = hnswlib.Index(space='cosine', dim=384)
    index.init_index(max_elements=N, ef_construction=200, M=16)
    index.add_items(embs)
    index.set_ef(25)

    # Extract top-layer nodes by querying with high ef and tracking which nodes appear most
    # Simpler: sample 100 random nodes, assume some are top-layer
    # Even simpler: use the first 100 nodes as "landmarks" and test on them
    landmark_ids = np.arange(0, min(200, N), 2)  # every other node as landmark

    print(f"Testing {NQ} queries...")
    random_wins = 0
    spectral_wins = 0
    ties = 0

    random_distances = []
    spectral_distances = []

    np.random.seed(42)
    query_ids = np.random.choice(N, NQ, replace=False)

    for qi in query_ids:
        true_nn = brute_force_nn(embs, qi)
        q_c = C[qi]

        # Random landmark
        rand_id = int(np.random.choice(landmark_ids))
        # Spectral-nearest landmark
        sims = C[landmark_ids] @ q_c
        spec_id = int(landmark_ids[np.argmax(sims)])

        # Distance from landmark to true NN (in embedding space)
        rand_dist = 1.0 - (embs[rand_id] @ embs[true_nn])  # cosine distance
        spec_dist = 1.0 - (embs[spec_id] @ embs[true_nn])

        random_distances.append(rand_dist)
        spectral_distances.append(spec_dist)

        if spec_dist < rand_dist:
            spectral_wins += 1
        elif rand_dist < spec_dist:
            random_wins += 1
        else:
            ties += 1

    mean_rand = np.mean(random_distances)
    mean_spec = np.mean(spectral_distances)

    print("\n" + "=" * 60)
    print("ZPOR EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Random landmark avg distance to true NN:  {mean_rand:.4f}")
    print(f"Spectral landmark avg distance to true NN: {mean_spec:.4f}")
    print(f"Spectral wins: {spectral_wins} / {NQ} ({spectral_wins/NQ*100:.1f}%)")
    print(f"Random wins:   {random_wins} / {NQ} ({random_wins/NQ*100:.1f}%)")
    print(f"Ties:          {ties}")

    if mean_spec < mean_rand * 0.95:
        print("\n✅ ZPOR WORKS: Spectral entry points are consistently closer.")
        print("   Next: implement full ZPOR query path.")
    elif mean_spec < mean_rand:
        print("\n🟡 ZPOR MARGINAL: Slightly better, but not enough to matter.")
    else:
        print("\n❌ ZPOR FAILS: Random entry points are as good as spectral.")

    result = {
        "mean_random_dist": float(mean_rand),
        "mean_spectral_dist": float(mean_spec),
        "spectral_wins": spectral_wins,
        "random_wins": random_wins,
        "ties": ties,
        "verdict": "works" if mean_spec < mean_rand * 0.95 else ("marginal" if mean_spec < mean_rand else "fails")
    }

    import os
    os.makedirs("logs", exist_ok=True)
    with open("logs/zpor_experiment.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
