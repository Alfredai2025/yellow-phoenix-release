#!/usr/bin/env python3
"""
Benchmark: trajectory search (rotor-based) vs cosine baseline.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.trajectory_search import load, search_by_trajectory, compute_rotor


def cosine_baseline(embs, query_path, k=5):
    """Just find nearest neighbor to last paper in path."""
    last = embs[query_path[-1]]
    sims = embs @ last
    sims[query_path] = -np.inf  # exclude seen
    top = np.argpartition(-sims, k)[:k]
    return top[np.argsort(-sims[top])]


def trajectory_search(C, traj, query_path, k=5):
    top_traj, _ = search_by_trajectory(C, traj, query_path, k=1)
    # Return papers from best matching trajectory, excluding seen
    traj_ids = traj["ids"][top_traj[0]]
    seen = set(query_path)
    return [i for i in traj_ids if i not in seen][:k]


def main():
    embs, C, traj = load()
    N = len(embs)

    # Synthetic test: 50 random 3-paper paths
    np.random.seed(42)
    n_test = 50
    path_len = 3

    traj_wins = 0
    cosine_wins = 0
    ties = 0

    for t in range(n_test):
        path = np.random.choice(N, path_len, replace=False).tolist()

        # "Ground truth": what is the true next paper?
        # Approximate: nearest neighbor to last paper in path
        true_next = int(np.argmax(embs @ embs[path[-1]]))
        true_next = [i for i in range(N) if i not in path][0]  # simplistic

        traj_results = trajectory_search(C, traj, path, k=5)
        cos_results = cosine_baseline(embs, path, k=5)

        # Check if true_next is in top-5
        traj_hit = true_next in traj_results
        cos_hit = true_next in cos_results

        if traj_hit and not cos_hit:
            traj_wins += 1
        elif cos_hit and not traj_hit:
            cosine_wins += 1
        else:
            ties += 1

    print(f"\nTrajectory wins: {traj_wins}/{n_test}")
    print(f"Cosine wins:     {cosine_wins}/{n_test}")
    print(f"Ties:            {ties}/{n_test}")

    if traj_wins > cosine_wins:
        print("\n✅ TRAJECTORY SEARCH WINS")
    else:
        print("\n❌ COSINE WINS (trajectory search not better)")


if __name__ == "__main__":
    main()
