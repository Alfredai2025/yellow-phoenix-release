#!/usr/bin/env python3
"""
Trajectory search: find papers by matching rotation sequences.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

EMBS = "data/paper_embeddings_100k.npy"
BASIS = "data/spectral_basis_100k_64.npz"
TRAJ = "data/trajectories.npz"


def load():
    embs = np.load(EMBS).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
    basis = np.load(BASIS)
    V = basis["V"]
    C = embs @ V
    traj = np.load(TRAJ)
    return embs, C, traj


def compute_rotor(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    dot = np.clip(a @ b, -1.0, 1.0)
    return np.arccos(dot)


def search_by_trajectory(C, traj_data, query_path_ids, k=5):
    """
    query_path_ids: list of paper IDs representing user's reading history
    Returns: trajectory IDs that match the query rotation sequence
    """
    # Compute query rotors
    query_angles = []
    for i in range(len(query_path_ids) - 1):
        a = C[query_path_ids[i]]
        b = C[query_path_ids[i+1]]
        query_angles.append(compute_rotor(a, b))

    # Match against database trajectories
    traj_angles = traj_data["angles"]  # (N_TRAJ, TRAJ_LEN-1)
    scores = []
    for t in range(len(traj_angles)):
        # Sum of absolute angle differences
        match_len = min(len(query_angles), len(traj_angles[t]))
        diff = sum(abs(query_angles[i] - traj_angles[t][i]) for i in range(match_len))
        scores.append(diff)

    top_traj = np.argsort(scores)[:k]
    return top_traj, [scores[i] for i in top_traj]


def main():
    print("Loading...")
    embs, C, traj = load()

    # Example: simulate "user read papers 0, 100, 200"
    query_path = [0, 100, 200]
    print(f"Query trajectory: {query_path}")

    top_traj, scores = search_by_trajectory(C, traj, query_path, k=5)
    print("\nTop matching trajectories:")
    for i, (tid, score) in enumerate(zip(top_traj, scores), 1):
        traj_ids = traj["ids"][tid]
        print(f"  {i}. Trajectory {tid}: papers {traj_ids[:5]}  score={score:.4f}")


if __name__ == "__main__":
    main()
