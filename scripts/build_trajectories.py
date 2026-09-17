#!/usr/bin/env python3
"""
Build paper trajectories using geometric rotors.
A trajectory is a sequence: paper_0 -> rotor_0->1 -> paper_1 -> ...
"""

import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

EMBS = "data/paper_embeddings_100k.npy"
BASIS = "data/spectral_basis_100k_64.npz"
OUTPUT = "data/trajectories.npz"
TRAJ_LEN = 5
N_TRAJECTORIES = 1000


def compute_rotor(a, b):
    """
    Simple rotor: rotation from a to b in 64-d spectral space.
    Returns the rotation angle and axis direction.
    """
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    dot = np.clip(a @ b, -1.0, 1.0)
    angle = np.arccos(dot)
    # Axis is orthogonal component
    axis = b - dot * a
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    return angle, axis


def main():
    print("Loading...")
    embs = np.load(EMBS).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
    basis = np.load(BASIS)
    V = basis["V"]
    C = embs @ V  # spectral coords

    N = len(C)
    trajectories = []
    metadata = []

    np.random.seed(42)

    for t in range(N_TRAJECTORIES):
        # Random start
        start = np.random.randint(0, N)
        traj_ids = [start]
        traj_angles = []
        traj_axes = []

        current = start
        for step in range(TRAJ_LEN - 1):
            # Find nearest neighbor in spectral space
            dists = 1.0 - (C @ C[current])  # cosine distance
            dists[current] = 999
            next_id = int(np.argmin(dists))

            angle, axis = compute_rotor(C[current], C[next_id])
            traj_angles.append(float(angle))
            traj_axes.append(axis.astype(np.float32))

            traj_ids.append(next_id)
            current = next_id

        trajectories.append({
            "ids": traj_ids,
            "angles": traj_angles,
        })

    # Save compact
    np.savez(OUTPUT,
             ids=np.array([t["ids"] for t in trajectories]),
             angles=np.array([t["angles"] + [0.0] * (TRAJ_LEN - 1 - len(t["angles"])) for t in trajectories]))

    print(f"Built {N_TRAJECTORIES} trajectories of length {TRAJ_LEN}")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
