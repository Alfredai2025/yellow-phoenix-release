#!/usr/bin/env python3
"""Extract random paper embeddings as synthetic queries for ITQ v2 training."""
import json
import numpy as np

EMB_PATH = "data/paper_embeddings.npy"
OUT_PATH = "data/queries.jsonl"
N_QUERIES = 1000


def main():
    embeddings = np.load(EMB_PATH).astype(np.float32)
    n = len(embeddings)
    print(f"[extract] Loaded embeddings: {embeddings.shape}")

    idx = np.random.choice(n, min(N_QUERIES, n), replace=False)
    with open(OUT_PATH, "w") as f:
        for i in idx:
            q = {
                "paper_id": int(i),
                "text": f"paper_{i}",
                "embedding": embeddings[i].tolist(),
            }
            f.write(json.dumps(q) + "\n")

    print(f"[extract] Wrote {len(idx)} queries to {OUT_PATH}")


if __name__ == "__main__":
    main()
