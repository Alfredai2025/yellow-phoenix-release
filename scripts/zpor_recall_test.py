#!/usr/bin/env python3
"""ZPOR recall test — does the spectral landmark help HNSW?"""

import numpy as np
import hnswlib, json, time

EMBS = "data/paper_embeddings_100k.npy"
BASIS = "data/spectral_basis_100k_64.npz"
NQ = 1000
K = 10
EF = 25
LANDMARK_STEP = 100  # every 100th node = 1K landmarks

def load():
    embs = np.load(EMBS).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
    basis = np.load(BASIS)
    V = basis["V"]
    C = embs @ V
    return embs, C

def brute_top10(embs, q_idx):
    q = embs[q_idx]
    sims = embs @ q
    sims[q_idx] = -np.inf
    return set(np.argpartition(sims, -K)[-K:])

def main():
    print("Loading...")
    embs, C = load()
    N = len(embs)
    
    landmarks = np.arange(0, N, LANDMARK_STEP)
    print(f"Landmarks: {len(landmarks)}")
    
    print("Building HNSW...")
    idx = hnswlib.Index(space='cosine', dim=384)
    idx.init_index(max_elements=N, ef_construction=200, M=16)
    idx.add_items(embs)
    idx.set_ef(EF)
    
    np.random.seed(42)
    queries = np.random.choice(N, NQ, replace=False)
    
    base_recalls = []
    zpor_recalls = []
    
    t0 = time.time()
    for qi in queries:
        truth = brute_top10(embs, qi)
        
        # Baseline
        labels, _ = idx.knn_query(embs[qi], k=EF)
        base_set = set(labels[0])
        base_recalls.append(len(base_set & truth) / K)
        
        # ZPOR: add spectral landmark to candidate pool
        q_c = C[qi]
        sims = C[landmarks] @ q_c
        best_land = int(landmarks[np.argmax(sims)])
        zpor_set = base_set | {best_land}
        zpor_recalls.append(len(zpor_set & truth) / K)
    
    elapsed = time.time() - t0
    
    print("\n" + "="*60)
    print("ZPOR RECALL TEST")
    print("="*60)
    print(f"Baseline R@{K}:  {np.mean(base_recalls)*100:.2f}%")
    print(f"ZPOR R@{K}:      {np.mean(zpor_recalls)*100:.2f}%")
    print(f"Delta:           {(np.mean(zpor_recalls)-np.mean(base_recalls))*100:+.2f}pp")
    print(f"Time:            {elapsed:.1f}s ({elapsed/NQ*1000:.1f} ms/query)")
    
    result = {
        "baseline_r10": float(np.mean(base_recalls)),
        "zpor_r10": float(np.mean(zpor_recalls)),
        "delta_pp": float((np.mean(zpor_recalls)-np.mean(base_recalls))*100),
        "verdict": "wins" if np.mean(zpor_recalls) > np.mean(base_recalls) + 0.005 else "flat"
    }
    with open("logs/zpor_recall.json","w") as f:
        json.dump(result, f, indent=2)
    
    if result["verdict"] == "wins":
        print("\n✅ ZPOR improves recall. Build the hybrid path.")
    else:
        print("\n❌ ZPOR doesn't translate. Commit pilot as future work.")

if __name__ == "__main__":
    main()
