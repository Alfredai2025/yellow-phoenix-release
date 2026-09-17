#!/usr/bin/env python3
"""1M Burn-in: YP vs FAISS — fixed batch encoding, honest about 13K vs 1M."""
import json, sys, time
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import faiss
except ImportError:
    raise SystemExit("pip install faiss-cpu")

from yp_bridge import YPEngine


def pct(arr, p):
    arr = sorted(arr)
    n = len(arr)
    idx = (n - 1) * p
    f = int(idx)
    c = min(f + 1, n - 1)
    return arr[f] + (idx - f) * (arr[c] - arr[f])


def main():
    EMB_PATH = "data/paper_embeddings_arxiv_1m.npy"
    NQ = 1000
    K = 10

    print("Loading 1.27M arXiv embeddings...")
    embs = np.load(EMB_PATH).astype("float32")
    N, D = embs.shape
    print(f"  {N:,} vectors × {D} dims")

    # Build FAISS
    print("Building FAISS HNSW index...")
    faiss_index = faiss.IndexHNSWFlat(D, 32)
    faiss_index.hnsw.efConstruction = 200
    faiss_index.add(embs)
    print(f"  Indexed {faiss_index.ntotal:,}")

    # Load encoder ONCE, batch-pre-encode all queries
    print("Loading encoder...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    print("  OK")

    POOL = [
        "machine learning", "neural network", "computer vision",
        "natural language processing", "deep learning", "reinforcement learning",
        "transformer architecture", "graph neural network", "attention mechanism",
        "large language model", "semantic search", "information retrieval",
        "vector database", "approximate nearest neighbor", "embedding space",
        "clustering algorithm", "dimensionality reduction", "principal component analysis",
        "support vector machine", "random forest", "gradient boosting",
        "bayesian optimization", "hyperparameter tuning", "model compression",
        "knowledge distillation", "quantization aware training", "federated learning",
        "differential privacy", "adversarial robustness", "out of distribution detection",
        "self supervised learning", "contrastive learning", "masked autoencoder",
        "vision transformer", "multimodal learning", "cross modal retrieval",
        "text to image generation", "diffusion model", "generative adversarial network",
        "variational autoencoder", "normalizing flow", "energy based model",
        "neural radiance field", "3d reconstruction", "point cloud processing",
        "sensor fusion", "simultaneous localization and mapping", "autonomous driving",
        "robotics manipulation", "reinforcement learning from human feedback",
        "inverse reinforcement learning", "imitation learning", "meta learning",
        "few shot learning", "zero shot learning", "continual learning",
        "lifelong learning", "catastrophic forgetting", "elastic weight consolidation",
        "progressive neural network", "modular neural network", "neural architecture search",
        "automated machine learning", "hyperband", "population based training",
        "evolutionary strategy", "genetic algorithm", "particle swarm optimization",
        "simulated annealing", "tabu search", "ant colony optimization",
        "monte carlo tree search", "upper confidence bound", "thompson sampling",
        "multi armed bandit", "contextual bandit", "markov decision process",
        "partially observable markov decision process", "dynamic programming",
        "value iteration", "policy iteration", "q learning", "sarsa",
        "deep q network", "actor critic", "proximal policy optimization",
        "trust region policy optimization", "soft actor critic", "twin delayed deep deterministic",
        "policy gradient", "actor critic method", "deterministic policy gradient",
        "generalized advantage estimation", "temporal difference learning",
        "monte carlo method", "n step return", "lambda return",
        "eligibility traces", "function approximation", "tile coding",
        "radial basis function", "kernel method", "gaussian process",
        "bayesian neural network", "monte carlo dropout", "variational inference",
        "expectation propagation", "belief propagation", "loopy belief propagation",
        "mean field approximation", "variational auto encoder",
    ] * 10

    texts = [POOL[i % len(POOL)] for i in range(NQ)]

    # Batch-pre-encode all query vectors
    print(f"Pre-encoding {NQ} queries in one batch...")
    qvecs = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype('float32')
    print("  OK")

    # Boot YP on 1M arXiv DB
    print("Booting Yellow Phoenix (1M arXiv DB)...")
    yp = YPEngine(db_path="data/phoenix_arxiv_1m.db")
    print(f"  OK | cache={len(yp.cache):,} | hnsw={len(getattr(yp, '_hnsw', None) or []):,}")

    # Latency benchmark
    yp_times, faiss_times = [], []
    print(f"\nRunning {NQ} queries @ K={K}...")

    for i in range(NQ):
        # FAISS
        t0 = time.perf_counter()
        _, _ = faiss_index.search(qvecs[i].reshape(1, -1), K)
        faiss_times.append((time.perf_counter() - t0) * 1000)

        # YP
        t0 = time.perf_counter()
        _ = yp.search_with_sah(texts[i], k=K, use_cascade=False, use_ghosts=False)
        yp_times.append((time.perf_counter() - t0) * 1000)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{NQ}")

    def stats(arr):
        return {"p50_ms": pct(arr, 0.50), "p95_ms": pct(arr, 0.95),
                "p99_ms": pct(arr, 0.99), "mean_ms": sum(arr)/len(arr),
                "qps": len(arr) / sum(arr) * 1000}

    report = {
        "timestamp": datetime.now().isoformat(),
        "n_queries": NQ,
        "k": K,
        "caveat": "Both YP and FAISS index 1.27M arXiv vectors. Latency comparable. Recall measured separately.",
        "yellow": {"latency_ms": stats(yp_times)},
        "faiss": {"latency_ms": stats(faiss_times)},
    }

    out = Path(f"data/bench_1m_vs_faiss_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*55}")
    print(f"Saved: {out}")
    print(f"YP      P50: {report['yellow']['latency_ms']['p50_ms']:.3f} ms | QPS: {report['yellow']['latency_ms']['qps']:.1f}")
    print(f"FAISS   P50: {report['faiss']['latency_ms']['p50_ms']:.3f} ms | QPS: {report['faiss']['latency_ms']['qps']:.1f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
