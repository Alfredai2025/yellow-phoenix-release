#!/usr/bin/env python3
"""Benchmark YP 1M cascade-on vs cascade-off vs FAISS on identical queries."""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

try:
    import faiss
except ImportError:
    raise SystemExit("pip install faiss-cpu")

from sentence_transformers import SentenceTransformer
from yp_bridge import YPEngine


def pct(arr, p):
    arr = sorted(arr)
    n = len(arr)
    idx = (n - 1) * p
    f = int(idx)
    c = min(f + 1, n - 1)
    return arr[f] + (idx - f) * (arr[c] - arr[f])


def stats(arr):
    return {
        "p50_ms": pct(arr, 0.50),
        "p95_ms": pct(arr, 0.95),
        "p99_ms": pct(arr, 0.99),
        "mean_ms": sum(arr) / len(arr),
        "qps": len(arr) / sum(arr) * 1000,
    }


def main():
    EMB_PATH = "data/paper_embeddings_arxiv_1m.npy"
    NQ = 1000
    K = 10

    print("Loading 1.27M arXiv embeddings...")
    embs = np.load(EMB_PATH).astype("float32")
    N, D = embs.shape
    print(f"  {N:,} vectors × {D} dims")

    print("Building FAISS HNSW index...")
    faiss_index = faiss.IndexHNSWFlat(D, 32)
    faiss_index.hnsw.efConstruction = 200
    faiss_index.add(embs)
    print(f"  Indexed {faiss_index.ntotal:,}")

    print("Loading encoder...")
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

    print(f"Pre-encoding {NQ} queries...")
    qvecs = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype('float32')
    print("  OK")

    print("Booting Yellow Phoenix (1M arXiv DB)...")
    yp = YPEngine(db_path="data/phoenix_arxiv_1m.db")
    print(f"  OK | cache={len(yp.cache):,} | hnsw={len(getattr(yp, '_hnsw', None) or []):,}")

    # Warm up cheat sheet / predictor with all queries once
    print("\nWarming up cascade...")
    for q in texts:
        yp.search_with_sah(q, k=K, use_cascade=True)
    print("  Warm-up complete")

    faiss_times, yp_cascade_times, yp_deep_times = [], [], []

    print(f"\nRunning {NQ} queries @ K={K}...")
    for i in range(NQ):
        # FAISS
        t0 = time.perf_counter()
        _, _ = faiss_index.search(qvecs[i].reshape(1, -1), K)
        faiss_times.append((time.perf_counter() - t0) * 1000)

        # YP cascade ON
        t0 = time.perf_counter()
        _ = yp.search_with_sah(texts[i], k=K, use_cascade=True)
        yp_cascade_times.append((time.perf_counter() - t0) * 1000)

        # YP cascade OFF (deep path)
        t0 = time.perf_counter()
        _ = yp.search_with_sah(texts[i], k=K, use_cascade=False)
        yp_deep_times.append((time.perf_counter() - t0) * 1000)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{NQ}")

    predictor_metrics = {}
    try:
        predictor_metrics = yp.predictor_bridge.metrics()
    except Exception as e:
        predictor_metrics = {"error": str(e)}

    report = {
        "timestamp": datetime.now().isoformat(),
        "n_queries": NQ,
        "k": K,
        "caveat": "All systems index 1.27M arXiv vectors. YP cascade uses cache/beacon/predictor shortcuts.",
        "faiss": {"latency_ms": stats(faiss_times)},
        "yp_cascade_on": {"latency_ms": stats(yp_cascade_times)},
        "yp_cascade_off": {"latency_ms": stats(yp_deep_times)},
        "predictor": predictor_metrics,
    }

    out = Path(f"data/bench_1m_cascade_modes_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Saved: {out}")
    print(f"FAISS          P50: {report['faiss']['latency_ms']['p50_ms']:.3f} ms | QPS: {report['faiss']['latency_ms']['qps']:.1f}")
    print(f"YP cascade ON  P50: {report['yp_cascade_on']['latency_ms']['p50_ms']:.3f} ms | QPS: {report['yp_cascade_on']['latency_ms']['qps']:.1f}")
    print(f"YP cascade OFF P50: {report['yp_cascade_off']['latency_ms']['p50_ms']:.3f} ms | QPS: {report['yp_cascade_off']['latency_ms']['qps']:.1f}")
    print(f"Predictor: {predictor_metrics}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
