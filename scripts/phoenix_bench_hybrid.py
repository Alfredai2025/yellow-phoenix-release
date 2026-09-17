#!/usr/bin/env python3
"""Phoenix Bench Hybrid — latency + throughput across three search modes."""
import json
import sys
import time
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yp_bridge import YPEngine
from scripts.sah_cascade import SAHCascade


def pct(sorted_data, p):
    k = (len(sorted_data) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def bench_mode(engine, queries, mode, warmup=100, timed=1000):
    for q in queries[:warmup]:
        if mode == "pure_paper":
            engine.search_with_sah(q, k=10, use_cascade=False, use_ghosts=False)
        elif mode == "beacon_only":
            h = engine._hash_query(q)
            SAHCascade(engine.rust, None).query(h, k=10)
        else:
            engine.search_with_sah(q, k=10, use_cascade=True, use_ghosts=True)

    times = []
    for q in queries[:timed]:
        t0 = time.perf_counter()
        if mode == "pure_paper":
            engine.search_with_sah(q, k=10, use_cascade=False, use_ghosts=False)
        elif mode == "beacon_only":
            h = engine._hash_query(q)
            SAHCascade(engine.rust, None).query(h, k=10)
        else:
            engine.search_with_sah(q, k=10, use_cascade=True, use_ghosts=True)
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return {
        "mode": mode,
        "queries": len(times),
        "p50_ms": pct(times, 0.50),
        "p95_ms": pct(times, 0.95),
        "p99_ms": pct(times, 0.99),
        "mean_ms": statistics.mean(times),
        "qps": len(times) / sum(times) * 1000,
    }


def main():
    engine = YPEngine()
    pool = [
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

    rows = []
    for mode in ("pure_paper", "beacon_only", "full_hybrid"):
        print(f"\n=== {mode} ===")
        r = bench_mode(engine, pool, mode)
        rows.append(r)
        print(f"  P50 {r['p50_ms']:.3f} ms | P95 {r['p95_ms']:.3f} ms | QPS {r['qps']:.1f}")

    out = Path(f"data/bench_hybrid_{datetime.now():%Y%m%d_%H%M%S}.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump({"ts": datetime.now().isoformat(), "modes": rows}, f, indent=2)
    print(f"\nSaved: {out}")

    print("\n| Mode | P50 (ms) | P95 (ms) | P99 (ms) | QPS |")
    print("|------|----------|----------|----------|-----|")
    for r in rows:
        print(f"| {r['mode']} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} | {r['p99_ms']:.3f} | {r['qps']:.1f} |")


if __name__ == "__main__":
    main()
