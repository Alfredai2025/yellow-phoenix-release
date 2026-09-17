#!/usr/bin/env python3
"""
Diversity test via Flask API so gravity accumulates in the SAME process.
Run this AFTER: python3 scripts/flask_api.py
"""
import requests
import random
import time

API = "http://localhost:5001"
N_SEARCHES = 5000

QUERIES = [
    "neural network", "deep learning", "machine learning",
    "gradient descent", "transformer", "attention mechanism",
    "reinforcement learning", "computer vision",
    "natural language processing", "GAN",
    "medical imaging", "clinical diagnosis", "drug discovery",
    "legal contract", "case law", "regulatory compliance",
    "physics simulation", "quantum computing", "thermodynamics",
    "optimization", "linear programming", "convex optimization",
    "graph neural network", "knowledge graph", "ontology",
    "federated learning", "differential privacy", "edge AI",
    "time series forecasting", "anomaly detection", "clustering",
    "self supervised learning", "contrastive learning",
    "large language model", "prompt engineering",
    "retrieval augmented generation", "semantic search",
    "diffusion model", "generative model",
    "mixture of experts", "sparse model",
    "continual learning", "meta learning",
    "adversarial robustness", "network pruning",
    "physics informed neural network", "scientific machine learning",
    "surrogate model", "Bayesian optimization",
    "active learning", "multi task learning",
    "domain adaptation", "fairness in machine learning",
    "neural network verification", "formal guarantee",
    "neuromorphic computing", "spiking neural network",
]


def main():
    print("=" * 60)
    print("DIVERSITY TEST VIA API")
    print("=" * 60)

    # Check API is up
    try:
        r = requests.get(f"{API}/health", timeout=2)
        print(f"API health: {r.json()}")
    except Exception as e:
        print(f"API not reachable at {API}: {e}")
        print("Start it first: python3 scripts/flask_api.py")
        return

    before = requests.get(f"{API}/health").json().get("dynamic_buckets", 0)
    print(f"\nDynamic buckets BEFORE: {before}")

    print(f"\nRunning {N_SEARCHES} searches via API...")
    t0 = time.time()

    for i in range(N_SEARCHES):
        q = random.choice(QUERIES)
        try:
            requests.post(f"{API}/search", json={"query": q, "top_k": 5}, timeout=2)
        except Exception:
            pass
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{N_SEARCHES} done ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nSearches complete in {elapsed:.1f}s")

    mid = requests.get(f"{API}/health").json().get("dynamic_buckets", 0)
    print(f"Dynamic buckets after searches (before re-bucketing): {mid}")

    # Trigger re-bucketing in the same API process
    print("\nTriggering re-bucketing via API...")
    try:
        rb = requests.post(f"{API}/mesh/rebucket", timeout=10).json()
        print(f"Re-bucket response: {rb}")
    except Exception as e:
        print(f"Re-bucket failed: {e}")
        rb = {}

    # Check after
    after = requests.get(f"{API}/health").json().get("dynamic_buckets", 0)
    print(f"Dynamic buckets AFTER: {after}")
    print(f"Papers moved this run: {after - before}")

    if after > before:
        print(f"\n✅ MESH GROWING: {after - before} new papers moved via API.")
    else:
        print(f"\n⚠️ No new movement. Gravity may need more searches or lower threshold.")

    print("=" * 60)


if __name__ == "__main__":
    main()
