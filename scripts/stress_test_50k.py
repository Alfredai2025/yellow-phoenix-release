#!/usr/bin/env python3
"""
Stress test: 50,000 searches on 12K-paper mesh.
Measures latency, tracks mesh movement, checks for degradation.
Run while Flask API is live on :5001.
"""
import requests
import random
import time
import statistics
import sys

API = "http://localhost:5001"
N_SEARCHES = 50_000
REBUCKET_EVERY = 10_000

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
    "recursive self modelling", "autopoiesis", "self organization",
    "predictive processing", "free energy principle", "embodied cognition",
    "consciousness", "qualia", "mind body problem",
    "language model reasoning", "chain of thought", "in context learning",
    "tool use", "agent architecture", "multi agent",
    "swarm intelligence", "collective intelligence", "emergence",
    "information theory", "entropy", "compression",
    "causal inference", "do calculus", "structural equation",
    "bayesian network", "probabilistic graphical model",
    "markov chain", "monte carlo", "mcmc",
    "stochastic process", "random walk", "brownian motion",
    "dynamical system", "chaos theory", "attractor",
    "complexity", "network science", "small world",
    "scale free network", "preferential attachment", "centrality",
    "community detection", "modularity", "spectral clustering",
    "dimensionality reduction", "manifold learning", "t-SNE",
    "principal component analysis", "independent component analysis",
    "factor analysis", "latent variable model",
    "hidden markov model", "state space model", "kalman filter",
    "particle filter", "sequential monte carlo",
    "gaussian process", "kernel method", "support vector machine",
    "decision tree", "random forest", "gradient boosting",
    "xgboost", "lightgbm", "catboost",
    "ensemble method", "model stacking", "blending",
    "hyperparameter optimization", "bayesian optimization", "optuna",
    "neural architecture search", "autoML", "meta learning",
    "few shot learning", "zero shot learning", "prompt tuning",
    "instruction tuning", "rlhf", "constitutional ai",
    "alignment", "safety", "interpretability",
    "mechanistic interpretability", "feature visualization", "activation patching",
    "probing", "representation learning", "disentanglement",
    "contrastive learning", "self supervision", "pretext task",
    "data augmentation", "mixup", "cutmix",
    "adversarial training", "robustness", "certified defense",
    "differential privacy", "federated learning", "secure aggregation",
    "homomorphic encryption", "secure multi party computation",
    "zk snark", "blockchain", "smart contract",
    "decentralized identity", "verifiable credential", "self sovereign",
    "web3", "dao", "tokenomics",
    "game theory", "mechanism design", "auction theory",
    "incentive alignment", "public good", "free rider",
    "social choice", "voting theory", "arrow theorem",
    "fair division", "envy freeness", "proportional allocation",
    "matching market", "stable marriage", "deferred acceptance",
    "market design", "school choice", "kidney exchange",
    "recommendation system", "collaborative filtering", "matrix factorization",
    "content based filtering", "hybrid recommendation", "session based",
    "sequential recommendation", "next basket prediction", "click through rate",
    "search ranking", "learning to rank", "listwise",
    "document retrieval", "passage retrieval", "dense retrieval",
    "sparse retrieval", "late interaction", "colbert",
    "re ranking", "cross encoder", "bi encoder",
    "knowledge distillation", "teacher student", "soft label",
    "quantization", "pruning", "knowledge transfer",
    "model compression", "mobile deployment", "edge inference",
    "tensorrt", "onnx", "openvino",
    "fpga acceleration", "asic design", "neural accelerator",
    "photonic computing", "optical neural network", "neuromorphic chip",
    "brain computer interface", "neural decoding", "neural encoding",
    "cognitive enhancement", "nootropic", "transhumanism",
    "longevity", "aging biology", "senolytic",
    "regenerative medicine", "stem cell", "tissue engineering",
    "synthetic biology", "gene editing", "crispr",
    "bioinformatics", "computational biology", "systems biology",
    "protein folding", "alphafold", "structure prediction",
    "molecular dynamics", "drug design", "virtual screening",
    "materials science", "battery chemistry", "solid state",
    "perovskite", "photovoltaic", "solar cell",
    "carbon capture", "climate model", "earth system",
    "oceanography", "atmospheric science", "weather prediction",
    "seismology", "earthquake prediction", "tsunami warning",
    "volcanology", "magma dynamics", "eruption forecasting",
    "planetary science", "exoplanet", "habitability",
    "astrobiology", " SETI", "technosignature",
    "cosmology", "dark matter", "dark energy",
    "gravitational wave", "ligo", "black hole",
    "neutron star", "pulsar", "fast radio burst",
    "gamma ray burst", "supernova", "nucleosynthesis",
    "quantum gravity", "string theory", "loop quantum gravity",
    "holographic principle", "ads cft", "gauge gravity",
    "condensed matter", "topological insulator", "superconductor",
    "high temperature superconductor", "meissner effect", "vortex",
    "quantum hall effect", "fractional statistics", "anyon",
    "spin liquid", "quantum spin ice", "frustrated magnet",
    "cold atom", "bose einstein condensate", "quantum simulation",
    "ion trap", "superconducting qubit", "photonic qubit",
    "quantum error correction", "surface code", "logical qubit",
    "quantum supremacy", "quantum advantage", "nisq",
    "variational quantum algorithm", "qaoa", "vqe",
    "quantum machine learning", "quantum neural network", "quantum kernel",
    "quantum sensing", "quantum metrology", "quantum imaging",
    "quantum communication", "quantum key distribution", "quantum internet",
    "quantum cryptography", "post quantum cryptography", "lattice based",
    "code based cryptography", "multivariate cryptography", "hash based",
    "isogeny based", "supersingular", "sidh",
]


def run_searches(n, session, api):
    """Run n searches, return latencies and any errors."""
    latencies = []
    errors = 0
    t_start = time.time()

    for i in range(n):
        q = random.choice(QUERIES)
        t0 = time.time()
        try:
            r = session.post(f"{api}/search", json={"query": q, "top_k": 5}, timeout=5)
            if r.status_code == 200:
                latencies.append((time.time() - t0) * 1000)
            else:
                errors += 1
        except Exception:
            errors += 1

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t_start
            print(f"  {i+1}/{n} done ({elapsed:.1f}s, {errors} errors)")

    return latencies, errors


def trigger_rebucket(session, api):
    try:
        r = session.post(f"{api}/mesh/rebucket", timeout=30)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        print(f"  Re-bucket failed: {e}")
        return None


def get_health(session, api):
    try:
        return session.get(f"{api}/health", timeout=2).json()
    except Exception:
        return {}


def main():
    print("=" * 70)
    print("STRESS TEST: 50,000 searches on 12K-paper mesh")
    print("=" * 70)

    session = requests.Session()

    # Verify API
    health = get_health(session, API)
    if not health:
        print("API not reachable. Start: python3 scripts/flask_api.py")
        return

    before = health.get("dynamic_buckets", 0)
    print(f"\nDynamic buckets BEFORE: {before}")
    print(f"Target searches: {N_SEARCHES}")
    print(f"Re-bucketing every: {REBUCKET_EVERY}")

    all_latencies = []
    total_errors = 0
    t0 = time.time()

    # Run in chunks with periodic re-bucketing
    for chunk in range(0, N_SEARCHES, REBUCKET_EVERY):
        chunk_size = min(REBUCKET_EVERY, N_SEARCHES - chunk)
        print(f"\n--- Chunk {chunk//REBUCKET_EVERY + 1}: {chunk_size} searches ---")

        lats, errs = run_searches(chunk_size, session, API)
        all_latencies.extend(lats)
        total_errors += errs

        # Re-bucket
        print("  Triggering re-bucketing...")
        rb = trigger_rebucket(session, API)
        if rb:
            print(f"  Re-bucket: {rb}")

    total_time = time.time() - t0

    # Final health
    after = get_health(session, API).get("dynamic_buckets", 0)

    # Stats
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Total searches:     {N_SEARCHES}")
    print(f"  Total time:         {total_time:.1f}s")
    print(f"  Avg throughput:     {N_SEARCHES/total_time:.0f} searches/sec")
    print(f"  Errors:             {total_errors}")
    print(f"  Dynamic buckets:    {after} (moved {after - before} total)")

    if all_latencies:
        print(f"\n  Latency (ms):")
        print(f"    Min:    {min(all_latencies):.2f}")
        print(f"    Max:    {max(all_latencies):.2f}")
        print(f"    Mean:   {statistics.mean(all_latencies):.2f}")
        print(f"    Median: {statistics.median(all_latencies):.2f}")
        print(f"    P95:    {sorted(all_latencies)[int(len(all_latencies)*0.95)]:.2f}")
        print(f"    P99:    {sorted(all_latencies)[int(len(all_latencies)*0.99)]:.2f}")

    # Verdict
    print(f"\n{'='*70}")
    if total_errors == 0 and statistics.median(all_latencies) < 5:
        print("  ✅ STABLE: API handled 50K searches without errors.")
        print("  ✅ FAST: Median latency under 5 ms.")
    elif total_errors > 100:
        print("  ❌ UNSTABLE: Too many errors. Check API logs.")
    else:
        print("  ⚠️  DEGRADED: Some errors or slowdown detected.")

    if after - before > 50:
        print(f"  ✅ MESH ACTIVE: {after - before} papers moved.")
    else:
        print(f"  ⚠️  MESH SLOW: Only {after - before} papers moved.")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
