#!/usr/bin/env python3
"""
100,000 searches to generate gravity on 1,000+ papers.
Uses the live Flask API so gravity accumulates in the same process.
"""
import requests
import random
import time
import sys

API = "http://localhost:5001"
N = 100_000
BATCH = 5_000

QUERIES = [
    "neural network", "deep learning", "machine learning", "gradient descent",
    "transformer", "attention mechanism", "reinforcement learning", "computer vision",
    "natural language processing", "GAN", "medical imaging", "clinical diagnosis",
    "drug discovery", "legal contract", "case law", "regulatory compliance",
    "physics simulation", "quantum computing", "thermodynamics", "optimization",
    "linear programming", "convex optimization", "graph neural network",
    "knowledge graph", "ontology", "federated learning", "differential privacy",
    "edge AI", "time series forecasting", "anomaly detection", "clustering",
    "self supervised learning", "contrastive learning", "large language model",
    "prompt engineering", "retrieval augmented generation", "semantic search",
    "diffusion model", "generative model", "mixture of experts", "sparse model",
    "continual learning", "meta learning", "adversarial robustness",
    "network pruning", "physics informed neural network", "scientific machine learning",
    "surrogate model", "Bayesian optimization", "active learning", "multi task learning",
    "domain adaptation", "fairness in machine learning", "neural network verification",
    "formal guarantee", "neuromorphic computing", "spiking neural network",
    "recursive self modelling", "autopoiesis", "self organization",
    "predictive processing", "free energy principle", "embodied cognition",
    "consciousness", "qualia", "mind body problem", "language model reasoning",
    "chain of thought", "in context learning", "tool use", "agent architecture",
    "multi agent", "swarm intelligence", "collective intelligence", "emergence",
    "information theory", "entropy", "compression", "causal inference",
    "do calculus", "structural equation", "bayesian network",
    "probabilistic graphical model", "markov chain", "monte carlo",
    "mcmc", "stochastic process", "random walk", "brownian motion",
    "dynamical system", "chaos theory", "attractor", "complexity",
    "network science", "small world", "scale free network",
    "preferential attachment", "centrality", "community detection",
    "modularity", "spectral clustering", "dimensionality reduction",
    "manifold learning", "t-SNE", "principal component analysis",
    "independent component analysis", "factor analysis", "latent variable model",
    "hidden markov model", "state space model", "kalman filter",
    "particle filter", "sequential monte carlo", "gaussian process",
    "kernel method", "support vector machine", "decision tree",
    "random forest", "gradient boosting", "xgboost", "lightgbm",
    "catboost", "ensemble method", "model stacking", "blending",
    "hyperparameter optimization", "bayesian optimization", "optuna",
    "neural architecture search", "autoML", "meta learning",
    "few shot learning", "zero shot learning", "prompt tuning",
    "instruction tuning", "rlhf", "constitutional ai", "alignment",
    "safety", "interpretability", "mechanistic interpretability",
    "feature visualization", "activation patching", "probing",
    "representation learning", "disentanglement", "contrastive learning",
    "self supervision", "pretext task", "data augmentation", "mixup",
    "cutmix", "adversarial training", "robustness", "certified defense",
    "differential privacy", "federated learning", "secure aggregation",
    "homomorphic encryption", "secure multi party computation", "zk snark",
    "blockchain", "smart contract", "decentralized identity",
    "verifiable credential", "self sovereign", "web3", "dao",
    "tokenomics", "game theory", "mechanism design", "auction theory",
    "incentive alignment", "public good", "free rider", "social choice",
    "voting theory", "arrow theorem", "fair division", "envy freeness",
    "proportional allocation", "matching market", "stable marriage",
    "deferred acceptance", "market design", "school choice", "kidney exchange",
    "recommendation system", "collaborative filtering", "matrix factorization",
    "content based filtering", "hybrid recommendation", "session based",
    "sequential recommendation", "next basket prediction", "click through rate",
    "search ranking", "learning to rank", "listwise", "document retrieval",
    "passage retrieval", "dense retrieval", "sparse retrieval", "late interaction",
    "colbert", "re ranking", "cross encoder", "bi encoder", "knowledge distillation",
    "teacher student", "soft label", "quantization", "pruning", "knowledge transfer",
    "model compression", "mobile deployment", "edge inference", "tensorrt",
    "onnx", "openvino", "fpga acceleration", "asic design", "neural accelerator",
    "photonic computing", "optical neural network", "neuromorphic chip",
    "brain computer interface", "neural decoding", "neural encoding",
    "cognitive enhancement", "nootropic", "transhumanism", "longevity",
    "aging biology", "senolytic", "regenerative medicine", "stem cell",
    "tissue engineering", "synthetic biology", "gene editing", "crispr",
    "bioinformatics", "computational biology", "systems biology", "protein folding",
    "alphafold", "structure prediction", "molecular dynamics", "drug design",
    "virtual screening", "materials science", "battery chemistry", "solid state",
    "perovskite", "photovoltaic", "solar cell", "carbon capture", "climate model",
    "earth system", "oceanography", "atmospheric science", "weather prediction",
    "seismology", "earthquake prediction", "tsunami warning", "volcanology",
    "magma dynamics", "eruption forecasting", "planetary science", "exoplanet",
    "habitability", "astrobiology", "SETI", "technosignature", "cosmology",
    "dark matter", "dark energy", "gravitational wave", "ligo", "black hole",
    "neutron star", "pulsar", "fast radio burst", "gamma ray burst", "supernova",
    "nucleosynthesis", "quantum gravity", "string theory", "loop quantum gravity",
    "holographic principle", "ads cft", "gauge gravity", "condensed matter",
    "topological insulator", "superconductor", "high temperature superconductor",
    "meissner effect", "vortex", "quantum hall effect", "fractional statistics",
    "anyon", "spin liquid", "quantum spin ice", "frustrated magnet", "cold atom",
    "bose einstein condensate", "quantum simulation", "ion trap",
    "superconducting qubit", "photonic qubit", "quantum error correction",
    "surface code", "logical qubit", "quantum supremacy", "quantum advantage",
    "nisq", "variational quantum algorithm", "qaoa", "vqe", "quantum machine learning",
    "quantum neural network", "quantum kernel", "quantum sensing", "quantum metrology",
    "quantum imaging", "quantum communication", "quantum key distribution",
    "quantum internet", "quantum cryptography", "post quantum cryptography",
    "lattice based", "code based cryptography", "multivariate cryptography",
    "hash based", "isogeny based", "supersingular", "sidh",
]


def main():
    print("=" * 70)
    print("MASS MOVEMENT TEST: 100,000 searches")
    print("=" * 70)

    session = requests.Session()

    # Check API
    try:
        health = session.get(f"{API}/health", timeout=2).json()
        print(f"API health: {health}")
    except Exception as e:
        print(f"API not reachable: {e}")
        return

    before = health.get("dynamic_buckets", 0)
    print(f"\nDynamic buckets BEFORE: {before}")
    print(f"Target: {N} searches")

    t0 = time.time()
    errors = 0

    for i in range(N):
        q = random.choice(QUERIES)
        try:
            session.post(f"{API}/search", json={"query": q, "top_k": 5}, timeout=2)
        except Exception:
            errors += 1

        if (i + 1) % BATCH == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{N} done ({elapsed:.1f}s, {errors} errors)")

    total_time = time.time() - t0
    print(f"\nSearches complete: {total_time:.1f}s")
    print(f"Throughput: {N/total_time:.0f} searches/sec")
    print(f"Errors: {errors}")

    # Trigger mass re-bucketing
    print("\nTriggering mass re-bucketing...")
    rb_t0 = time.time()
    try:
        r = session.post(f"{API}/mesh/rebucket", timeout=120)
        rb_result = r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        rb_result = {"error": str(e)}

    rb_time = time.time() - rb_t0
    print(f"Re-bucketing done in {rb_time:.1f}s: {rb_result}")

    # Check after
    after = session.get(f"{API}/health", timeout=2).json().get("dynamic_buckets", 0)
    moved = after - before
    print(f"\n{'='*70}")
    print(f"  Dynamic buckets BEFORE: {before}")
    print(f"  Dynamic buckets AFTER:  {after}")
    print(f"  Papers moved: {moved}")
    print(f"  % of 12K corpus: {moved/12702*100:.1f}%")

    if moved > 1000:
        print(f"\n  ✅ MASS MOVEMENT: {moved} papers moved. Engine scales.")
    elif moved > 500:
        print(f"\n  ✅ STRONG MOVEMENT: {moved} papers moved.")
    elif moved > 100:
        print(f"\n  ✅ GOOD MOVEMENT: {moved} papers moved.")
    else:
        print(f"\n  ⚠️ LOW: {moved} moved. Need more searches or wider scan.")

    # Stability check
    print(f"\nStability check...")
    t0 = time.time()
    r = session.post(f"{API}/search", json={"query": "neural network", "top_k": 5}, timeout=5)
    latency = (time.time() - t0) * 1000
    if r.status_code == 200:
        print(f"  ✅ Search works: {latency:.2f} ms")
    else:
        print(f"  ❌ Search broken: {r.status_code}")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
