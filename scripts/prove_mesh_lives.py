#!/usr/bin/env python3
"""
Prove the mesh is alive: run searches, trigger re-bucketing, show papers moved.
"""
import sys
import time
sys.path.insert(0, ".")

from yp_bridge import YPEngine

DB_PATH = "data/phoenix_arxiv_1m.db"
N_SEARCHES = 100

# Queries that should co-occur (AI topics)
TOPIC_QUERIES = [
    "neural network", "deep learning", "machine learning",
    "gradient descent", "backpropagation", "convolutional",
    "transformer", "attention mechanism", "BERT",
    "reinforcement learning", "Q learning", "policy gradient",
    "computer vision", "image classification", "object detection",
    "natural language processing", "word embedding", "seq2seq",
    "generative adversarial", "GAN", "variational autoencoder",
    "graph neural network", "node embedding", "GNN",
    "optimization", "stochastic gradient", "Adam optimizer",
    "transfer learning", "fine tuning", "pretrained model",
    "federated learning", "differential privacy", "edge AI",
    "neural architecture search", "AutoML", "hyperparameter tuning",
    "knowledge distillation", "model compression", "quantization",
    "explainable AI", "interpretability", "SHAP",
    "causal inference", "bayesian network", "probabilistic model",
    "time series forecasting", "LSTM", "recurrent neural",
    "self supervised learning", "contrastive learning", "SimCLR",
    "multimodal learning", "vision transformer", "CLIP",
    "large language model", "prompt engineering", "in context learning",
    "retrieval augmented generation", "vector database", "semantic search",
    "diffusion model", "stable diffusion", "score based",
    "neural radiance field", "NeRF", "3D reconstruction",
    "reinforcement learning from feedback", "RLHF", "chatbot",
    "mixture of experts", "sparse model", "MoE",
    "continual learning", "catastrophic forgetting", "elastic weight",
    "meta learning", "learning to learn", "MAML",
    "few shot learning", "zero shot", "prompt tuning",
    "adversarial robustness", "network pruning", "lottery ticket",
    "neural tangent kernel", "infinite width", "kernel method",
    "symmetry in neural", "equivariant", "group convolution",
    "physics informed neural", "PINN", "scientific machine",
    "neural operator", "Fourier neural", "deep operator",
    "surrogate model", "Gaussian process", "Bayesian optimization",
    "active learning", "uncertainty sampling", "query strategy",
    "multi task learning", "task relation", "shared representation",
    "domain adaptation", "domain generalization", "invariant risk",
    "fairness in machine", "bias mitigation", "demographic parity",
    "differential privacy", "secure aggregation", "federated",
    "neural network verification", "formal guarantee", "abstract interpretation",
    "neural network compression", "knowledge transfer", "teacher student",
    "self distillation", "mutual learning", "deep mutual",
    "ensemble learning", "boosting", "bagging",
    "stacking", "blending", "model averaging",
    "evolutionary neural", "genetic algorithm", "neuroevolution",
    "swarm intelligence", "particle swarm", "ant colony",
    "neural architecture", "cell based", "DARTS",
    "one shot architecture", "weight sharing", "supernet",
    "hardware aware neural", "neural accelerator", "FPGA",
    "quantum machine learning", "variational quantum", "QML",
    "neuromorphic computing", "spiking neural", "event based",
]

def get_counts(lib):
    """Read current mesh state from Rust."""
    try:
        dyn = lib.yp_mesh_dynamic_bucket_count()
    except Exception:
        dyn = 0
    return {"dynamic_buckets": dyn}

def main():
    print("=" * 60)
    print("PROOF: THE MESH IS ALIVE")
    print("=" * 60)

    # 1. Init engine
    print("\n[1/5] Initializing engine...")
    t0 = time.time()
    engine = YPEngine()
    print(f"      Engine ready in {time.time()-t0:.1f}s")

    lib = engine.rust.lib if hasattr(engine, 'rust') else None

    # 2. BEFORE state
    print("\n[2/5] BEFORE searches:")
    before = get_counts(lib)
    print(f"      Papers in dynamic buckets: {before['dynamic_buckets']}")

    # 3. Run 100 searches
    print(f"\n[3/5] Running {N_SEARCHES} searches...")
    for i in range(N_SEARCHES):
        q = TOPIC_QUERIES[i % len(TOPIC_QUERIES)]
        try:
            engine.search(q, top_k=5)
        except Exception:
            pass
        if (i + 1) % 20 == 0:
            print(f"      ...{i+1} done")

    # 4. Check gravity accumulated
    print("\n[4/5] AFTER searches (before re-bucketing):")
    mid = get_counts(lib)
    print(f"      Papers in dynamic buckets: {mid['dynamic_buckets']}")

    # 5. Trigger re-bucketing NOW (don't wait for daemon)
    print("\n[5/5] Triggering re-bucketing cycle...")
    try:
        engine._run_rebucket_cycle()
    except Exception as e:
        print(f"      Re-bucketing error: {e}")

    # 6. AFTER re-bucketing
    print("\n=== RESULT ===")
    after = get_counts(lib)
    moved = after['dynamic_buckets'] - before['dynamic_buckets']
    print(f"  Dynamic buckets BEFORE: {before['dynamic_buckets']}")
    print(f"  Dynamic buckets AFTER:  {after['dynamic_buckets']}")
    print(f"  Papers moved: {moved}")

    if moved > 0:
        print("\n  ✅ MESH IS ALIVE: Papers moved based on gravity.")
    else:
        print("\n  ⚠️  No papers moved yet. Need more searches or lower threshold.")
        print("      (Gravity threshold is 2 queries per paper)")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
