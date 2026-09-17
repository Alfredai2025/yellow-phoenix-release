#!/usr/bin/env python3
"""
TGE Phase 0/1: Trained Geometric Embedding Prototype.
PyTorch autograd version — runs in minutes on MPS/CPU.
Tests whether a learned 384 -> 5 projection makes the geometric
product's bivector carry retrievable signal.
"""
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_embeddings():
    for name in ['paper_embeddings_100k.npy', 'paper_embeddings.npy']:
        path = os.path.expanduser(f'~/yellow_phoenix/data/{name}')
        if os.path.exists(path):
            print(f"Loading: {path}")
            return np.load(path).astype(np.float32)
    raise FileNotFoundError("No embeddings found")


def sample_triplets(embeddings, n_triplets=2000, k_pos=5, k_neg=50, seed=42):
    """Self-supervised triplets from MiniLM embeddings."""
    rng = np.random.RandomState(seed)
    n = len(embeddings)
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    triplets = []

    for _ in range(n_triplets):
        q_idx = rng.randint(0, n)
        q = emb_norm[q_idx]

        # Positive: k-NN neighbor
        sims = emb_norm @ q
        top_k = np.argsort(sims)[-k_pos - 1:-1]
        pos_idx = rng.choice(top_k)

        # Negative: hard (similar but not top) or easy (random)
        if rng.random() < 0.7:
            candidates = np.argsort(sims)[-k_neg:-k_pos]
            if len(candidates) > 0:
                neg_idx = rng.choice(candidates)
            else:
                neg_idx = rng.randint(0, n)
        else:
            neg_idx = rng.randint(0, n)
            while neg_idx == q_idx or neg_idx == pos_idx:
                neg_idx = rng.randint(0, n)

        triplets.append((q_idx, pos_idx, neg_idx))

    return np.array(triplets, dtype=np.int64)


class TGEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(384, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 5)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        z = self.fc3(h)
        return F.normalize(z, dim=-1)


def geometric_score_batch(Q, P, weights):
    """
    Q: (B, 5), P: (B, 5) — unit vectors.
    Returns score = w0*|dot| + w2*||wedge||.
    """
    # Scalar part
    scalar = (Q * P).sum(dim=-1).abs()  # (B,)

    # Bivector components for each pair (i,j): Q_i P_j - Q_j P_i
    # There are 10 pairs in 5-D.
    pairs = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
    bivector = torch.stack([
        Q[:, i] * P[:, j] - Q[:, j] * P[:, i]
        for i, j in pairs
    ], dim=-1)  # (B, 10)
    bivector_norm = bivector.norm(dim=-1)  # (B,)

    return weights[0] * scalar + weights[1] * bivector_norm


def train(encoder, weights, triplets, embeddings, epochs=200, lr=1e-3,
          margin=0.5, batch_size=128, device='cpu'):
    optimizer = torch.optim.Adam(list(encoder.parameters()) + [weights], lr=lr)
    history = []
    n = len(triplets)

    emb_t = torch.from_numpy(embeddings).to(device)

    for epoch in range(epochs):
        np.random.shuffle(triplets)
        total_loss = 0.0
        n_violations = 0

        for i in range(0, n, batch_size):
            batch = triplets[i:i + batch_size]
            q = emb_t[batch[:, 0]]
            pos = emb_t[batch[:, 1]]
            neg = emb_t[batch[:, 2]]

            q_enc = encoder(q)
            pos_enc = encoder(pos)
            neg_enc = encoder(neg)

            sp = geometric_score_batch(q_enc, pos_enc, weights)
            sn = geometric_score_batch(q_enc, neg_enc, weights)

            loss = F.relu(margin - sp + sn).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Keep bivector weight alive
            with torch.no_grad():
                weights.clamp_(min=0.01)

            total_loss += loss.item() * len(batch)
            n_violations += (sp <= sn + margin).sum().item()

        avg_loss = total_loss / n
        history.append({
            'epoch': epoch,
            'loss': float(avg_loss),
            'violations': int(n_violations),
            'w_scalar': float(weights[0].item()),
            'w_bivector': float(weights[1].item()),
        })

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: loss={avg_loss:.4f} viol={n_violations}/{n} "
                  f"w=[{weights[0].item():.3f}, {weights[1].item():.3f}]")

    return history


def evaluate(encoder, weights, embeddings, n_test=200, device='cpu', k=5):
    """Geometric R@k vs Cosine R@k."""
    n = len(embeddings)
    rng = np.random.RandomState(123)
    q_idx = rng.choice(n, n_test, replace=False)

    emb_t = torch.from_numpy(embeddings).to(device)
    all_enc = encoder(emb_t)  # (n, 5)

    geo_hits = 0
    cos_hits = 0

    for q in q_idx:
        q_vec = all_enc[q:q + 1]  # (1, 5)

        # Geometric scores vs all
        scalar = (all_enc * q_vec).sum(dim=-1).abs()  # (n,)
        pairs = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
        bivec = torch.stack([
            all_enc[:, i] * q_vec[:, j] - all_enc[:, j] * q_vec[:, i]
            for i, j in pairs
        ], dim=-1)
        bivec_norm = bivec.norm(dim=-1)
        geo_scores = weights[0] * scalar + weights[1] * bivec_norm
        geo_scores[q] = -1e9
        geo_top = torch.topk(geo_scores, k).indices.cpu().numpy()

        # Cosine scores vs all
        cos_scores = emb_t @ emb_t[q]
        cos_scores[q] = -1e9
        cos_top = torch.topk(cos_scores, k).indices.cpu().numpy()

        # Ground truth = cosine nearest neighbor
        gt = cos_top[0]
        if gt in geo_top:
            geo_hits += 1
        if gt in cos_top:
            cos_hits += 1

    return geo_hits / n_test, cos_hits / n_test


def main():
    print("=" * 60)
    print("TGE PROTOTYPE: Trained Geometric Embedding")
    print("=" * 60)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    embeddings = load_embeddings()
    print(f"Embeddings: {embeddings.shape}")

    # Normalize embeddings for cosine sampling
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    # Training subset
    n_train = min(5000, len(emb_norm))
    train_emb = emb_norm[:n_train]
    print(f"Training subset: {n_train}")

    triplets = sample_triplets(train_emb, n_triplets=2000, k_pos=5, k_neg=50)
    print(f"Triplets: {len(triplets)}")

    encoder = TGEEncoder().to(device)
    weights = torch.nn.Parameter(torch.tensor([1.0, 0.5], device=device))

    print("\nTraining...")
    t0 = time.time()
    history = train(encoder, weights, triplets, train_emb, epochs=200,
                    lr=1e-3, margin=0.5, batch_size=128, device=device)
    print(f"Training time: {time.time() - t0:.1f}s")

    print("\nEvaluating...")
    geo_r1, cos_r1 = evaluate(encoder, weights, train_emb, n_test=200, device=device, k=1)
    geo_r5, cos_r5 = evaluate(encoder, weights, train_emb, n_test=200, device=device, k=5)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Geometric R@1: {geo_r1*100:.1f}%")
    print(f"Cosine R@1:    {cos_r1*100:.1f}%")
    print(f"Geometric R@5: {geo_r5*100:.1f}%")
    print(f"Cosine R@5:    {cos_r5*100:.1f}%")
    print(f"Scalar weight:   {weights[0].item():.3f}")
    print(f"Bivector weight: {weights[1].item():.3f}")

    # Save
    out_dir = os.path.expanduser('~/yellow_phoenix/data/tge')
    os.makedirs(out_dir, exist_ok=True)

    torch.save({
        'encoder': encoder.state_dict(),
        'weights': weights.detach().cpu(),
        'history': history,
    }, f"{out_dir}/tge_phase0.pt")

    with open(f"{out_dir}/history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved to {out_dir}/")

    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    if weights[1].item() > 0.3 and geo_r1 > 0.20:
        print("✅ TGE VIABLE — Proceed to Phase 1/2 (more data, harder negatives)")
    elif weights[1].item() > 0.3:
        print("⚠️  Bivector lives but recall low — Need more data / better negatives")
    else:
        print("❌ Bivector dead — TGE goes to MUSEUM")


if __name__ == '__main__':
    main()
