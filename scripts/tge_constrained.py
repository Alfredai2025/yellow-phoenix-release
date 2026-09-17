#!/usr/bin/env python3
"""
TGE Constrained: w2 = 0.3 * w0.
Forces the geometric score to peak near 0 degrees (identical),
using the bivector only as a small modulation.
PyTorch autograd version — runs in seconds.
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
    rng = np.random.RandomState(seed)
    n = len(embeddings)
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    triplets = []
    for _ in range(n_triplets):
        q_idx = rng.randint(0, n)
        q = emb_norm[q_idx]
        sims = emb_norm @ q
        top_k = np.argsort(sims)[-k_pos - 1:-1]
        pos_idx = rng.choice(top_k)
        if rng.random() < 0.7:
            candidates = np.argsort(sims)[-k_neg:-k_pos]
            neg_idx = rng.choice(candidates) if len(candidates) > 0 else rng.randint(0, n)
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


def geometric_score_batch(Q, P, w0=1.0):
    """Constrained: w2 = 0.3 * w0."""
    scalar = (Q * P).sum(dim=-1).abs()
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    bivec = torch.stack([
        Q[:, i] * P[:, j] - Q[:, j] * P[:, i]
        for i, j in pairs
    ], dim=-1)
    bivector_norm = bivec.norm(dim=-1)
    return w0 * (scalar + 0.3 * bivector_norm)


def train(encoder, triplets, embeddings, epochs=200, lr=1e-3, margin=0.5,
          batch_size=128, device='cpu'):
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
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

            sp = geometric_score_batch(q_enc, pos_enc)
            sn = geometric_score_batch(q_enc, neg_enc)

            loss = F.relu(margin - sp + sn).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)
            n_violations += (sp <= sn + margin).sum().item()

        avg_loss = total_loss / n
        history.append({
            'epoch': epoch,
            'loss': float(avg_loss),
            'violations': int(n_violations),
        })

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: loss={avg_loss:.4f} viol={n_violations}/{n}")

    return history


def evaluate(encoder, embeddings, n_test=200, device='cpu', k=1):
    n = len(embeddings)
    rng = np.random.RandomState(123)
    q_idx = rng.choice(n, n_test, replace=False)

    emb_t = torch.from_numpy(embeddings).to(device)
    all_enc = encoder(emb_t)

    hits = 0
    for q in q_idx:
        q_vec = all_enc[q:q + 1]
        scalar = (all_enc * q_vec).sum(dim=-1).abs()
        pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        bivec = torch.stack([
            all_enc[:, i] * q_vec[:, j] - all_enc[:, j] * q_vec[:, i]
            for i, j in pairs
        ], dim=-1)
        bivec_norm = bivec.norm(dim=-1)
        geo_scores = scalar + 0.3 * bivec_norm
        geo_scores[q] = -1e9
        geo_top = torch.topk(geo_scores, k).indices.cpu().numpy()

        cos_scores = emb_t @ emb_t[q]
        cos_scores[q] = -1e9
        cos_top = torch.topk(cos_scores, k).indices.cpu().numpy()

        gt = cos_top[0]
        if gt in geo_top:
            hits += 1

    return hits / n_test


def main():
    print("=" * 60)
    print("TGE CONSTRAINED: w2 = 0.3 * w0")
    print("=" * 60)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    embeddings = load_embeddings()
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    n_train = min(5000, len(emb_norm))
    train_emb = emb_norm[:n_train]
    print(f"Training subset: {n_train}")

    triplets = sample_triplets(train_emb, n_triplets=2000, k_pos=5, k_neg=50)
    print(f"Triplets: {len(triplets)}")

    encoder = TGEEncoder().to(device)

    print("\nTraining...")
    t0 = time.time()
    history = train(encoder, triplets, train_emb, epochs=200, lr=1e-3,
                    margin=0.5, batch_size=128, device=device)
    print(f"Training time: {time.time() - t0:.1f}s")

    print("\nEvaluating...")
    geo_r1 = evaluate(encoder, train_emb, n_test=200, device=device, k=1)
    geo_r5 = evaluate(encoder, train_emb, n_test=200, device=device, k=5)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Geometric R@1: {geo_r1*100:.1f}%")
    print(f"Geometric R@5: {geo_r5*100:.1f}%")
    print(f"w0 (scalar):   1.000 (fixed)")
    print(f"w2 (bivector): 0.300 (constrained)")

    out_dir = os.path.expanduser('~/yellow_phoenix/data/tge')
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        'encoder': encoder.state_dict(),
        'history': history,
        'constrained': True,
        'w2_ratio': 0.3,
    }, f"{out_dir}/tge_constrained.pt")

    with open(f"{out_dir}/history_constrained.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved to {out_dir}/")

    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    if geo_r1 > 0.50:
        print("✅ TGE VIABLE — Constrained bivector helps nearest-neighbor retrieval!")
    elif geo_r1 > 0.20:
        print("⚠️  Some signal, but not strong enough yet")
    else:
        print("❌ 5-D projection too crushed — TGE goes to MUSEUM")


if __name__ == '__main__':
    main()
