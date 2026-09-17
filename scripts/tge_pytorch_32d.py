#!/usr/bin/env python3
"""
TGE Phase 1: 32-D PyTorch Geometric Embedding
Uses MPS (Mac GPU) if available. CPU fallback.
Score = w0*|dot| + w2*||q p^T - p q^T||_F (skew-symmetric norm)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import json


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class TGEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(384, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.normalize(x, p=2, dim=1)


def geometric_score(q, p, w0, w2):
    """
    q, p: (batch, 32) unit vectors
    Returns: scalar + skew-symmetric norm
    """
    dot = (q * p).sum(dim=1)
    skew_norm = torch.sqrt(2.0 * (1.0 - dot**2 + 1e-8))
    return w0 * torch.abs(dot) + w2 * skew_norm


def sample_triplets(embeddings, n_triplets=5000, k_pos=5, seed=42):
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
        neg_idx = rng.randint(0, n)
        while neg_idx == q_idx or neg_idx == pos_idx:
            neg_idx = rng.randint(0, n)
        triplets.append((q_idx, pos_idx, neg_idx))
    return triplets


def train_epoch(encoder, embeddings, triplets, optimizer, w0, w2, margin=0.5, batch_size=256):
    encoder.train()
    n = len(triplets)
    total_loss = 0.0
    n_violations = 0

    indices = np.random.permutation(n)

    for i in range(0, n, batch_size):
        batch_idx = indices[i:i + batch_size]
        q_idx = [triplets[j][0] for j in batch_idx]
        pos_idx = [triplets[j][1] for j in batch_idx]
        neg_idx = [triplets[j][2] for j in batch_idx]

        q = torch.tensor(embeddings[q_idx], dtype=torch.float32, device=device)
        pos = torch.tensor(embeddings[pos_idx], dtype=torch.float32, device=device)
        neg = torch.tensor(embeddings[neg_idx], dtype=torch.float32, device=device)

        q_z = encoder(q)
        pos_z = encoder(pos)
        neg_z = encoder(neg)

        sp = geometric_score(q_z, pos_z, w0, w2)
        sn = geometric_score(q_z, neg_z, w0, w2)

        loss = F.relu(margin - sp + sn).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Keep w2 non-negative and not exploding
        with torch.no_grad():
            w2.clamp_(0.0, 5.0)

        total_loss += loss.item() * len(batch_idx)
        n_violations += (sp <= sn + margin).sum().item()

    return total_loss / n, n_violations


def evaluate(encoder, w0, w2, embeddings, n_test=500, batch_size=512):
    encoder.eval()
    n = len(embeddings)
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    all_z = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.tensor(emb_norm[i:i + batch_size], dtype=torch.float32, device=device)
            all_z.append(encoder(batch).cpu().numpy())
    all_z = np.concatenate(all_z, axis=0)

    geo_r1 = 0
    cos_r1 = 0

    for _ in range(n_test):
        q_idx = np.random.randint(0, n)
        q = emb_norm[q_idx]

        cos_sims = emb_norm @ q
        gt = np.argsort(cos_sims)[-2]

        qz = all_z[q_idx:q_idx + 1]
        dots = (all_z @ qz.T).flatten()
        skews = np.sqrt(2.0 * (1.0 - dots**2 + 1e-8))
        geo_scores = w0 * np.abs(dots) + w2 * skews
        geo_scores[q_idx] = -999
        geo_best = np.argmax(geo_scores)

        if geo_best == gt:
            geo_r1 += 1

        cos_best = np.argsort(cos_sims)[-2]
        if cos_best == gt:
            cos_r1 += 1

    return geo_r1 / n_test, cos_r1 / n_test


def main():
    print("=" * 60)
    print("TGE PHASE 1: 32-D PyTorch Geometric Embedding")
    print("=" * 60)
    print(f"Device: {device}")

    for name in ['paper_embeddings_100k.npy', 'paper_embeddings.npy']:
        path = os.path.expanduser(f'~/yellow_phoenix/data/{name}')
        if os.path.exists(path):
            print(f"Loading: {path}")
            embeddings = np.load(path).astype(np.float32)
            break

    print(f"Embeddings: {embeddings.shape}")

    n_train = min(10000, len(embeddings))
    train_emb = embeddings[:n_train]
    print(f"Training subset: {n_train}")

    print("Sampling triplets...")
    triplets = sample_triplets(train_emb, n_triplets=5000, k_pos=5)
    print(f"Triplets: {len(triplets)}")

    encoder = TGEEncoder().to(device)
    w0 = torch.tensor(1.0, device=device)
    w2 = nn.Parameter(torch.tensor(0.3, device=device))

    optimizer = torch.optim.Adam(list(encoder.parameters()) + [w2], lr=0.001)

    print("\nTraining...")
    best_loss = float('inf')
    history = []

    for epoch in range(100):
        t0 = time.time()
        loss, viol = train_epoch(encoder, train_emb, triplets, optimizer, w0, w2,
                                 margin=0.5, batch_size=256)
        elapsed = time.time() - t0

        history.append({
            'epoch': epoch,
            'loss': float(loss),
            'violations': int(viol),
            'w2': float(w2.item()),
        })

        if loss < best_loss:
            best_loss = loss

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={loss:.4f} viol={viol}/{len(triplets)} w2={w2.item():.3f} ({elapsed:.1f}s)")

    print("\nEvaluating...")
    geo_r1, cos_r1 = evaluate(encoder, 1.0, w2.item(), train_emb, n_test=500, batch_size=512)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Geometric R@1: {geo_r1*100:.1f}%")
    print(f"Cosine R@1:    {cos_r1*100:.1f}%")
    print(f"w2 (bivector): {w2.item():.3f}")

    out_dir = os.path.expanduser('~/yellow_phoenix/data/tge')
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        'encoder': encoder.state_dict(),
        'w2': w2.detach().cpu(),
        'history': history,
    }, f"{out_dir}/tge_32d.pt")

    with open(f"{out_dir}/history_32d.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved: {out_dir}/tge_32d.pt")

    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    if geo_r1 > 0.50:
        print("✅ 32-D TGE VIABLE — Proceed to Phase 2 (100K triplets, hard negatives)")
    elif geo_r1 > 0.20:
        print("⚠️  Some signal — Need more data / hard negatives / tuning")
    else:
        print("❌ 32-D also fails — Geometric structure is MUSEUM for retrieval")


if __name__ == '__main__':
    main()
