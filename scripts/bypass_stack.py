#!/usr/bin/env python3
"""
Bypass Stack: Stacked cheap filters + reverse reconstruction.
Goal: eliminate 99% of candidates before full cosine.
"""
import numpy as np
import os
import time


def load_embeddings_and_hashes():
    emb_path = os.path.expanduser('~/yellow_phoenix/data/paper_embeddings_100k.npy')
    for name in ['paper_hashes_100k.npy', 'paper_hashes.npy', 'paper_hashes_512.npy']:
        hash_path = os.path.expanduser(f'~/yellow_phoenix/data/{name}')
        if os.path.exists(hash_path):
            break
    embeddings = np.load(emb_path).astype(np.float32)
    hashes = np.load(hash_path)
    if hashes.dtype == np.bool_:
        hashes = hashes.astype(np.uint8)
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    return emb_norm, hashes


def bypass_l0(q_hash, p_hash, threshold=0.875):
    q8 = q_hash[:8]
    p8 = p_hash[:8]
    dist = sum((x ^ y).bit_count() for x, y in zip(q8, p8))
    sim = 1.0 - dist / 64.0
    return sim > threshold


def bypass_l1(q_hash, p_hash, threshold=0.30):
    q16 = (q_hash[:16].astype(np.float32) / 255.0) * 0.99
    p16 = (p_hash[:16].astype(np.float32) / 255.0) * 0.99
    u_sq = np.sum(q16**2)
    v_sq = np.sum(p16**2)
    diff_sq = np.sum((q16 - p16)**2)
    num = 2 * diff_sq
    denom = (1 - u_sq) * (1 - v_sq) + 1e-8
    d = np.arccosh(1 + num / denom)
    return d < threshold


class ReverseReconstructor:
    def __init__(self):
        self.W1 = np.array([[2.0, 1.5, 0.5],
                            [1.0, 2.0, 1.0],
                            [0.5, 1.5, 2.0],
                            [1.0, 1.0, 1.0],
                            [0.5, 0.5, 0.5],
                            [2.0, 0.0, 0.0],
                            [0.0, 2.0, 0.0],
                            [0.0, 0.0, 2.0]]).T
        self.b1 = np.zeros(8)
        self.W2 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
        self.b2 = 0.0

    def relu(self, x):
        return np.maximum(0, x)

    def predict(self, s8, s16, s32):
        x = np.array([s8, s16, s32])
        h = self.relu(x @ self.W1 + self.b1)
        return np.dot(h, self.W2) + self.b2

    def train(self, pairs, cosines, epochs=100, lr=0.01):
        X = np.array(pairs)
        y = np.array(cosines)
        for epoch in range(epochs):
            h = self.relu(X @ self.W1 + self.b1)
            pred = h @ self.W2 + self.b2
            loss = np.mean((pred - y)**2)
            d_pred = 2 * (pred - y) / len(y)
            dW2 = h.T @ d_pred
            db2 = np.sum(d_pred)
            d_h = np.outer(d_pred, self.W2)
            d_h[h <= 0] = 0
            dW1 = X.T @ d_h
            db1 = np.sum(d_h, axis=0)
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            if epoch % 20 == 0:
                print(f"  Reverse MLP epoch {epoch}: loss={loss:.6f}")
        return self


def bypass_l3(q_emb, p_emb):
    dot = np.dot(q_emb, p_emb)
    return abs(dot) > 0.85 or abs(dot) < 0.15


def full_cosine(q_emb, p_emb):
    return np.dot(q_emb, p_emb)


class BypassStack:
    def __init__(self, reconstructor=None, l0_thresh=0.875, l1_thresh=0.30):
        self.recon = reconstructor
        self.l0_thresh = l0_thresh
        self.l1_thresh = l1_thresh
        self.stats = {k: 0 for k in [
            'l0_pass', 'l0_reject',
            'l1_pass', 'l1_reject',
            'l2_pass', 'l2_reject', 'l2_ambiguous',
            'l3_pass', 'l3_reject', 'l3_ambiguous',
            'full_cosine'
        ]}

    def reset_stats(self):
        for k in self.stats:
            self.stats[k] = 0

    def query(self, q_hash, p_hash, q_emb, p_emb):
        path = []
        if not bypass_l0(q_hash, p_hash, self.l0_thresh):
            self.stats['l0_reject'] += 1
            return 0.0, 0, path + ['L0_REJECT']
        self.stats['l0_pass'] += 1
        path.append('L0')

        if not bypass_l1(q_hash, p_hash, self.l1_thresh):
            self.stats['l1_reject'] += 1
            return 0.1, 1, path + ['L1_REJECT']
        self.stats['l1_pass'] += 1
        path.append('L1')

        if self.recon is not None:
            q8 = q_hash[:8]; p8 = p_hash[:8]
            s8 = 1.0 - sum((x ^ y).bit_count() for x, y in zip(q8, p8)) / 64.0
            q16 = q_hash[:16]; p16 = p_hash[:16]
            s16 = 1.0 - sum((x ^ y).bit_count() for x, y in zip(q16, p16)) / 128.0
            q32 = q_hash[:32]; p32 = p_hash[:32]
            s32 = 1.0 - sum((x ^ y).bit_count() for x, y in zip(q32, p32)) / 256.0
            pred = self.recon.predict(s8, s16, s32)
            if pred > 0.95:
                self.stats['l2_pass'] += 1
                return pred, 2, path + ['L2_ACCEPT']
            elif pred < 0.50:
                self.stats['l2_reject'] += 1
                return pred, 2, path + ['L2_REJECT']
            else:
                self.stats['l2_ambiguous'] += 1
                path.append('L2_AMBIGUOUS')
        else:
            path.append('L2_SKIP')

        if bypass_l3(q_emb, p_emb):
            score = full_cosine(q_emb, p_emb)
            self.stats['l3_pass'] += 1
            return score, 3, path + ['L3_FAST']
        else:
            self.stats['l3_ambiguous'] += 1
            path.append('L3_AMBIGUOUS')

        score = full_cosine(q_emb, p_emb)
        self.stats['full_cosine'] += 1
        return score, 4, path + ['FULL']


def train_reconstructor(emb_norm, hashes, n_samples=10000):
    print("Training reverse reconstructor...")
    n = len(emb_norm)
    pairs = []
    cosines = []
    for _ in range(n_samples):
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i == j:
            continue
        q_hash = hashes[i]
        p_hash = hashes[j]
        s8 = 1.0 - sum((x ^ y).bit_count() for x, y in zip(q_hash[:8], p_hash[:8])) / 64.0
        s16 = 1.0 - sum((x ^ y).bit_count() for x, y in zip(q_hash[:16], p_hash[:16])) / 128.0
        s32 = 1.0 - sum((x ^ y).bit_count() for x, y in zip(q_hash[:32], p_hash[:32])) / 256.0
        cos = np.dot(emb_norm[i], emb_norm[j])
        pairs.append((s8, s16, s32))
        cosines.append(cos)

    recon = ReverseReconstructor()
    recon.train(pairs, cosines, epochs=100, lr=0.01)
    preds = [recon.predict(p[0], p[1], p[2]) for p in pairs[:1000]]
    corr = np.corrcoef(preds, cosines[:1000])[0, 1]
    print(f"Reverse recon correlation: {corr:.3f}")
    return recon


def evaluate_bypass_stack(emb_norm, hashes, n_test=100):
    n = len(emb_norm)
    recon = train_reconstructor(emb_norm, hashes, n_samples=10000)
    stack = BypassStack(reconstructor=recon)

    cos_r1 = 0
    bypass_r1 = 0
    cos_time = 0.0
    bypass_time = 0.0
    total_depth = 0

    for _ in range(n_test):
        q_idx = np.random.randint(0, n)
        q_hash = hashes[q_idx]
        q_emb = emb_norm[q_idx]

        t0 = time.time()
        cos_sims = emb_norm @ q_emb
        cos_sims[q_idx] = -999
        gt = np.argsort(cos_sims)[::-1][0]
        cos_time += time.time() - t0

        t0 = time.time()
        stack.reset_stats()
        best_score = -999
        best_idx = -1
        for i in range(n):
            if i == q_idx:
                continue
            score, depth, _ = stack.query(q_hash, hashes[i], q_emb, emb_norm[i])
            total_depth += depth
            if score > best_score:
                best_score = score
                best_idx = i
        bypass_time += time.time() - t0

        if best_idx == gt:
            bypass_r1 += 1
        cos_r1 += 1

    avg_depth = total_depth / (n_test * n)

    print(f"\n{'='*60}")
    print("BYPASS STACK RESULTS")
    print(f"{'='*60}")
    print(f"Cosine R@1:      {cos_r1}/{n_test} = 100.0%")
    print(f"Bypass R@1:      {bypass_r1}/{n_test} = {bypass_r1/n_test*100:.1f}%")
    print(f"Cosine time:     {cos_time*1000:.1f} ms")
    print(f"Bypass time:     {bypass_time*1000:.1f} ms")
    print(f"Speedup:         {cos_time/bypass_time:.2f}x")
    print(f"Avg bypass depth: {avg_depth:.2f} / 4")

    print(f"\n{'='*60}")
    print("BYPASS STATISTICS")
    print(f"{'='*60}")
    for k, v in stack.stats.items():
        print(f"  {k}: {v}")

    total_queries = n_test * n
    l0_kill = stack.stats['l0_reject'] / total_queries * 100
    l1_kill = stack.stats['l1_reject'] / total_queries * 100
    l2_kill = stack.stats['l2_reject'] / total_queries * 100
    full_run = stack.stats['full_cosine'] / total_queries * 100

    print(f"\nKill rates:")
    print(f"  L0 (8-byte):      {l0_kill:.1f}%")
    print(f"  L1 (hyperbolic):  {l1_kill:.1f}%")
    print(f"  L2 (reverse):     {l2_kill:.1f}%")
    print(f"  L4 (full cosine): {full_run:.1f}% still alive")

    if bypass_r1 / n_test > 0.90 and cos_time / bypass_time > 2.0:
        print(f"\n✅ BYPASS STACK WORKS")
    else:
        print(f"\n⚠️  Needs tuning — kill rates or accuracy too low")


def main():
    print("=" * 60)
    print("BYPASS STACK: Stacked cheap filters + reverse reconstruction")
    print("=" * 60)

    emb_norm, hashes = load_embeddings_and_hashes()
    print(f"Embeddings: {emb_norm.shape}")
    print(f"Hashes: {hashes.shape}")

    n = min(2000, len(emb_norm))
    evaluate_bypass_stack(emb_norm[:n], hashes[:n], n_test=100)


if __name__ == '__main__':
    np.random.seed(42)
    main()
