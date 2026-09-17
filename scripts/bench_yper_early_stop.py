#!/usr/bin/env python3
"""
YPER Block 2: Adaptive early stopping benchmark.
Tests if we can stop at 8-byte or 16-byte and still find the right answer.
Uses true nearest-neighbor recall (query excluded) and vectorized Hamming.
"""
import numpy as np, os, time

def find_hashes():
    for name in ['paper_hashes_100k.npy', 'paper_hashes.npy', 'paper_hashes_1m_synthetic.npy']:
        p = os.path.expanduser(f'~/yellow_phoenix/data/{name}')
        if os.path.exists(p):
            h = np.load(p)
            if h.dtype == np.bool_:
                h = h.astype(np.uint8)
            return h
    raise FileNotFoundError("No hash file found")

def ham_all(hashes, q, k):
    """Vectorized Hamming distances from query q to all papers using first k bytes."""
    return np.bitwise_count(hashes[:, :k] ^ hashes[q, :k]).sum(axis=1)

def exact_nn(hashes, q, k=5):
    d = ham_all(hashes, q, 64)
    d[q] = 2**31
    return np.argsort(d)[:k]

def yper(hashes, q, k=5, tau=0.90):
    # Level 3: 8-byte scan over all papers
    d8 = ham_all(hashes, q, 8)
    d8[q] = 2**31
    top1_score = 1.0 - d8.min() / (8 * 8)
    if top1_score >= tau:
        return np.argsort(d8)[:k], 1

    cands = np.argpartition(d8, 200)[:200]
    d16 = np.bitwise_count(hashes[cands, :16] ^ hashes[q, :16]).sum(axis=1)
    top16 = 1.0 - d16.min() / (16 * 8)
    if top16 >= tau:
        order = np.argsort(d16)
        return cands[order[:k]], 2

    sub = cands[np.argpartition(d16, 50)[:50]]
    d32 = np.bitwise_count(hashes[sub, :32] ^ hashes[q, :32]).sum(axis=1)
    top32 = 1.0 - d32.min() / (32 * 8)
    if top32 >= tau:
        order = np.argsort(d32)
        return sub[order[:k]], 3

    sub2 = sub[np.argpartition(d32, 20)[:20]]
    d64 = np.bitwise_count(hashes[sub2, :64] ^ hashes[q, :64]).sum(axis=1)
    order = np.argsort(d64)
    return sub2[order[:k]], 4

def main():
    hashes = find_hashes()
    N = len(hashes)
    print(f"Hashes: {N}")

    np.random.seed(42)
    n_test = 500
    q_idx = np.random.choice(N, n_test, replace=False)

    full_time = 0.0
    yper_time = 0.0
    dist = {1: 0, 2: 0, 3: 0, 4: 0}
    r1_y = r5_y = 0

    for q in q_idx:
        t0 = time.time()
        full = exact_nn(hashes, q, 5)
        full_time += time.time() - t0

        t0 = time.time()
        yp, depth = yper(hashes, q, 5, tau=0.90)
        yper_time += time.time() - t0
        dist[depth] += 1

        r1_y += 1 if full[0] == yp[0] else 0
        r5_y += 1 if len(set(full) & set(yp)) >= 1 else 0

    print(f"\n{'='*60}")
    print("YPER EARLY STOPPING BENCHMARK (true NN recall, query excluded)")
    print(f"{'='*60}")
    print(f"Queries tested: {n_test}")
    print(f"\nSpeed:")
    print(f"  Full 64-byte:   {full_time*1000:.1f} ms total | {full_time/n_test*1000:.3f} ms/query")
    print(f"  YPER adaptive:  {yper_time*1000:.1f} ms total | {yper_time/n_test*1000:.3f} ms/query")
    print(f"  Speedup:        {full_time/yper_time:.2f}x")
    print(f"\nRecall vs exact NN:")
    print(f"  YPER R@1: {r1_y}/{n_test} = {r1_y/n_test*100:.1f}%")
    print(f"  YPER R@5: {r5_y}/{n_test} = {r5_y/n_test*100:.1f}%")
    print(f"\nEarly stop distribution:")
    print(f"  Stopped at 8-byte  (L3): {dist[1]}/{n_test} = {dist[1]/n_test*100:.1f}%")
    print(f"  Stopped at 16-byte (L2): {dist[2]}/{n_test} = {dist[2]/n_test*100:.1f}%")
    print(f"  Stopped at 32-byte (L1): {dist[3]}/{n_test} = {dist[3]/n_test*100:.1f}%")
    print(f"  Full 64-byte    (L0): {dist[4]}/{n_test} = {dist[4]/n_test*100:.1f}%")
    avg_depth = sum(d * dist[d] for d in dist) / n_test
    print(f"\nAvg depth: {avg_depth:.2f}/4 levels")

if __name__ == '__main__':
    main()
