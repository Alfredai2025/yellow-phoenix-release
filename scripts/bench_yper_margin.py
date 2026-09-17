#!/usr/bin/env python3
"""
YPER Block 2b: Margin-based early stopping.
Stop when top1 - top2 > delta, not when score > tau.
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
    d = np.bitwise_count(hashes[:, :k] ^ hashes[q, :k]).sum(axis=1)
    d[q] = 2**31
    return d

def exact_nn(hashes, q, k=5):
    d = ham_all(hashes, q, 64)
    return np.argsort(d)[:k]

def yper_margin(hashes, q, delta, k=5):
    # Level 3: 8-byte scan over all papers
    d8 = ham_all(hashes, q, 8)
    top2 = np.partition(d8, 1)[:2]
    gap3 = (1.0 - top2[0] / 64.0) - (1.0 - top2[1] / 64.0)
    if gap3 > delta:
        return np.argsort(d8)[:k], 3

    # Level 2: 16-byte on top 200
    cands = np.argpartition(d8, 200)[:200]
    d16 = np.bitwise_count(hashes[cands, :16] ^ hashes[q, :16]).sum(axis=1)
    top2 = np.partition(d16, 1)[:2]
    gap2 = (1.0 - top2[0] / 128.0) - (1.0 - top2[1] / 128.0)
    if gap2 > delta:
        return cands[np.argsort(d16)[:k]], 2

    # Level 1: 32-byte on top 50
    sub = cands[np.argpartition(d16, 50)[:50]]
    d32 = np.bitwise_count(hashes[sub, :32] ^ hashes[q, :32]).sum(axis=1)
    top2 = np.partition(d32, 1)[:2]
    gap1 = (1.0 - top2[0] / 256.0) - (1.0 - top2[1] / 256.0)
    if gap1 > delta:
        return sub[np.argsort(d32)[:k]], 1

    # Level 0: 64-byte on top 20
    sub2 = sub[np.argpartition(d32, 20)[:20]]
    d64 = np.bitwise_count(hashes[sub2, :64] ^ hashes[q, :64]).sum(axis=1)
    return sub2[np.argsort(d64)[:k]], 0

def main():
    hashes = find_hashes()
    N = len(hashes)
    print(f"Hashes: {N}")

    np.random.seed(42)
    n_test = 500
    q_idx = np.random.choice(N, n_test, replace=False)

    for delta in [0.001, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200]:
        full_t = 0.0
        yper_t = 0.0
        r1 = r5 = 0
        dist = {3: 0, 2: 0, 1: 0, 0: 0}

        for q in q_idx:
            t0 = time.time()
            ex = exact_nn(hashes, q, 5)
            full_t += time.time() - t0

            t0 = time.time()
            yp, level = yper_margin(hashes, q, delta, 5)
            yper_t += time.time() - t0
            dist[level] += 1

            r1 += 1 if ex[0] == yp[0] else 0
            r5 += 1 if len(set(ex) & set(yp)) >= 1 else 0

        avg_depth = sum(l * dist[l] for l in dist) / n_test
        print(f"\ndelta={delta:5.3f}  "
              f"R@1={r1/n_test*100:5.1f}%  R@5={r5/n_test*100:5.1f}%  "
              f"speedup={full_t/yper_t:4.2f}x  "
              f"depths={dist}  avg={avg_depth:.2f}")

if __name__ == '__main__':
    main()
