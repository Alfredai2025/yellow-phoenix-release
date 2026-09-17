#!/usr/bin/env python3
"""
Overnight ITQ Math Search — tries multiple formulations to maximize Hamming R@1.
Leave running. Review results in the morning.
"""
import numpy as np, os, sys, time, json, signal, datetime
from pathlib import Path

# ============ CONFIG ============
EMB_PATHS = [
    os.path.expanduser("~/Documents/11 Aug Yellow Back up/paper_embeddings.npy"),
    os.path.expanduser("~/yellow_phoenix/data/paper_embeddings.npy"),
    os.path.expanduser("~/yellow_phoenix/data/paper_embeddings_1m_synthetic.npy"),
]
OUT_DIR = os.path.expanduser("~/yellow_phoenix/data/overnight_itq")
os.makedirs(OUT_DIR, exist_ok=True)

LOG_FILE   = os.path.join(OUT_DIR, "overnight.log")
RESULTS_FILE = os.path.join(OUT_DIR, "results.jsonl")
BEST_MODEL = os.path.join(OUT_DIR, "best_model.npz")

# Thermal safety — sleep if CPU is throttled
def thermal_ok():
    try:
        import subprocess
        r = subprocess.run(["pmset","-g","therm"], capture_output=True, text=True, timeout=5)
        for line in r.stdout.split("\n"):
            if "CPU_Speed_Limit" in line:
                limit = int(line.split("=")[-1].strip())
                if limit < 60:
                    log(f"⚠️  THERMAL THROTTLE: CPU limit {limit}%. Sleeping 90s...")
                    time.sleep(90)
                    return False
    except Exception:
        pass
    return True

def log(msg):
    ts = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ============ LOAD ============
def load_embeddings():
    for p in EMB_PATHS:
        if os.path.exists(p):
            log(f"Loading: {p}")
            X = np.load(p).astype(np.float32)
            log(f"  Shape: {X.shape}")
            return X
    raise FileNotFoundError("No embeddings found")

# ============ EVALUATION ============
def hamming_r1(X_proj, R, n_bits, full=False):
    """Compute Hamming R@1. If not full, use 500-sample subset for speed."""
    N = X_proj.shape[0]
    B = (X_proj @ R > 0).astype(np.uint8)
    codes = np.packbits(B.reshape(N, n_bits//8, 8), axis=2).reshape(N, n_bits//8)
    
    idx = np.arange(N) if full else np.random.choice(N, min(500, N), replace=False)
    correct = 0
    for i in idx:
        xor = np.bitwise_xor(codes, codes[i])
        pc = np.unpackbits(xor.reshape(1,-1), axis=1).sum(axis=1)
        pc[i] = 999999
        if pc.argmin() == i:
            correct += 1
    return correct / len(idx)

# ============ METHOD 1: STANDARD ITQ (CORRECTED) ============
def method_standard(X, n_bits, n_iter, normalize, pca_scale, init):
    N, d = X.shape
    Xw = X.copy()
    if normalize:
        norms = np.linalg.norm(Xw, axis=1, keepdims=True)
        norms[norms==0] = 1
        Xw = Xw / norms
    
    mean = Xw.mean(axis=0)
    Xc = Xw - mean
    pca_dim = min(d, int(n_bits * pca_scale))
    
    if init == 'pca':
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        W = Vt[:pca_dim].T.astype(np.float32)
    else:
        W = np.random.randn(d, pca_dim).astype(np.float32)
        W, _ = np.linalg.qr(W)
    
    Xp = Xc @ W
    if pca_dim > n_bits:
        _, _, Vt2 = np.linalg.svd(Xp, full_matrices=False)
        W2 = Vt2[:n_bits].T.astype(np.float32)
        Xp, W = Xp @ W2, W @ W2
    elif pca_dim < n_bits:
        pad = np.random.randn(d, n_bits - pca_dim).astype(np.float32)
        pad, _ = np.linalg.qr(pad)
        W = np.concatenate([W, pad], axis=1)
        Xp = Xc @ W
    
    R = np.eye(n_bits, dtype=np.float32)
    best_R, best_r1 = R.copy(), -1.0
    
    for it in range(n_iter):
        if not thermal_ok():
            it -= 1
            continue
        V = Xp @ R
        B = np.sign(V); B[B==0] = 1
        C = Xp.T @ B
        U, _, Vt = np.linalg.svd(C)
        R_new = Vt.T @ U.T
        diff = np.linalg.norm(R - R_new)
        R = R_new
        if (it+1) % 20 == 0 or it < 5:
            r1 = hamming_r1(Xp, R, n_bits)
            if r1 > best_r1:
                best_r1, best_R = r1, R.copy()
            log(f"  iter {it+1:3d}: diff={diff:.6f}  sample_R@1={r1:.1%}")
        if diff < 1e-7:
            log(f"  Converged at {it+1}")
            break
    R = best_R
    return {'W_pca':W, 'R':R, 'mean':mean, 'r1_sample':best_r1,
            'r1_full': hamming_r1(Xp, R, n_bits, full=True)}

# ============ METHOD 2: BALANCED BITS (per-bit threshold) ============
def method_balanced(X, n_bits, n_iter, normalize):
    N, d = X.shape
    Xw = X.copy()
    if normalize:
        norms = np.linalg.norm(Xw, axis=1, keepdims=True); norms[norms==0]=1; Xw = Xw/norms
    mean = Xw.mean(axis=0); Xc = Xw - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:n_bits].T.astype(np.float32)
    Xp = Xc @ W
    R = np.eye(n_bits, dtype=np.float32)
    for it in range(n_iter):
        V = Xp @ R
        # Per-bit median threshold instead of zero
        B = np.zeros_like(V, dtype=np.float32)
        for b in range(n_bits):
            med = np.median(V[:,b])
            B[:,b] = np.where(V[:,b] > med, 1, -1)
        C = Xp.T @ B
        U, _, Vt = np.linalg.svd(C)
        R = Vt.T @ U.T
        if (it+1) % 20 == 0:
            r1 = hamming_r1(Xp, R, n_bits)
            log(f"  iter {it+1}: sample_R@1={r1:.1%}")
    return {'W_pca':W, 'R':R, 'mean':mean, 'r1_full': hamming_r1(Xp, R, n_bits, full=True)}

# ============ METHOD 3: GRAPH-AWARE INITIALIZATION ============
def method_graph(X, n_bits, n_iter):
    N, d = X.shape
    # k-NN graph in original space
    k = min(10, N-1)
    log(f"  Building {k}-NN graph...")
    sims = X @ X.T
    knn = np.argsort(sims, axis=1)[:, -k-1:-1]  # top k neighbors
    # Laplacian
    A = np.zeros((N,N), dtype=np.float32)
    for i in range(N):
        A[i, knn[i]] = 1
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigvals, eigvecs = np.linalg.eigh(L)
    # Use graph eigenvectors as initial projection
    graph_emb = eigvecs[:, 1:n_bits+1].astype(np.float32)
    norms = np.linalg.norm(graph_emb, axis=1, keepdims=True); norms[norms==0]=1
    graph_emb = graph_emb / norms
    mean = graph_emb.mean(axis=0); Xp = graph_emb - mean
    W = np.eye(d, n_bits)  # dummy — graph method is non-generalizable baseline
    R = np.eye(n_bits, dtype=np.float32)
    for it in range(n_iter):
        V = Xp @ R; B = np.sign(V); B[B==0]=1
        C = Xp.T @ B
        U, _, Vt = np.linalg.svd(C)
        R = Vt.T @ U.T
    return {'W_pca':W, 'R':R, 'mean':mean, 'note':'graph_baseline',
            'r1_full': hamming_r1(Xp, R, n_bits, full=True)}

# ============ METHOD 4: REFINEMENT WITH HARD NEGATIVES ============
def method_refinement(X, base, n_refine=100):
    N, d = X.shape; n_bits = base['W_pca'].shape[1]
    mean = base['mean']; W = base['W_pca']; R = base['R'].copy()
    Xc = X - mean; Xp = Xc @ W
    log("  Refining with hard negatives...")
    for rit in range(n_refine):
        B = (Xp @ R > 0).astype(np.float32) * 2 - 1  # {-1,+1}
        # Sample hard negatives
        sample = np.random.choice(N, min(300, N), replace=False)
        grad = np.zeros((n_bits, n_bits), dtype=np.float32)
        for i in sample:
            # True neighbor in PCA space
            sims = Xp[i] @ Xp.T
            true_nn = np.argsort(sims)[-5:]
            for j in true_nn:
                if i == j: continue
                bi, bj = B[i], B[j]
                disagree = (bi != bj).mean()
                if disagree > 0.25:  # hard negative
                    grad += 0.05 * np.outer(Xp[i] - Xp[j], bi - bj)
        if np.linalg.norm(grad) > 1e-8:
            U, _, Vt = np.linalg.svd(grad)
            R = R @ (U @ Vt)
        if (rit+1) % 20 == 0:
            r1 = hamming_r1(Xp, R, n_bits)
            log(f"  ref iter {rit+1}: sample_R@1={r1:.1%}")
    return {'W_pca':W, 'R':R, 'mean':mean,
            'r1_full': hamming_r1(Xp, R, n_bits, full=True)}

# ============ METHOD 5: ITQ ON L2-NORMALIZED PCA ============
def method_l2pca(X, n_bits, n_iter):
    N, d = X.shape
    mean = X.mean(axis=0); Xc = X - mean
    # L2 normalize each dimension
    col_norms = np.linalg.norm(Xc, axis=0, keepdims=True)
    col_norms[col_norms==0] = 1
    Xc = Xc / col_norms
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:n_bits].T.astype(np.float32)
    Xp = Xc @ W
    R = np.eye(n_bits, dtype=np.float32)
    for it in range(n_iter):
        V = Xp @ R; B = np.sign(V); B[B==0]=1
        C = Xp.T @ B
        U, _, Vt = np.linalg.svd(C)
        R = Vt.T @ U.T
    return {'W_pca':W, 'R':R, 'mean':mean,
            'r1_full': hamming_r1(Xp, R, n_bits, full=True)}

# ============ METHOD 6: DOUBLE ITQ (PCA→ITQ→PCA→ITQ) ============
def method_double(X, n_bits, n_iter):
    N, d = X.shape
    mean = X.mean(axis=0); Xc = X - mean
    # First ITQ
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    W1 = Vt[:n_bits].T.astype(np.float32)
    X1 = Xc @ W1
    R1 = np.eye(n_bits, dtype=np.float32)
    for it in range(n_iter//2):
        V = X1 @ R1; B = np.sign(V); B[B==0]=1
        C = X1.T @ B
        U, _, Vt = np.linalg.svd(C)
        R1 = Vt.T @ U.T
    # Project through first rotation
    X2 = X1 @ R1
    # Second PCA on rotated space
    _, _, Vt2 = np.linalg.svd(X2, full_matrices=False)
    W2 = Vt2[:n_bits].T.astype(np.float32)
    Xp = X2 @ W2
    # Second ITQ
    R2 = np.eye(n_bits, dtype=np.float32)
    for it in range(n_iter//2):
        V = Xp @ R2; B = np.sign(V); B[B==0]=1
        C = Xp.T @ B
        U, _, Vt = np.linalg.svd(C)
        R2 = Vt.T @ U.T
    # Combined rotation
    R = R1 @ W2 @ R2  # Approximate combined
    W = W1 @ W2
    return {'W_pca':W, 'R':R, 'mean':mean,
            'r1_full': hamming_r1(Xp, R2, n_bits, full=True)}  # eval with R2 on Xp

# ============ METHOD 7: RANDOM FOURIER FEATURES + ITQ ============
def method_rff(X, n_bits, n_iter, gamma=1.0):
    N, d = X.shape
    mean = X.mean(axis=0); Xc = X - mean
    # Random Fourier Features
    W_rff = np.random.randn(d, n_bits).astype(np.float32) * np.sqrt(2*gamma)
    b_rff = np.random.uniform(0, 2*np.pi, n_bits).astype(np.float32)
    X_rff = np.cos(Xc @ W_rff + b_rff) * np.sqrt(2.0 / n_bits)
    # ITQ on RFF
    R = np.eye(n_bits, dtype=np.float32)
    for it in range(n_iter):
        V = X_rff @ R; B = np.sign(V); B[B==0]=1
        C = X_rff.T @ B
        U, _, Vt = np.linalg.svd(C)
        R = Vt.T @ U.T
    # Store W_rff and b_rff in W_pca for inference
    W = np.concatenate([W_rff, b_rff.reshape(1,-1)], axis=0)
    return {'W_pca':W, 'R':R, 'mean':mean, 'note':'rff',
            'r1_full': hamming_r1(X_rff, R, n_bits, full=True)}

# ============ MAIN ============
def main():
    log("="*60)
    log("OVERNIGHT ITQ MATH SEARCH")
    log("="*60)
    
    X = load_embeddings()
    N, d = X.shape
    best_r1 = -1.0
    best_model = None
    results = []
    
    # --- Phase 1: Grid Search (Standard) ---
    configs = []
    for n_bits in [256, 384, 512]:
        for norm in [False, True]:
            for scale in [1.0, 2.0]:
                for n_iter in [50, 100, 200]:
                    for init in ['pca', 'random']:
                        configs.append(('standard', n_bits, norm, scale, n_iter, init))
    
    log(f"\nPhase 1: {len(configs)} standard configs")
    for idx, (method, n_bits, norm, scale, n_iter, init) in enumerate(configs):
        log(f"\n[{idx+1}/{len(configs)}] standard | bits={n_bits} norm={norm} scale={scale} iter={n_iter} init={init}")
        try:
            res = method_standard(X, n_bits, n_iter, norm, scale, init)
            log(f"  FULL R@1 = {res['r1_full']:.2%}")
            results.append({'method':'standard', 'bits':n_bits, 'norm':norm, 'scale':scale,
                           'iter':n_iter, 'init':init, 'r1':res['r1_full']})
            if res['r1_full'] > best_r1:
                best_r1, best_model = res['r1_full'], res
                log(f"  *** NEW BEST: {best_r1:.2%} ***")
                np.savez(BEST_MODEL, **{k:v for k,v in res.items() if k!='r1_sample'})
        except Exception as e:
            log(f"  ERROR: {e}")
    
    # --- Phase 2: Alternative Methods ---
    alts = [
        ('balanced', 512, True, 100),
        ('graph', 512, 100),
        ('l2pca', 512, 100),
        ('double', 512, 100),
        ('rff', 512, 100),
    ]
    if best_model:
        alts.append(('refinement', best_model, 100))
    
    log(f"\nPhase 2: {len(alts)} alternative methods")
    for name, *args in alts:
        log(f"\nMethod: {name}")
        try:
            if name == 'balanced':
                res = method_balanced(X, *args)
            elif name == 'graph':
                res = method_graph(X, *args)
            elif name == 'l2pca':
                res = method_l2pca(X, *args)
            elif name == 'double':
                res = method_double(X, *args)
            elif name == 'rff':
                res = method_rff(X, *args)
            elif name == 'refinement':
                res = method_refinement(X, *args)
            log(f"  FULL R@1 = {res['r1_full']:.2%}")
            results.append({'method':name, 'r1':res['r1_full']})
            if res['r1_full'] > best_r1:
                best_r1, best_model = res['r1_full'], res
                log(f"  *** NEW BEST: {best_r1:.2%} ***")
                np.savez(BEST_MODEL, **{k:v for k,v in res.items() if not isinstance(v, (type(None), str)) or k=='note'})
        except Exception as e:
            log(f"  ERROR: {e}")
    
    # --- Summary ---
    log("\n" + "="*60)
    log("SEARCH COMPLETE")
    log("="*60)
    log(f"Best R@1: {best_r1:.2%}")
    log(f"Best model: {BEST_MODEL}")
    
    # Save all results
    with open(RESULTS_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    # Top 10
    results.sort(key=lambda x: x['r1'], reverse=True)
    log("\nTop 10 results:")
    for i, r in enumerate(results[:10]):
        log(f"  {i+1}. {r['method']:12s} R@1={r['r1']:.2%}  { {k:v for k,v in r.items() if k not in ('method','r1')} }")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s,f: (log("Interrupted by user."), sys.exit(0)))
    main()
