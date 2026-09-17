import numpy as np, os

EMB_PATH = os.path.expanduser("~/Documents/11 Aug Yellow Back up/paper_embeddings.npy")
print("[+] Loading: " + EMB_PATH)
X = np.load(EMB_PATH).astype(np.float32)
N, d = X.shape
print("    Shape: {:,} x {}".format(N, d))

N_BITS = 512
N_ITER = 50

mean = X.mean(axis=0)
Xc = X - mean

_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
pca_dim = min(d, N_BITS)
W_pca = Vt[:pca_dim].T.astype(np.float32)

if pca_dim < N_BITS:
    pad = np.random.randn(d, N_BITS - pca_dim).astype(np.float32)
    pad, _ = np.linalg.qr(pad)
    W_pca = np.concatenate([W_pca, pad], axis=1)

X_pca = Xc @ W_pca

R = np.eye(N_BITS, dtype=np.float32)
for it in range(N_ITER):
    V = X_pca @ R
    B = np.sign(V)
    B[B == 0] = 1
    C = X_pca.T @ B
    U, _, Vt = np.linalg.svd(C)
    R_new = Vt.T @ U.T
    diff = np.linalg.norm(R - R_new)
    R = R_new
    if (it + 1) % 10 == 0:
        print("    Iter {:2d}: diff={:.6f}".format(it+1, diff))

B_bin = ((X_pca @ R) > 0).astype(np.uint8)
codes = np.packbits(B_bin, axis=1)

correct = 0
for i in range(N):
    xor = np.bitwise_xor(codes, codes[i])
    pc = np.unpackbits(xor, axis=1).sum(axis=1)
    pc[i] = 999999
    if pc.argmin() == i:
        correct += 1

r1 = correct / N
print("\n" + "="*50)
print("CORRECTED ITQ RESULT")
print("="*50)
print("R@1: {:.1%}".format(r1))

np.savez("data/itq_model_512_fixed.npz",
         W=W_pca, R=R, proj=W_pca @ R, mean=mean, n_bits=N_BITS)
codes.tofile("data/itq_codes_512_fixed.bin")
print("[+] Saved: data/itq_model_512_fixed.npz")
