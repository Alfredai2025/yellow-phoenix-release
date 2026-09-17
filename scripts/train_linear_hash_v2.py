#!/usr/bin/env python3
"""Train linear hash model with binary bag-of-words to match Rust inference."""

import numpy as np, sqlite3, struct, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

N_TOTAL = 50_000
N_TRAIN = 10_000
N_VAL = 2_000
VOCAB_SIZE = 50_000
LR = 0.5
EPOCHS = 30
BATCH_SIZE = 1_000

def tokenize(text):
    return text.lower().split()

print("[+] Loading data...")
conn = sqlite3.connect('data/phoenix_arxiv_1m.db')
cur = conn.cursor()
cur.execute("SELECT title, abstract FROM papers LIMIT ?", (N_TOTAL,))
texts = [f"{t or ''} {a or ''}".strip() for t, a in cur.fetchall()]
conn.close()

print("[+] Computing true ITQ hashes...")
model = SentenceTransformer('models/all-MiniLM-L6-v2', local_files_only=True)
itq = np.load('data/itq_model_512.npz')
mean = itq['mean'].astype(np.float32)
M = itq['proj'].astype(np.float32)
embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
projected = (embs - mean) @ M
true_hashes = (projected > 0).astype(np.float32)

print("[+] Building binary bag-of-words vectors...")
# Use a custom tokenizer that exactly matches Rust: lowercase + whitespace split
vectorizer = TfidfVectorizer(max_features=VOCAB_SIZE, min_df=2, binary=True, tokenizer=tokenize, preprocessor=None, lowercase=False, token_pattern=None)
X = vectorizer.fit_transform(texts)
vocab = vectorizer.get_feature_names_out()
V = X.shape[1]
print(f"    Vocabulary: {V}")

X_train = X[:N_TRAIN]; H_train = true_hashes[:N_TRAIN]
X_val = X[N_TRAIN:N_TRAIN+N_VAL]; H_val = true_hashes[N_TRAIN:N_TRAIN+N_VAL]

W = np.zeros((V, 512), dtype=np.float32)
b = np.zeros(512, dtype=np.float32)

def compute_hamming(Xs, Hs):
    Z = Xs @ W + b
    pred = (Z > 0).astype(np.float32)
    return np.mean(np.abs(pred - Hs)) * 512

from scipy.special import expit as sigmoid
print("[+] Training...")
print(f"    Epoch 0: val_hamming={compute_hamming(X_val, H_val):.1f} bits")
N_batches = N_TRAIN // BATCH_SIZE
for epoch in range(EPOCHS):
    perm = np.random.permutation(N_TRAIN)
    X_train = X_train[perm]; H_train = H_train[perm]
    epoch_loss = 0.0
    for i in range(N_batches):
        start = i * BATCH_SIZE; end = start + BATCH_SIZE
        xb = X_train[start:end]; hb = H_train[start:end]
        Z = xb @ W + b
        P = sigmoid(Z)
        loss = -np.mean(hb * np.log(P + 1e-7) + (1 - hb) * np.log(1 - P + 1e-7))
        epoch_loss += loss
        grad = (P - hb) / BATCH_SIZE
        W -= LR * (xb.T @ grad)
        b -= LR * grad.sum(axis=0)
    print(f"    Epoch {epoch+1}: train_loss={epoch_loss/N_batches:.4f}, val_hamming={compute_hamming(X_val, H_val):.1f} bits")

final_ham = compute_hamming(X_val, H_val)
print(f"\nFinal validation Hamming: {final_ham:.1f} bits")
print(f"Final bit accuracy: {(512 - final_ham) / 512 * 100:.1f}%")

np.savez('data/linear_hash_model_v2.npz', W=W, b=b, vocab=vocab)
print("[+] Saved to data/linear_hash_model_v2.npz")

# Export binary
with open('data/linear_hash_weights_v2.bin', 'wb') as f:
    f.write(b'YPLH')
    f.write(struct.pack('<I', 1))
    f.write(struct.pack('<I', V))
    f.write(struct.pack('<I', 512))
    b.astype(np.float32).tofile(f)
    for i, tok in enumerate(vocab):
        tok_b = tok.encode('utf-8')
        f.write(struct.pack('<H', len(tok_b)))
        f.write(tok_b)
        W[i].astype(np.float32).tofile(f)
size_mb = os.path.getsize('data/linear_hash_weights_v2.bin') / 1024 / 1024
print(f"[+] Exported data/linear_hash_weights_v2.bin ({size_mb:.2f} MB)")
