#!/usr/bin/env python3
"""Train a supervised linear model: TF-IDF bag-of-words -> 512-bit ITQ hash."""

import os
import numpy as np
import sqlite3
from scipy.special import expit as sigmoid
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

N_TOTAL = 50_000
N_TRAIN = 10_000
N_VAL = 2_000
VOCAB_SIZE = 50_000
LR = 0.5
EPOCHS = 30
BATCH_SIZE = 1_000

print("[+] Loading data...")
conn = sqlite3.connect('data/phoenix_arxiv_1m.db')
cur = conn.cursor()
cur.execute("SELECT title, abstract FROM papers LIMIT ?", (N_TOTAL,))
texts = [f"{t or ''} {a or ''}".strip() for t, a in cur.fetchall()]
conn.close()

print("[+] Computing true ITQ hashes...")
# Use local model to avoid HuggingFace timeouts
model = SentenceTransformer('models/all-MiniLM-L6-v2', local_files_only=True)
itq = np.load('data/itq_model_512.npz')
mean = itq['mean'].astype(np.float32)
M = itq['proj'].astype(np.float32)

embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
centered = embs - mean
projected = centered @ M  # N x 512
true_hashes = (projected > 0).astype(np.float32)  # N x 512

print("[+] Building TF-IDF vectors...")
vectorizer = TfidfVectorizer(max_features=VOCAB_SIZE, min_df=2, dtype=np.float32)
X = vectorizer.fit_transform(texts)  # N x V sparse
vocab = vectorizer.get_feature_names_out()
V = X.shape[1]
print(f"    Vocabulary: {V}")

# Split
X_train = X[:N_TRAIN]
H_train = true_hashes[:N_TRAIN]
X_val = X[N_TRAIN:N_TRAIN + N_VAL]
H_val = true_hashes[N_TRAIN:N_TRAIN + N_VAL]

# Initialize weights
W = np.zeros((V, 512), dtype=np.float32)
b = np.zeros(512, dtype=np.float32)

N_batches = N_TRAIN // BATCH_SIZE

def compute_hamming(Xs, Hs):
    Z = Xs @ W + b
    pred = (Z > 0).astype(np.float32)
    return np.mean(np.abs(pred - Hs)) * 512

print("[+] Training linear hash model...")
initial_ham = compute_hamming(X_val, H_val)
print(f"    Epoch 0: val_hamming={initial_ham:.1f} bits")

for epoch in range(EPOCHS):
    # Shuffle training data
    perm = np.random.permutation(N_TRAIN)
    X_train = X_train[perm]
    H_train = H_train[perm]
    
    epoch_loss = 0.0
    for i in range(N_batches):
        start = i * BATCH_SIZE
        end = start + BATCH_SIZE
        xb = X_train[start:end]  # sparse batch
        hb = H_train[start:end]  # dense batch
        
        # Forward
        Z = xb @ W + b  # B x 512
        P = sigmoid(Z)
        
        # BCE loss
        loss = -np.mean(hb * np.log(P + 1e-7) + (1 - hb) * np.log(1 - P + 1e-7))
        epoch_loss += loss
        
        # Gradient
        grad = (P - hb) / BATCH_SIZE  # B x 512
        W_grad = xb.T @ grad  # V x 512
        b_grad = grad.sum(axis=0)
        
        # Update
        W -= LR * W_grad
        b -= LR * b_grad
    
    val_ham = compute_hamming(X_val, H_val)
    print(f"    Epoch {epoch+1}: train_loss={epoch_loss/N_batches:.4f}, val_hamming={val_ham:.1f} bits")

print("[+] Saving model...")
np.savez('data/linear_hash_model.npz', W=W, b=b, vocab=vocab)
print("[+] Saved to data/linear_hash_model.npz")

# Final stats
final_ham = compute_hamming(X_val, H_val)
print(f"\nFinal validation Hamming: {final_ham:.1f} bits")
print(f"Final validation bit accuracy: {(512 - final_ham) / 512 * 100:.1f}%")
