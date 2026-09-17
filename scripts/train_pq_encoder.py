import os, numpy as np, sqlite3
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans

print("[+] Loading MiniLM and ITQ...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
itq = np.load('data/itq_model_512.npz')
mean = itq['mean']
M = itq['proj']  # 384 x 512

# Load tokens
print("[+] Loading tokens...")
conn = sqlite3.connect('data/phoenix_arxiv_1m.db')
cur = conn.cursor()
cur.execute("SELECT title, abstract FROM papers LIMIT 50000")
texts = [f"{t or ''} {a or ''}".strip() for t, a in cur.fetchall()]
conn.close()

tokens = list(set(t.lower() for text in texts for t in text.split()))
print(f"[+] {len(tokens)} unique tokens")

# Embed tokens
print("[+] Embedding tokens...")
embs = model.encode(tokens, convert_to_numpy=True, show_progress_bar=True)

# Center
embs_centered = embs - mean

# Train 4 k-means codebooks (one per quarter)
for chunk in range(4):
    bit_start = chunk * 128
    bit_end = bit_start + 128
    
    # Project tokens to this quarter's subspace
    projected = embs_centered @ M[:, bit_start:bit_end]  # (N, 128)
    
    # K-means with 128 clusters
    print(f"[+] K-means for chunk {chunk}...")
    kmeans = MiniBatchKMeans(n_clusters=128, batch_size=1000, random_state=42, n_init=3)
    labels = kmeans.fit_predict(projected)
    
    # Build P matrix: P[cluster][bit] = cluster center
    P = kmeans.cluster_centers_.astype(np.float32)  # (128, 128)
    P.tofile(f'data/pq_proj_{chunk}.f32')
    
    # Bias = 0 (mean already subtracted)
    bias = np.zeros(128, dtype=np.float32)
    bias.tofile(f'data/pq_bias_{chunk}.f32')
    
    print(f"[+] Chunk {chunk}: centers shape {P.shape}")

print("[+] Done.")
