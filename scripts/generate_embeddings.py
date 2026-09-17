#!/usr/bin/env python3
"""
Generate paper embeddings from your existing SQLite DB.
Uses title + abstract (or content/full_text if abstract missing).
Outputs: data/paper_embeddings.npy
"""
import sqlite3
import numpy as np
import os
import sys

# Try the engine's primary DB first, then fallback DBs
DB_CANDIDATES = [
    "data/phoenix_arxiv_1m.db",
    "papers.db",
    "data/papers.db",
    "data/papers_arxiv_1m.db",
    "data/droplet_papers_106298.db",
]
OUTPUT = "data/paper_embeddings.npy"
BATCH_SIZE = 256

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[embed] pip install sentence-transformers")
    sys.exit(1)

model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
print(f"[embed] Model loaded: all-MiniLM-L6-v2 ({model.get_sentence_embedding_dimension()} dims)")


def find_db():
    for path in DB_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def get_texts_from_db(path):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    # Find a table with title/abstract or full text
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    table = None
    for candidate in ["papers", "archive_papers"]:
        if candidate in tables:
            table = candidate
            break
    if table is None:
        raise ValueError(f"No known paper table found in {path}. Tables: {tables}")

    cursor.execute(f"PRAGMA table_info({table})")
    cols = {row[1].lower() for row in cursor.fetchall()}
    print(f"[embed] Using table '{table}' with columns: {sorted(cols)}")

    has_title = 'title' in cols
    has_abstract = 'abstract' in cols
    has_content = 'content' in cols or 'full_text' in cols

    select_cols = []
    if has_title:
        select_cols.append("title")
    if has_abstract:
        select_cols.append("abstract")
    if has_content:
        select_cols.append("content" if 'content' in cols else "full_text")

    if not select_cols:
        raise ValueError("No text columns found in table")

    cursor.execute(f"SELECT {', '.join(select_cols)} FROM {table} WHERE title IS NOT NULL")
    rows = cursor.fetchall()

    texts = []
    for row in rows:
        parts = []
        for i, col in enumerate(select_cols):
            val = row[i]
            if val:
                parts.append(str(val))
        if parts:
            # Prioritize title + abstract; use content as fallback
            if has_title and has_abstract and len(parts) >= 2:
                text = f"{parts[0]}. {parts[1]}"
            else:
                text = " ".join(parts)
            texts.append(text)

    conn.close()
    print(f"[embed] Loaded {len(texts)} papers from {path}")
    return texts


def main():
    db_path = find_db()
    if db_path is None:
        print("[embed] No known paper database found. Searched:")
        for p in DB_CANDIDATES:
            print(f"  - {p}")
        sys.exit(1)

    os.makedirs("data", exist_ok=True)

    texts = get_texts_from_db(db_path)

    print(f"[embed] Encoding {len(texts)} papers in batches of {BATCH_SIZE}...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(OUTPUT, embeddings.astype(np.float32))
    print(f"[embed] Saved {embeddings.shape} to {OUTPUT}")
    print(f"[embed] Now run: python3 scripts/train_itq_512.py")


if __name__ == "__main__":
    main()
