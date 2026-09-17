#!/usr/bin/env python3
"""
Generate paper embeddings from data/phoenix_arxiv_1m.db using FULL TEXT (content).
Falls back to title+abstract if content is empty.
Outputs: data/paper_embeddings.npy
"""
import sqlite3
import numpy as np
import os
import sys

DB_PATH = "data/phoenix_arxiv_1m.db"
OUTPUT = "data/paper_embeddings.npy"
BATCH_SIZE = 64  # smaller batches for long full-text papers

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[embed] pip install sentence-transformers")
    sys.exit(1)

model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
print(f"[embed] Model loaded: all-MiniLM-L6-v2 ({model.get_sentence_embedding_dimension()} dims)")


def get_texts_from_db(path):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    if 'papers' not in tables:
        raise ValueError("No 'papers' table found")

    cursor.execute("PRAGMA table_info(papers)")
    cols = {row[1].lower() for row in cursor.fetchall()}
    print(f"[embed] Columns: {sorted(cols)}")

    # Priority: title + content (full text) > title + abstract > title only
    cursor.execute("""
        SELECT title, abstract, content
        FROM papers
        WHERE title IS NOT NULL
    """)
    rows = cursor.fetchall()
    conn.close()

    texts = []
    used_source = []
    for title, abstract, content in rows:
        t = title or ""
        a = abstract or ""
        c = content or ""

        if len(c) > 200:
            # Truncate full text to first 2000 chars to keep encoding fast
            text = f"{t}. {c[:2000]}"
            source = "full_text"
        elif len(a) > 50:
            text = f"{t}. {a}"
            source = "abstract"
        else:
            text = t
            source = "title_only"

        texts.append(text)
        used_source.append(source)

    counts = {
        "full_text": used_source.count("full_text"),
        "abstract": used_source.count("abstract"),
        "title_only": used_source.count("title_only"),
    }
    print(f"[embed] Source breakdown: {counts}")
    print(f"[embed] Loaded {len(texts)} papers from {path}")
    return texts


def main():
    if not os.path.exists(DB_PATH):
        print(f"[embed] DB not found at {DB_PATH}")
        sys.exit(1)

    os.makedirs("data", exist_ok=True)

    texts = get_texts_from_db(DB_PATH)

    print(f"[embed] Encoding {len(texts)} papers (full-text, batch={BATCH_SIZE})...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(OUTPUT, embeddings.astype(np.float32))
    print(f"[embed] Saved {embeddings.shape} to {OUTPUT}")
    print(f"[embed] Next: python3 scripts/train_itq_512.py")


if __name__ == "__main__":
    main()
