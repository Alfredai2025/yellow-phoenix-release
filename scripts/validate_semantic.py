#!/usr/bin/env python3
"""
Semantic recall: query by paraphrased topic, not exact title.
Tests real-world search quality.
"""
import sqlite3
import random
import sys
import re

sys.path.insert(0, ".")
from yp_bridge import YPEngine

DB_PATH = "data/phoenix_arxiv_1m.db"
N_TEST = 200

# Paraphrase strategies
def paraphrase_drop_start(title):
    """Drop first 1-3 words — tests if engine finds paper from partial topic."""
    words = title.lower().split()
    if len(words) < 5:
        return title
    drop = random.randint(1, min(3, len(words) - 2))
    return " ".join(words[drop:])

def paraphrase_keywords(title):
    """Extract 2-4 content words — tests keyword search."""
    words = re.findall(r'\b[a-z]{4,}\b', title.lower())
    if len(words) < 3:
        return title
    k = random.randint(2, min(4, len(words)))
    return " ".join(random.sample(words, k))

def paraphrase_vague(title):
    """Replace specific terms with generic synonyms."""
    vague = title.lower()
    replacements = {
        "neural": "brain-like",
        "network": "system",
        "deep": "advanced",
        "learning": "training",
        "optimization": "improvement",
        "algorithm": "method",
        "classification": "categorization",
        "regression": "prediction",
        "transformer": "attention model",
        "convolutional": "filter-based",
    }
    for old, new in replacements.items():
        vague = vague.replace(old, new)
    return vague

STRATEGIES = [
    ("partial_topic", paraphrase_drop_start),
    ("keywords", paraphrase_keywords),
    ("vague", paraphrase_vague),
]


def main():
    random.seed(42)
    engine = YPEngine()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, title FROM papers WHERE title IS NOT NULL ORDER BY RANDOM() LIMIT ?", (N_TEST,))
    papers = cur.fetchall()
    conn.close()

    print(f"[semantic] Testing {len(papers)} papers with 3 strategies each...\n")

    for name, fn in STRATEGIES:
        hits = {1: 0, 5: 0, 10: 0}
        total = 0

        for pid, title in papers:
            query = fn(title)
            if query == title:
                continue  # skip if paraphrase didn't change anything
            total += 1
            try:
                results = engine.search(query, top_k=10)
                result_pids = [r[1][0] for r in results] if results else []
                for k in [1, 5, 10]:
                    if pid in result_pids[:k]:
                        hits[k] += 1
            except Exception as e:
                pass

        print(f"=== {name.upper()} ===")
        for k in [1, 5, 10]:
            rate = hits[k] / total if total > 0 else 0
            print(f"  R@{k}: {rate:.3%} ({hits[k]}/{total})")
        print()


if __name__ == "__main__":
    main()
