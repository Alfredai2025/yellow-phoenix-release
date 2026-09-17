#!/usr/bin/env python3
"""Split ArXiv papers into domain shards and build HNSW indexes.

Uses keyword-based classification on titles because the pickle files do not
contain arXiv category metadata.
"""

import json
import os
import pickle
import struct
import subprocess
import sys

# Keywords are matched in lower-cased titles.
DOMAIN_KEYWORDS = {
    "cs": [
        "neural", "deep learning", "algorithm", "network", "optimization",
        "machine learning", "computer vision", "nlp", "transformer", "gpt",
        "reinforcement learning", "classification", "regression", "clustering",
        "convolutional", "recurrent", "language model", "graph neural",
        "generative", "adversarial", "dataset", "benchmark",
    ],
    "medical": [
        "cancer", "tumor", "protein", "gene", "clinical", "patient",
        "drug", "therapy", "disease", "biomarker", "cell", "virus",
        "bacteria", "genome", "treatment", "diagnosis", "pathology",
        "medical", "health", "hospital", "surgical", "mri", "ct scan",
    ],
}


def classify_title(title: str) -> str:
    lower = title.lower()
    scores = {"cs": 0, "medical": 0}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                scores[domain] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def write_yism(path: str, records):
    """records: list of (id: int, hash: bytes)"""
    with open(path, "wb") as f:
        f.write(b"YISM")
        f.write(struct.pack("<B", 1))  # version
        f.write(struct.pack("<Q", len(records)))  # count
        f.write(struct.pack("<B", 64))  # hash_len
        for rid, h in records:
            f.write(struct.pack("<Q", rid))
            f.write(h)


def write_titles(path: str, title_records):
    """title_records: list of {"id": int, "title": str, "pid": str}"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(title_records, f)


def build_hnsw(yism_path: str, hnsw_path: str):
    subprocess.run(
        [
            "cargo", "run", "--release", "--bin", "shootout_build_hnsw_106k", "--",
            yism_path, hnsw_path,
        ],
        cwd=os.path.expanduser("~/yellow_phoenix"),
        check=True,
    )


def main():
    hash_path = os.path.expanduser("~/yellow_phoenix/data/paper_hashes_arxiv_1m.pkl")
    meta_path = os.path.expanduser("~/yellow_phoenix/data/paper_meta_arxiv_1m.pkl")
    out_dir = os.path.expanduser("~/yellow_phoenix_mobile/YPPhone/Resources")

    print("Loading hashes...")
    with open(hash_path, "rb") as f:
        hash_data = pickle.load(f)  # pid -> bytes

    print("Loading metadata...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    pids = meta["pids"]
    titles = meta["titles"]
    pid_to_title = dict(zip(pids, titles))

    # Assign numeric IDs in the order they appear in the metadata list.
    # This keeps IDs deterministic and matches emb_idx if the list is aligned.
    shards = {"cs": [], "medical": [], "general": []}
    title_shards = {"cs": [], "medical": [], "general": []}

    print("Classifying...")
    for idx, pid in enumerate(pids):
        h = hash_data.get(pid)
        if h is None:
            continue
        title = pid_to_title.get(pid, "")
        domain = classify_title(title)
        shards[domain].append((idx, h))
        title_shards[domain].append({"id": idx, "title": title, "pid": pid})

    print("\nShard counts:")
    for domain in ["cs", "medical", "general"]:
        print(f"  {domain}: {len(shards[domain])} papers")

    for domain in ["cs", "medical", "general"]:
        yism_path = os.path.join(out_dir, f"yp_edge_{domain}.bin")
        hnsw_path = os.path.join(out_dir, f"binary_hnsw_{domain}.bin")
        titles_path = os.path.join(out_dir, f"titles_{domain}.json")

        write_yism(yism_path, shards[domain])
        write_titles(titles_path, title_shards[domain])
        print(f"\nBuilding {domain} HNSW...")
        build_hnsw(yism_path, hnsw_path)
        print(f"  Wrote {hnsw_path}")

    print("\nAll domain shards built.")


if __name__ == "__main__":
    main()
