#!/usr/bin/env python3
"""
Mine hard negatives from query logs.
Hard negative = query lands in bucket X, but mesh/cascade found a better paper.
"""
import json
import os
from collections import defaultdict

QUERY_LOG = "logs/query_bucket_log.jsonl"
OUTPUT = "data/hard_negatives.jsonl"
os.makedirs("data", exist_ok=True)


def load_queries_by_bucket(path):
    buckets = defaultdict(list)
    if not os.path.exists(path):
        return buckets
    with open(path) as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                buckets[e["bucket"]].append(e["query"])
    return buckets


def mine(log_path=QUERY_LOG, out_path=OUTPUT):
    buckets = load_queries_by_bucket(log_path)
    hard_negatives = []
    seen = set()

    for bucket, queries in buckets.items():
        if len(queries) < 2:
            continue
        for i, q1 in enumerate(queries):
            for q2 in queries[i+1:]:
                if q1 == q2:
                    continue
                key = tuple(sorted([q1, q2]))
                if key in seen:
                    continue
                seen.add(key)
                hard_negatives.append({
                    "query": q1,
                    "positive": q2,
                    "negative": None,
                    "source": "same_bucket_different_query"
                })

    with open(out_path, "w") as f:
        for hn in hard_negatives:
            f.write(json.dumps(hn) + "\n")

    print(f"[hard_negatives] Mined {len(hard_negatives)} pairs → {out_path}")
    return len(hard_negatives)


if __name__ == "__main__":
    mine()
