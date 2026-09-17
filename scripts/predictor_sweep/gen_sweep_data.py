#!/usr/bin/env python3
import json
import random
import sys

TOPICS = [
    ("neural network", [40000, 41000, 42000]),
    ("deep learning", [41000, 41500, 42000]),
    ("gradient descent", [10000, 10500, 11000]),
    ("optimization", [30000, 31000, 32000]),
    ("transformer", [50000, 50500, 51000]),
    ("attention mechanism", [51000, 51500, 52000]),
    ("reinforcement learning", [20000, 21000, 22000]),
    ("computer vision", [60000, 61000, 62000]),
    ("natural language", [70000, 71000, 72000]),
    ("graph neural", [80000, 81000, 82000]),
]

def make_query(topic):
    prefixes = ["", "advanced ", "introduction to ", "survey of ", ""]
    suffixes = ["", " methods", " techniques", " algorithms", ""]
    return f"{random.choice(prefixes)}{topic}{random.choice(suffixes)}"

def generate(count, out_path, seed=42):
    random.seed(seed)
    with open(out_path, "w") as f:
        for _ in range(count):
            topic, buckets = random.choice(TOPICS)
            query = make_query(topic)
            if random.random() < 0.1:
                query = query.replace("neural", "neuro")
            bucket = random.choice(buckets)
            f.write(json.dumps({"query": query, "bucket": bucket}) + "\n")
    print(f"Wrote {count} lines to {out_path}")

if __name__ == "__main__":
    generate(int(sys.argv[1]), sys.argv[2])
