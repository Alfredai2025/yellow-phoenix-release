#!/usr/bin/env python3
import json
import os
import time
from collections import defaultdict
from datetime import datetime

WINDOW_SIZE = 2000
MIN_QUERIES = 500
TUNE_INTERVAL_SEC = 3600
STATE_FILE = "scripts/predictor_sweep/auto_tuner_state.json"
QUERY_BUCKET_LOG = "logs/query_bucket_log.jsonl"


def tokenize(text):
    return text.lower().split()


def hash_token(s):
    return hash(s) & 0xFFFFFFFFFFFFFFFF


class TokenPredictor:
    def __init__(self, confidence=0.65, min_votes=2):
        self.token_index = defaultdict(list)
        self.bigram_index = defaultdict(list)
        self.char3_index = defaultdict(list)
        self.confidence = confidence
        self.min_votes = min_votes

    def learn(self, query, bucket):
        tokens = tokenize(query)
        for t in tokens:
            self.token_index[hash_token(t)].append((bucket, 1))
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            self.bigram_index[hash_token(bigram)].append((bucket, 1))
        for t in tokens:
            if len(t) >= 3:
                for j in range(len(t) - 2):
                    tri = t[j:j+3]
                    self.char3_index[hash_token(tri)].append((bucket, 1))

    def predict(self, query):
        tokens = tokenize(query)
        votes = defaultdict(float)

        for t in tokens:
            h = hash_token(t)
            if h in self.token_index:
                for b, f in self.token_index[h]:
                    votes[b] += 0.3 * f

        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            h = hash_token(bigram)
            if h in self.bigram_index:
                for b, f in self.bigram_index[h]:
                    votes[b] += 0.7 * f

        for t in tokens:
            h = hash_token(t)
            token_missed = h not in self.token_index
            if token_missed and len(t) >= 3:
                for j in range(len(t) - 2):
                    tri = t[j:j+3]
                    h3 = hash_token(tri)
                    if h3 in self.char3_index:
                        for b, f in self.char3_index[h3]:
                            votes[b] += 0.4 * f

        if len(votes) < self.min_votes:
            return None

        total = sum(votes.values())
        best_bucket = max(votes, key=lambda k: votes[k])
        best_score = votes[best_bucket]
        confidence = best_score / total

        if confidence >= self.confidence:
            return best_bucket, confidence
        return None


def load_window(log_path, window_size):
    if not os.path.exists(log_path):
        return []
    lines = []
    with open(log_path) as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
    return lines[-window_size:]


def sweep(entries):
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    best_score = -1
    best_thresh = 0.65
    split = int(len(entries) * 0.8)
    train = entries[:split]
    test = entries[split:]

    for thresh in thresholds:
        pred = TokenPredictor(confidence=thresh, min_votes=2)
        for e in train:
            pred.learn(e["query"], e["bucket"])

        hits = 0
        confidences = []
        for e in test:
            result = pred.predict(e["query"])
            if result:
                bucket, conf = result
                if bucket == e["bucket"]:
                    hits += 1
                confidences.append(conf)

        hit_rate = hits / len(test) if test else 0
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        score = hit_rate * avg_conf

        if score > best_score:
            best_score = score
            best_thresh = thresh

    return best_thresh


def run(log_path, bridge=None):
    entries = load_window(log_path, WINDOW_SIZE)
    if len(entries) < MIN_QUERIES:
        print(f"[auto_tuner] Only {len(entries)} queries, need {MIN_QUERIES}. Skipping.")
        return None

    best_thresh = sweep(entries)

    if bridge:
        bridge.set_threshold(best_thresh, 2)
        print(f"[auto_tuner] Updated predictor threshold to {best_thresh:.2f}")

    state = {
        "last_run": datetime.now().isoformat(),
        "queries_evaluated": len(entries),
        "best_threshold": best_thresh,
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

    print(f"[auto_tuner] Best threshold: {best_thresh:.2f}")
    return best_thresh


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python auto_tuner.py <query_bucket_log.jsonl>")
        sys.exit(1)
    run(sys.argv[1])
