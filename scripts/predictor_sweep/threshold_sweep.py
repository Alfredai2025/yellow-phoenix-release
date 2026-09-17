#!/usr/bin/env python3
import json
import sys
from collections import defaultdict

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
        has_exact = False
        for t in tokens:
            h = hash_token(t)
            if h in self.token_index:
                has_exact = True
                for b, f in self.token_index[h]:
                    votes[b] += 0.3 * f
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            h = hash_token(bigram)
            if h in self.bigram_index:
                has_exact = True
                for b, f in self.bigram_index[h]:
                    votes[b] += 0.7 * f
        if not has_exact:
            for t in tokens:
                if len(t) >= 3:
                    for j in range(len(t) - 2):
                        tri = t[j:j+3]
                        h = hash_token(tri)
                        if h in self.char3_index:
                            for b, f in self.char3_index[h]:
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

def sweep(train_path, test_path):
    train = [json.loads(l) for l in open(train_path)]
    test = [json.loads(l) for l in open(test_path)]
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    results = []
    for thresh in thresholds:
        pred = TokenPredictor(confidence=thresh, min_votes=2)
        for entry in train:
            pred.learn(entry["query"], entry["bucket"])
        hits = 0
        confidences = []
        for entry in test:
            result = pred.predict(entry["query"])
            if result:
                bucket, conf = result
                if bucket == entry["bucket"]:
                    hits += 1
                confidences.append(conf)
        hit_rate = hits / len(test) if test else 0
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        score = hit_rate * avg_conf
        results.append({"threshold": thresh, "hit_rate": hit_rate, "avg_confidence": avg_conf, "score": score, "hits": hits, "total": len(test)})
        print(f"thresh={thresh:.2f} | hit_rate={hit_rate:.3f} | avg_conf={avg_conf:.3f} | score={score:.3f} | {hits}/{len(test)}")
    best = max(results, key=lambda x: x["score"])
    print(f"\nBEST: threshold={best['threshold']:.2f} with score={best['score']:.3f}")
    return best["threshold"], results

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python threshold_sweep.py train.jsonl test.jsonl")
        sys.exit(1)
    sweep(sys.argv[1], sys.argv[2])
