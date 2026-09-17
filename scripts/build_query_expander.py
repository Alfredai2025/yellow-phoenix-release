#!/usr/bin/env python3
"""
Build query expansion from title word co-occurrence.
"""
import sys, os, json, re, collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yp_bridge import YPEngine

print("=" * 65)
print("BUILDING QUERY EXPANDER")
print("=" * 65)

# ── Load titles ──
print("\n[1/2] Loading titles...")
engine = YPEngine()
titles = list(engine.cache.values())
print(f"      Titles: {len(titles)}")

# ── Build co-occurrence ──
print("\n[2/2] Building co-occurrence...")
cooccur = collections.defaultdict(collections.Counter)

for t in titles:
    words = list(set(re.findall(r"[a-zA-Z]+", t.lower())))
    words = [w for w in words if len(w) > 2]
    for i, w1 in enumerate(words):
        for w2 in words[i+1:]:
            cooccur[w1][w2] += 1
            cooccur[w2][w1] += 1

# For each word, keep top 5 co-occurring words
expansions = {}
for word, counter in cooccur.items():
    top = [w for w, c in counter.most_common(5) if c >= 2]
    if top:
        expansions[word] = top

print(f"      Expansion entries: {len(expansions)}")
print(f"      Example 'neural': {expansions.get('neural', [])[:5]}")

os.makedirs('data', exist_ok=True)
with open('data/query_expansions.json', 'w') as f:
    json.dump(expansions, f, indent=2)
print(f"\n  Saved: data/query_expansions.json")
