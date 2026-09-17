#!/usr/bin/env python3
"""
Build a lightweight spell-checker from paper titles.
No external deps — uses edit distance against title vocabulary.
"""
import sys, os, json, re, collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yp_bridge import YPEngine

print("=" * 65)
print("BUILDING SPELL CHECKER")
print("=" * 65)

# ── Load all titles ──
print("\n[1/2] Loading titles...")
engine = YPEngine()
titles = list(engine.cache.values())
print(f"      Titles: {len(titles)}")

# ── Build vocabulary ──
print("\n[2/2] Building vocabulary...")
word_freq = collections.Counter()
for t in titles:
    words = re.findall(r"[a-zA-Z]+", t.lower())
    for w in words:
        if len(w) > 2:
            word_freq[w] += 1

# Keep words that appear at least twice (filters typos/noise)
vocab = {w for w, c in word_freq.items() if c >= 2}
print(f"      Vocabulary size: {len(vocab)}")
print(f"      Top 10 words: {word_freq.most_common(10)}")

# Save
out = {
    'vocab': sorted(vocab),
    'freq': dict(word_freq.most_common(5000)),
}
os.makedirs('data', exist_ok=True)
with open('data/spell_vocab.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved: data/spell_vocab.json")
