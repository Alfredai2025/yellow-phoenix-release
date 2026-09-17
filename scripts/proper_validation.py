#!/usr/bin/env python3
"""
PROPER VALIDATION: Test all fixes for quality + speed + robustness.
"""
import sys, os, random, time, statistics, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yp_bridge import YPEngine

random.seed(42)
engine = YPEngine()

# ── Load helpers ──
def load_spell():
    import json
    try:
        with open('data/spell_vocab.json', 'r') as f:
            return json.load(f)['vocab']
    except:
        return set()

def load_expansions():
    import json
    try:
        with open('data/query_expansions.json', 'r') as f:
            return json.load(f)
    except:
        return {}

def correct_word(word, vocab, freq):
    if word in vocab:
        return word
    # One-edit candidates
    letters = 'abcdefghijklmnopqrstuvwxyz'
    candidates = set()
    for i in range(len(word)):
        candidates.add(word[:i] + word[i+1:])  # delete
        if i < len(word) - 1:
            candidates.add(word[:i] + word[i+1] + word[i] + word[i+2:])  # transpose
        for c in letters:
            candidates.add(word[:i] + c + word[i+1:])  # replace
            candidates.add(word[:i] + c + word[i:])   # insert
    valid = candidates & set(vocab)
    if valid:
        return max(valid, key=lambda w: freq.get(w, 0))
    return word

def correct_text(text, vocab, freq):
    def repl(m):
        return correct_word(m.group(0).lower(), vocab, freq)
    return re.sub(r"[a-zA-Z]+", repl, text)

def expand_text(text, expansions):
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    words = {w for w in words if len(w) > 2}
    added = set()
    for w in words:
        for related in expansions.get(w, []):
            if related not in words and related not in added:
                added.add(related)
                if len(added) >= 3:
                    break
        if len(added) >= 3:
            break
    if added:
        return text + ' ' + ' '.join(sorted(added))
    return text

# ── Load data ──
print("=" * 65)
print("PROPER VALIDATION")
print("=" * 65)

vocab_data = load_spell()
vocab = set(vocab_data) if isinstance(vocab_data, list) else vocab_data
freq = {}
try:
    import json
    with open('data/spell_vocab.json', 'r') as f:
        freq = json.load(f).get('freq', {})
except:
    pass

expansions = load_expansions()
papers = list(engine.cache.items())
sample = random.sample(papers, 500)

print(f"\nLoaded: {len(vocab)} vocab words, {len(expansions)} expansions")
print(f"Sample: {len(sample)} papers")

# ── Perturbation functions ──
def delete_words(text, ratio=0.3):
    words = text.split()
    keep = max(1, int(len(words) * (1 - ratio)))
    return ' '.join(random.sample(words, keep))

def char_swap(text, n=3):
    chars = list(text)
    for _ in range(n):
        i = random.randint(0, len(chars) - 2)
        chars[i], chars[i+1] = chars[i+1], chars[i]
    return ''.join(chars)

def insert_noise(text, ratio=0.2):
    words = text.split()
    n = max(1, int(len(words) * ratio))
    noise = ['X', 'NOISE', 'FOO']
    for _ in range(n):
        words.insert(random.randint(0, len(words)), random.choice(noise))
    return ' '.join(words)

PERTURBATIONS = [
    ('Original', lambda t: t),
    ('Word deletion 30%', delete_words),
    ('Char swap (3)', char_swap),
    ('Noise insertion 20%', insert_noise),
    ('Combined (del+noise)', lambda t: insert_noise(delete_words(t, 0.2), 0.15)),
]

# ── Test configurations ──
configs = [
    ('Baseline (no fixes)', lambda t: t, False),
    ('+ Spell check', lambda t: correct_text(t, vocab, freq), False),
    ('+ Query expansion', lambda t: expand_text(t, expansions), False),
    ('+ Spell + Expand', lambda t: expand_text(correct_text(t, vocab, freq), expansions), False),
]

# ── Run ──
print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)

for config_name, preprocess, _ in configs:
    print(f"\n{'─' * 65}")
    print(f"CONFIG: {config_name}")
    print('─' * 65)
    
    for pert_label, pert_fn in PERTURBATIONS:
        r1_hits = 0
        r5_hits = 0
        latencies = []
        
        for true_pid, title in sample:
            corrupted = pert_fn(title)
            query = preprocess(corrupted)
            
            t0 = time.perf_counter()
            try:
                res = engine.search(query, top_k=5)
            except:
                res = []
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1_000_000)
            
            result_pids = []
            for r in res or []:
                payload = r[1] if len(r) > 1 else None
                if isinstance(payload, tuple):
                    result_pids.append(payload[0])
                elif payload is not None:
                    result_pids.append(payload)
            
            if true_pid in result_pids[:1]:
                r1_hits += 1
            if true_pid in result_pids[:5]:
                r5_hits += 1
        
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        print(f"  {pert_label:<25} | R@1: {r1_hits/len(sample)*100:>5.1f}% | R@5: {r5_hits/len(sample)*100:>5.1f}% | P50: {p50:>7.1f} µs")

# ── Fast path sanity check ──
print("\n" + "=" * 65)
print("FAST PATH SANITY CHECK")
print("=" * 65)

# 1. Exact title should be FAST
exact_times = []
for _, title in random.sample(papers, 100):
    t0 = time.perf_counter()
    res = engine.search(title, top_k=1)
    t1 = time.perf_counter()
    exact_times.append((t1 - t0) * 1_000_000)
exact_times.sort()
print(f"  Exact title P50: {exact_times[50]:.1f} µs (target: < 10 µs) {'✅' if exact_times[50] < 10 else '❌'}")

# 2. Non-title query should NOT hit fast path (should be ~5 ms)
nonsense = "quantum entanglement in biological neural networks during sleep"
t0 = time.perf_counter()
res = engine.search(nonsense, top_k=1)
t1 = time.perf_counter()
nonsense_lat = (t1 - t0) * 1_000_000
print(f"  Nonsense query:  {nonsense_lat:.1f} µs (should be ~5,000 µs) {'✅' if nonsense_lat > 1000 else '⚠️ suspiciously fast'}")

# 3. Partial title (first 40 chars) should still hit fast path
partial_times = []
for _, title in random.sample(papers, 100):
    partial = title[:40]
    t0 = time.perf_counter()
    res = engine.search(partial, top_k=1)
    t1 = time.perf_counter()
    partial_times.append((t1 - t0) * 1_000_000)
partial_times.sort()
print(f"  Partial title P50: {partial_times[50]:.1f} µs (target: < 50 µs) {'✅' if partial_times[50] < 50 else '❌'}")

print("\n" + "=" * 65)
print("VALIDATION COMPLETE")
print("=" * 65)
