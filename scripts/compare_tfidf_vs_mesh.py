#!/usr/bin/env python3
"""
Compare: Original keyword mesh vs TF-IDF mesh vs Strict Pinhole V2
"""
import sys, os, time, random, statistics, json, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yp_bridge import YPEngine

random.seed(42)

def delete_words(text, ratio=0.3):
    words = text.split()
    keep = max(1, int(len(words) * (1 - ratio)))
    keep = min(keep, len(words))
    return ' '.join(random.sample(words, keep))

def char_swap(text, n=2):
    chars = list(text)
    for _ in range(n):
        i = random.randint(0, len(chars) - 2)
        chars[i], chars[i+1] = chars[i+1], chars[i]
    return ''.join(chars)

def noise_insert(text, ratio=0.2):
    words = text.split()
    n = max(1, int(len(words) * ratio))
    noise = ['X', 'NOISE']
    for _ in range(n):
        words.insert(random.randint(0, len(words)), random.choice(noise))
    return ' '.join(words)

PERTURBATIONS = [
    ('Original', lambda t: t),
    ('Word deletion 30%', lambda t: delete_words(t, 0.3)),
    ('Char swap (2)', lambda t: char_swap(t, 2)),
    ('Noise insertion 20%', lambda t: noise_insert(t, 0.2)),
]

def benchmark_variant(engine, label, n=200):
    sample = random.sample(list(engine.cache.items()), n)
    results = {}
    
    for pert_name, pert_fn in PERTURBATIONS:
        hits = 0
        latencies = []
        for true_pid, title in sample:
            query = pert_fn(title)
            t0 = time.perf_counter()
            res = engine.search(query, top_k=1)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1_000_000)
            result_pids = [r[1][0] for r in res if len(r) > 1 and isinstance(r[1], tuple)]
            if true_pid in result_pids:
                hits += 1
        
        latencies.sort()
        results[pert_name] = {
            'r1': hits / n * 100,
            'p50': latencies[n // 2],
            'p99': latencies[int(n * 0.99)],
        }
    return results

print("=" * 70)
print("COMPARISON: Original Mesh vs TF-IDF Mesh vs Pinhole V2")
print("=" * 70)

print("\n[1/3] Benchmarking ORIGINAL keyword mesh...")
engine_orig = YPEngine()
orig_results = benchmark_variant(engine_orig, "Original")

print("\n[2/3] Benchmarking TF-IDF keyword mesh...")
with open('data/tfidf_keyword_mesh.json', 'r') as f:
    tfidf_mesh = json.load(f)

engine_tfidf = YPEngine()
engine_tfidf._keyword_mesh = tfidf_mesh
tfidf_results = benchmark_variant(engine_tfidf, "TF-IDF")

print("\n[3/3] Benchmarking STRICT PINHOLE V2...")
with open('data/pinhole_v2_strict.json', 'r') as f:
    pinhole_v2 = json.load(f)

engine_pin = YPEngine()
original_search = engine_pin.search

def pinhole_search(query_text, **kwargs):
    query_lower = query_text.lower().strip()
    query_words = set(re.findall(r"[a-zA-Z]+", query_lower))
    
    query_mask = 0
    for w in query_words:
        if w in pinhole_v2['anchor_words']:
            query_mask |= (1 << pinhole_v2['anchor_words'].index(w))
    
    if query_mask:
        paper_scores = {}
        for i in range(64):
            if query_mask & (1 << i):
                for pid in pinhole_v2['bit_to_papers'].get(str(i), []):
                    paper_scores[pid] = paper_scores.get(pid, 0) + 1
        
        if paper_scores:
            sorted_scores = sorted(paper_scores.items(), key=lambda x: -x[1])
            best_pid, best_score = sorted_scores[0]
            
            query_bits = bin(query_mask).count('1')
            if best_score >= query_bits * 0.875:
                if len(sorted_scores) == 1 or best_score >= sorted_scores[1][1] * 2:
                    h = pinhole_v2['paper_pinholes'][best_pid]['hash']
                    if h:
                        return engine_pin.search_sharded_hash(h, top_k=kwargs.get('top_k', 5))
    
    return original_search(query_text, **kwargs)

engine_pin.search = pinhole_search
pin_results = benchmark_variant(engine_pin, "PinholeV2")

print("\n" + "=" * 70)
print("RESULTS TABLE")
print("=" * 70)
print(f"\n  {'Perturbation':<25} {'Orig R@1':>10} {'TFIDF R@1':>10} {'PinV2 R@1':>10} | {'Orig P50':>10} {'TFIDF P50':>10} {'PinV2 P50':>10}")
print("  " + "-" * 90)

for pert in PERTURBATIONS:
    name = pert[0]
    o = orig_results[name]
    t = tfidf_results[name]
    p = pin_results[name]
    print(f"  {name:<25} {o['r1']:>9.1f}% {t['r1']:>9.1f}% {p['r1']:>9.1f}% | {o['p50']:>9.0f}µs {t['p50']:>9.0f}µs {p['p50']:>9.0f}µs")

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

best_overall = max([('Original', sum(orig_results[p[0]]['r1'] for p in PERTURBATIONS)),
                    ('TF-IDF', sum(tfidf_results[p[0]]['r1'] for p in PERTURBATIONS)),
                    ('PinholeV2', sum(pin_results[p[0]]['r1'] for p in PERTURBATIONS))],
                   key=lambda x: x[1])

print(f"\n  Best overall recall: {best_overall[0]}")
print(f"  Total R@1 sum: {best_overall[1]:.1f}")

out = {
    'original': orig_results,
    'tfidf': tfidf_results,
    'pinhole_v2': pin_results,
    'winner': best_overall[0],
}
with open('logs/comparison_tfidf_vs_mesh.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: logs/comparison_tfidf_vs_mesh.json")
