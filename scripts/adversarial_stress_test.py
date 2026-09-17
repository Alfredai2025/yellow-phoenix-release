#!/usr/bin/env python3
"""
Adversarial semantic stress test.
Corrupts paper titles and measures recall degradation.
"""
import sys, os, random, time, statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yp_bridge import YPEngine

random.seed(42)

SYNONYMS = {
    'learning': 'training', 'model': 'framework', 'network': 'graph',
    'deep': 'profound', 'optimization': 'improvement', 'algorithm': 'method',
    'analysis': 'study', 'system': 'platform', 'data': 'information',
    'neural': 'cognitive', 'machine': 'automated', 'intelligence': 'cognition',
    'prediction': 'forecasting', 'classification': 'categorization',
    'training': 'learning', 'performance': 'efficiency', 'accuracy': 'precision',
    'feature': 'attribute', 'vector': 'array', 'embedding': 'encoding',
    'clustering': 'grouping', 'regression': 'fitting', 'robust': 'resilient',
    'efficient': 'effective', 'novel': 'new', 'approach': 'strategy',
    'framework': 'model', 'method': 'technique', 'evaluation': 'assessment',
}

def delete_words(text, ratio=0.3):
    words = text.split()
    keep = max(1, int(len(words) * (1 - ratio)))
    return ' '.join(random.sample(words, keep))

def synonym_swap(text, ratio=0.3):
    words = text.split()
    n = max(1, int(len(words) * ratio))
    indices = random.sample(range(len(words)), min(n, len(words)))
    for i in indices:
        w = words[i].lower().strip('.,;:!?')
        if w in SYNONYMS:
            words[i] = SYNONYMS[w]
    return ' '.join(words)

def char_swap(text, n_swaps=3):
    chars = list(text)
    for _ in range(n_swaps):
        i = random.randint(0, len(chars) - 2)
        chars[i], chars[i+1] = chars[i+1], chars[i]
    return ''.join(chars)

def insert_noise(text, ratio=0.2):
    words = text.split()
    n = max(1, int(len(words) * ratio))
    noise = ['X', 'NOISE', 'FOO', 'ZZZ', '123']
    for _ in range(n):
        words.insert(random.randint(0, len(words)), random.choice(noise))
    return ' '.join(words)

def case_shuffle(text):
    return ''.join(c.upper() if random.random() > 0.5 else c.lower() for c in text)

PERTURBATIONS = [
    ('Original', lambda t: t),
    ('Word deletion 30%', delete_words),
    ('Synonym swap 30%', synonym_swap),
    ('Char swap (3)', char_swap),
    ('Noise insertion 20%', insert_noise),
    ('Case shuffle', case_shuffle),
    ('Combined (del+syn+noise)', lambda t: insert_noise(synonym_swap(delete_words(t, 0.2), 0.2), 0.15)),
]

print("=" * 65)
print("ADVERSARIAL STRESS TEST")
print("=" * 65)

print("\n[1/2] Loading engine...")
engine = YPEngine()
papers = list(engine.cache.items())
print(f"      Papers: {len(papers)}")

SAMPLE = 500
sample_papers = random.sample(papers, SAMPLE)
print(f"      Sample size: {SAMPLE}")

print("\n[2/2] Running perturbations...")

results = []
for label, perturb_fn in PERTURBATIONS:
    r1_hits = 0
    r5_hits = 0
    r10_hits = 0
    latencies = []
    
    for true_pid, title in sample_papers:
        corrupted = perturb_fn(title)
        
        t0 = time.perf_counter()
        try:
            res = engine.search(corrupted, top_k=10)
        except Exception as e:
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
        if true_pid in result_pids[:10]:
            r10_hits += 1
    
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    
    results.append({
        'perturbation': label,
        'r1': r1_hits / SAMPLE * 100,
        'r5': r5_hits / SAMPLE * 100,
        'r10': r10_hits / SAMPLE * 100,
        'p50_us': p50,
    })
    
    print(f"\n  {label}:")
    print(f"    R@1:  {r1_hits/SAMPLE*100:.1f}%")
    print(f"    R@5:  {r5_hits/SAMPLE*100:.1f}%")
    print(f"    R@10: {r10_hits/SAMPLE*100:.1f}%")
    print(f"    P50:  {p50:.1f} µs")

# Summary table
print("\n" + "=" * 65)
print("SUMMARY TABLE")
print("=" * 65)
print(f"\n  {'Perturbation':<30} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'P50 µs':>10}")
print("  " + "-" * 68)
for r in results:
    print(f"  {r['perturbation']:<30} {r['r1']:>7.1f}% {r['r5']:>7.1f}% {r['r10']:>7.1f}% {r['p50_us']:>9.1f}")

import json
out = {
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'sample_size': SAMPLE,
    'results': results,
}
os.makedirs('logs', exist_ok=True)
with open('logs/adversarial_stress_test.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: logs/adversarial_stress_test.json")
