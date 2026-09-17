#!/usr/bin/env python3
"""Generate diverse query log entries for autopoiesis to analyze."""
import sys, os, random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yp_bridge import YPEngine

random.seed(42)
engine = YPEngine()
papers = list(engine.cache.items())

# Import logger
from scripts.query_logger import init_db, log_query
init_db()

def delete_words(text, ratio=0.3):
    words = text.split()
    keep = max(1, int(len(words) * (1 - ratio)))
    return ' '.join(random.sample(words, keep))

def char_swap(text, n=2):
    chars = list(text)
    for _ in range(n):
        i = random.randint(0, len(chars) - 2)
        chars[i], chars[i+1] = chars[i+1], chars[i]
    return ''.join(chars)

print("Seeding query log with 500 diverse queries...")

for i, (pid, title) in enumerate(random.sample(papers, 500)):
    # Mix of query types
    r = random.random()
    if r < 0.3:
        query = title  # Exact
        pert = 'exact'
    elif r < 0.5:
        query = delete_words(title, 0.3)  # Short/deleted
        pert = 'deletion'
    elif r < 0.7:
        query = char_swap(title, 2)  # Typo
        pert = 'typo'
    else:
        query = title[:random.randint(20, 60)]  # Partial
        pert = 'partial'
    
    import time
    t0 = time.perf_counter()
    results = engine.search(query, top_k=5)
    t1 = time.perf_counter()
    lat = (t1 - t0) * 1_000_000
    
    result_pids = [r[1][0] for r in results if len(r) > 1 and isinstance(r[1], tuple)]
    r1 = 1 if pid in result_pids[:1] else 0
    r5 = 1 if pid in result_pids[:5] else 0
    
    log_query(query, lat, results, 5, r1_match=r1, r5_match=r5, 
              perturbation_type=pert)
    
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/500 seeded")

print("Query log seeded. Running autopoiesis tuner...")
