import sys, time, random
sys.path.insert(0, "/Users/mac/yellow_phoenix")
from yp_bridge import YPEngine

e = YPEngine()

# Access internal paper storage
titles = []
if hasattr(e, '_title_hash_map'):
    titles = list(e._title_hash_map.keys())
elif hasattr(e, 'unison') and hasattr(e.unison, 'papers'):
    titles = [p.get('title', '') for p in e.unison.papers.values()]
else:
    # Fallback: load from the JSONL that the engine uses
    import json
    with open("/Users/mac/yellow_phoenix/data/paper_metadata.jsonl") as f:
        for line in f:
            d = json.loads(line)
            titles.append(d.get("title", ""))

if not titles:
    print("ERROR: No titles found.")
    sys.exit(1)

print(f"Burning in predictor with {len(titles)} real papers...")

for i in range(10000):
    t = random.choice(titles)
    e.query_auto(t, top_k=1)
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/10000...")

print("\nDone. Predictor state updated.")
