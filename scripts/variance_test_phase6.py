# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
Phase 6: Variance Test — Do CS papers cluster tighter under Fibonacci mask vs Lucas?

Run this after you have:
  • data/itq_hashes_1m.npy  (N x 64 uint8)  OR  data/itq_hashes_1m.bin
  • arXiv category metadata (DB or JSONL with pid → categories)

If CS intra-category variance under Fibonacci < Lucas, the channel theory is proven.
If not, use simple category-based sharding instead of sequence-mask routing.
"""

import numpy as np
import math
import json
import sqlite3
from pathlib import Path
import time

# ───────────────────────────────────────────────
# 1. MASK GENERATORS (self-contained)
# ───────────────────────────────────────────────
PHI = (1 + math.sqrt(5)) / 2

def gen_mask(ratio, n_bits=512, keep=256, phase=0.0):
    seen = set()
    mask = []
    k = 0
    while len(mask) < keep:
        idx = int(((k + phase) * n_bits) / ratio) % n_bits
        if idx not in seen:
            mask.append(idx)
            seen.add(idx)
        k += 1
    return np.array(mask, dtype=np.uint16)

print("Generating masks...")
LUCAS_MASK   = gen_mask(PHI, phase=0.0)      # Lucas
FIB_MASK     = gen_mask(PHI, phase=1.05)     # Fibonacci (phase-shifted)
PELL_MASK    = gen_mask(1 + math.sqrt(2), phase=0.0)
PADOVAN_MASK = gen_mask(1.324717957244746, phase=0.0)

print(f"Lucas:   {LUCAS_MASK[:5]}... overlap with Fib = {len(set(LUCAS_MASK) & set(FIB_MASK))}/256")

# ───────────────────────────────────────────────
# 2. LOAD HASHES
# ───────────────────────────────────────────────
HASH_PATH = Path("data/itq_hashes_1m.npy")
BIN_PATH  = Path("data/itq_hashes_1m.bin")

hashes = None
if HASH_PATH.exists():
    hashes = np.load(HASH_PATH)
    if hashes.ndim == 1:
        hashes = hashes.reshape(-1, 64)
    print(f"Loaded hashes: {hashes.shape} from {HASH_PATH}")
elif BIN_PATH.exists():
    raw = np.fromfile(BIN_PATH, dtype=np.uint8)
    n = len(raw) // 64
    hashes = raw[:n*64].reshape(n, 64)
    print(f"Loaded hashes: {hashes.shape} from {BIN_PATH}")
else:
    print("ERROR: No hash file found.")
    print("  Expected: data/itq_hashes_1m.npy  or  data/itq_hashes_1m.bin")
    print("  Create by exporting your 1M ITQ hashes as N x 64 uint8 array.")
    exit(1)

# ───────────────────────────────────────────────
# 3. LOAD CATEGORIES
# ───────────────────────────────────────────────
META_PATH = Path("data/arxiv_1m_metadata.jsonl")
DB_PATH   = Path("data/phoenix_arxiv_1m.db")

cat_map = {}  # pid_idx (int) → list of category strings

if DB_PATH.exists():
    print(f"Loading categories from {DB_PATH} ...")
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    # Try common schemas
    try:
        cur.execute("SELECT rowid, categories FROM papers LIMIT 5")
    except sqlite3.OperationalError:
        try:
            cur.execute("SELECT id, categories FROM papers LIMIT 5")
        except sqlite3.OperationalError:
            try:
                cur.execute("SELECT pid, categories FROM papers LIMIT 5")
            except sqlite3.OperationalError:
                print("WARNING: Could not auto-detect category column. Tabling available columns...")
                cur.execute("SELECT name FROM pragma_table_info('papers')")
                cols = [r[0] for r in cur.fetchall()]
                print(f"  Available columns: {cols}")
                conn.close()
                exit(1)

    # Re-run with detected schema (simplified: assume rowid or first col)
    cur.execute("SELECT rowid, categories FROM papers")
    for row in cur.fetchall():
        idx, cats = row
        if cats:
            cat_map[idx] = [c.strip() for c in str(cats).split(',')]
    conn.close()
    print(f"Loaded {len(cat_map)} category records from DB.")

elif META_PATH.exists():
    print(f"Loading categories from {META_PATH} ...")
    with open(META_PATH) as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            cats = obj.get("categories") or obj.get("category", "")
            if cats:
                cat_map[i] = [c.strip() for c in str(cats).split(',')]
    print(f"Loaded {len(cat_map)} category records from JSONL.")

else:
    print("WARNING: No metadata found. Generating synthetic category split for demo...")
    # Fallback: split by hash prefix as fake categories
    for i in range(len(hashes)):
        prefix = hashes[i, 0] % 4
        fake_cats = {0: ["cs.AI"], 1: ["cs.LG"], 2: ["q-bio"], 3: ["physics"]}
        cat_map[i] = fake_cats[prefix]
    print("  (Synthetic categories used — results are NOT meaningful.)")

# ───────────────────────────────────────────────
# 4. EXTRACT SKELETONS
# ───────────────────────────────────────────────
def extract_skeleton(hashes_2d, mask):
    """N x 64 bytes → N x 32 bytes (256-bit skeleton)"""
    bits = np.unpackbits(hashes_2d, axis=1)  # N x 512
    skel_bits = bits[:, mask]                # N x 256
    return np.packbits(skel_bits, axis=1)    # N x 32

print("Extracting skeletons...")
lucas_skel   = extract_skeleton(hashes, LUCAS_MASK)
fib_skel     = extract_skeleton(hashes, FIB_MASK)

# ───────────────────────────────────────────────
# 5. BUILD CATEGORY INDEX
# ───────────────────────────────────────────────
cs_indices = []
med_indices = []
phys_indices = []
other_indices = []

for idx, cats in cat_map.items():
    if idx >= len(hashes):
        continue
    cat_str = ",".join(cats).lower()
    if "cs." in cat_str or "cmp-lg" in cat_str:
        cs_indices.append(idx)
    elif "q-bio" in cat_str or "stat." in cat_str or "med" in cat_str:
        med_indices.append(idx)
    elif "physics" in cat_str or "astro" in cat_str or "hep" in cat_str:
        phys_indices.append(idx)
    elif "math" in cat_str:
        other_indices.append(idx)  # math grouped with other for now
    else:
        other_indices.append(idx)

print(f"Categories found: CS={len(cs_indices)}, Med={len(med_indices)}, Phys={len(phys_indices)}, Other={len(other_indices)}")

if len(cs_indices) < 100:
    print("ERROR: Not enough CS papers for meaningful variance test (<100).")
    print("  Check your category data source.")
    exit(1)

# ───────────────────────────────────────────────
# 6. VARIANCE METRICS
# ───────────────────────────────────────────────
def fast_hamming_var(skeletons, indices, label=""):
    """Fast intra-group variance via sampling."""
    subset = skeletons[indices]
    if len(subset) > 2000:
        subset = subset[np.random.choice(len(subset), 2000, replace=False)]

    # Convert to uint64 for fast XOR
    u = subset.view(np.uint64)  # N x 4
    n = len(u)

    # Sample 10,000 random pairs
    np.random.seed(42)
    i = np.random.randint(0, n, 10000)
    j = np.random.randint(0, n, 10000)
    mask = i != j
    i, j = i[mask], j[mask]

    xor = u[i] ^ u[j]  # shape: (k, 4)
    # popcount each uint64
    pc = np.zeros(len(xor), dtype=np.uint32)
    for w in range(4):
        v = xor[:, w]
        # Use Python int popcount (slower but correct; for 10k pairs it's fine)
        pc += np.array([bin(x).count("1") for x in v], dtype=np.uint32)

    mean_d = float(pc.mean())
    var_d = float(pc.var())
    print(f"  {label}: n={n}, pairs={len(pc)}, mean_dist={mean_d:.2f}, var={var_d:.2f}")
    return var_d, mean_d

# ───────────────────────────────────────────────
# 7. RUN TESTS
# ───────────────────────────────────────────────
print("\n" + "="*60)
print("VARIANCE TEST: CS Papers under Lucas vs Fibonacci")
print("="*60)

# Intra-CS variance
print("\n[Intra-CS] Lucas mask:")
cs_var_lucas, cs_mean_lucas = fast_hamming_var(lucas_skel, cs_indices, "CS-Lucas")

print("\n[Intra-CS] Fibonacci mask:")
cs_var_fib, cs_mean_fib = fast_hamming_var(fib_skel, cs_indices, "CS-Fib")

# Inter-category variance (CS vs Other)
print("\n[Inter: CS vs Other] Lucas mask:")
inter_var_lucas, inter_mean_lucas = fast_hamming_var(lucas_skel, cs_indices[:500] + other_indices[:500], "Cross-Lucas")

print("\n[Inter: CS vs Other] Fibonacci mask:")
inter_var_fib, inter_mean_fib = fast_hamming_var(fib_skel, cs_indices[:500] + other_indices[:500], "Cross-Fib")

# ───────────────────────────────────────────────
# 8. VERDICT
# ───────────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"""
Metric                    Lucas           Fibonacci
─────────────────────────────────────────────────────
CS intra mean dist        {cs_mean_lucas:.2f}           {cs_mean_fib:.2f}
CS intra variance         {cs_var_lucas:.2f}           {cs_var_fib:.2f}
Cross mean dist           {inter_mean_lucas:.2f}           {inter_mean_fib:.2f}
Cross variance            {inter_var_lucas:.2f}           {inter_var_fib:.2f}
""")

# Ratios
intra_ratio = cs_var_fib / cs_var_lucas if cs_var_lucas > 0 else 999
separation_lucas = inter_mean_lucas / cs_mean_lucas if cs_mean_lucas > 0 else 0
separation_fib   = inter_mean_fib / cs_mean_fib if cs_mean_fib > 0 else 0

print(f"Fibonacci / Lucas intra-variance ratio: {intra_ratio:.3f}")
print(f"  (< 1.0  → Fibonacci clusters CS tighter  → BUILD sequence channels)")
print(f"  (>= 1.0 → No benefit  → USE category sharding instead)")
print()
print(f"Separation ratio (cross / intra):")
print(f"  Lucas:   {separation_lucas:.3f}  (higher = better separation)")
print(f"  Fibonacci: {separation_fib:.3f}")

print("\n" + "="*60)
if intra_ratio < 0.95 and separation_fib > separation_lucas:
    print("VERDICT: Fibonacci mask improves CS clustering.")
    print("         Sequence-channel routing is SUPPORTED.")
elif intra_ratio < 1.05:
    print("VERDICT: Marginal difference. Sequence masks are cosmetic.")
    print("         Use simple CATEGORY sharding — same effect, less complexity.")
else:
    print("VERDICT: Fibonacci does NOT improve CS clustering.")
    print("         ABANDON sequence-channel routing. Use category sharding.")
print("="*60)

# ───────────────────────────────────────────────
# 9. BONUS: All 4 masks on CS
# ───────────────────────────────────────────────
print("\n[BONUS] All 4 masks — CS intra-variance:")
for name, mask in [("Lucas", LUCAS_MASK), ("Fibonacci", FIB_MASK), ("Pell", PELL_MASK), ("Padovan", PADOVAN_MASK)]:
    skel = extract_skeleton(hashes, mask)
    v, m = fast_hamming_var(skel, cs_indices, f"CS-{name}")

print("\nDone. If variance test passes, proceed to Phase 2 (Single-Graph Multi-Edge).")
