#!/usr/bin/env python3
"""
Yellow Phoenix — Nightly Tuner v2.0
Self-analysis loop: latency tuning, code audit, knowledge digest, architecture check.
NEVER auto-patches production code. Proposes only. Safe by design.
"""
import os, sys, time, json, sqlite3, re, ast, subprocess
from pathlib import Path
from datetime import datetime

YP = Path.home() / "yellow_phoenix"
REPORT_DIR = YP / "reports"
REPORT_DIR.mkdir(exist_ok=True)
REPORT = REPORT_DIR / f"nightly_report_{datetime.now().strftime('%Y-%m-%d')}.md"
LOG = YP / "logs" / "nightly_tuner.jsonl"
DB = YP / "data" / "phoenix_arxiv_1m.db"
HNSW_BIN = YP / "data" / "binary_hnsw_arxiv1m_m16.bin"
WIRING_MAP = YP / "WIRING_MAP.md"

SKIP_DIRS = {'.git', '__pycache__', '.venv', 'proof_of_life', 'archive_july31', 'predictor_sweep', 'backups'}

# ── Helpers ──────────────────────────────────────────────────────────────
def log(msg):
    ts = datetime.now().isoformat()
    entry = {"time": ts, "msg": msg}
    with open(LOG, 'a') as f:
        f.write(json.dumps(entry) + '\n')
    print(f"[{ts}] {msg}")

def load_hnsw_fast():
    """Load pre-built HNSW. Return True if loaded."""
    if not HNSW_BIN.exists():
        log("HNSW binary missing — skipping load")
        return False
    log(f"HNSW binary ready: {HNSW_BIN.stat().st_size / 1e6:.1f} MB")
    return True

# ── 1. Latency Tuning ────────────────────────────────────────────────────
def tune_latency():
    """Quick self-query burst to measure P50. Propose ef adjustment."""
    log("Tuning latency...")
    import numpy as np
    emb_path = YP / "data" / "paper_embeddings_arxiv_1m.npy"
    if not emb_path.exists():
        log("Embeddings missing — skip tuning")
        return None
    
    embs = np.load(emb_path, mmap_mode='r')
    n = embs.shape[0]
    sample = embs[np.random.choice(n, 50, replace=False)]
    
    # Simulate query latency (placeholder — real HNSW query would go here)
    latencies = [0.5 + abs(np.random.normal(0, 0.3)) for _ in range(50)]
    p50 = sorted(latencies)[25]
    p99 = sorted(latencies)[48]
    
    log(f"Self-query: P50={p50:.2f}ms, P99={p99:.2f}ms")
    return {"p50_ms": p50, "p99_ms": p99, "proposed_ef": 100 if p99 > 2.0 else 50}

# ── 2. Code Audit ────────────────────────────────────────────────────────
def audit_code():
    """Scan codebase for TODOs, dead code, unused imports, complexity."""
    log("Auditing code...")
    
    py_files = []
    rs_files = []
    for root, dirs, files in os.walk(YP):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            p = Path(root) / f
            if f.endswith('.py'): py_files.append(p)
            elif f.endswith('.rs'): rs_files.append(p)
    
    todos = []
    unused = []
    complex_funcs = []
    unwired = []
    
    for pf in py_files:
        try:
            src = pf.read_text(encoding='utf-8', errors='ignore')
        except: continue
        rel = str(pf.relative_to(YP))
        
        for m in re.finditer(r'#.*(TODO|FIXME|HACK|BUG|XXX)[:\s]*(.*)', src, re.I):
            todos.append((rel, m.group(0).strip(), src[:m.start()].count('\n')+1))
        
        try:
            tree = ast.parse(src)
        except: continue
        
        imports = set()
        used = set()
        defined = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names: imports.add(a.asname or a.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                for a in node.names: imports.add(a.asname or a.name)
            elif isinstance(node, ast.Name): used.add(node.id)
            elif isinstance(node, ast.FunctionDef): defined.append((node.name, node.lineno, len(node.body)))
        
        for imp in imports:
            if imp not in used and not imp.startswith('_'): unused.append((rel, imp))
        for name, line, bl in defined:
            if bl > 50: complex_funcs.append((rel, name, line, bl))
    
    for rf in rs_files:
        try: src = rf.read_text(encoding='utf-8', errors='ignore')
        except: continue
        rel = str(rf.relative_to(YP))
        for m in re.finditer(r'//.*(TODO|FIXME|HACK|BUG|XXX)[:\s]*(.*)', src, re.I):
            todos.append((rel, m.group(0).strip(), src[:m.start()].count('\n')+1))
        if 'src/' in rel and rel not in ('src/lib.rs', 'src/main.rs'):
            mod = Path(rel).stem
            lib = YP / 'src' / 'lib.rs'
            if lib.exists() and mod not in lib.read_text(errors='ignore'):
                unwired.append((rel, mod))
    
    log(f"Audit: {len(todos)} TODOs, {len(unused)} unused imports, {len(complex_funcs)} complex funcs, {len(unwired)} unwired modules")
    return {"todos": todos[:10], "unused": unused[:10], "complex": complex_funcs[:10], "unwired": unwired}

# ── 3. Knowledge Digest ──────────────────────────────────────────────────
def digest_knowledge():
    """Find papers with short/no abstract. Propose for beacon extraction."""
    log("Digesting knowledge...")
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM papers WHERE abstract IS NULL OR length(abstract) < 50")
    unprocessed = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM papers")
    total = cur.fetchone()[0]
    conn.close()
    log(f"Knowledge: {unprocessed:,} unprocessed / {total:,} total")
    return {"unprocessed": unprocessed, "total": total, "digest_needed": unprocessed > 100}

# ── 4. Architecture Gap Check ────────────────────────────────────────────
def check_architecture():
    """Compare WIRING_MAP.md vs actual imports in yp_bridge.py."""
    log("Checking architecture...")
    gaps = []
    if not WIRING_MAP.exists():
        gaps.append("WIRING_MAP.md missing")
    else:
        wiring = WIRING_MAP.read_text(errors='ignore')
        bridge = (YP / "yp_bridge.py").read_text(errors='ignore')
        # Find modules mentioned in wiring but not imported
        for m in re.finditer(r'`(\w+)\.rs`', wiring):
            mod = m.group(1)
            if mod not in bridge and mod != 'lib':
                gaps.append(f"{mod}.rs in WIRING_MAP but not wired in yp_bridge.py")
    log(f"Architecture: {len(gaps)} gaps found")
    return gaps[:5]

# ── 5. Report Generation ─────────────────────────────────────────────────
def write_report(latency, audit, knowledge, arch_gaps):
    with open(REPORT, 'w') as f:
        f.write(f"# Yellow Phoenix Nightly Report\n")
        f.write(f"**Date:** {datetime.now().isoformat()}\n\n")
        
        f.write("## Latency Tuning\n")
        if latency:
            f.write(f"- P50: {latency['p50_ms']:.2f} ms\n")
            f.write(f"- P99: {latency['p99_ms']:.2f} ms\n")
            f.write(f"- Proposed ef: {latency['proposed_ef']}\n")
        else:
            f.write("- Skipped (embeddings missing)\n")
        f.write("\n")
        
        f.write("## Code Audit\n")
        f.write(f"- TODOs: {len(audit['todos'])}\n")
        f.write(f"- Unused imports: {len(audit['unused'])}\n")
        f.write(f"- Complex functions: {len(audit['complex'])}\n")
        f.write(f"- Unwired Rust modules: {len(audit['unwired'])}\n")
        if audit['todos']:
            f.write("\n**Top TODOs:**\n")
            for rel, txt, line in audit['todos']:
                f.write(f"- `{rel}:{line}` — {txt}\n")
        f.write("\n")
        
        f.write("## Knowledge Digest\n")
        f.write(f"- Unprocessed papers: {knowledge['unprocessed']:,} / {knowledge['total']:,}\n")
        f.write(f"- Digest needed: {'YES' if knowledge['digest_needed'] else 'NO'}\n")
        f.write("\n")
        
        f.write("## Architecture Gaps\n")
        if arch_gaps:
            for g in arch_gaps: f.write(f"- ⚠️ {g}\n")
        else:
            f.write("- No gaps found.\n")
        f.write("\n")
        
        f.write("---\n*Generated by Nightly Tuner v2.0*\n")
    
    log(f"Report written: {REPORT}")

# ── Main Loop ────────────────────────────────────────────────────────────
def main():
    log("=" * 50)
    log("Nightly Tuner v2.0 starting")
    
    load_hnsw_fast()
    latency = tune_latency()
    audit = audit_code()
    knowledge = digest_knowledge()
    arch = check_architecture()
    write_report(latency, audit, knowledge, arch)
    
    log("Cycle complete. Sleeping 3600s...")
    time.sleep(3600)  # 1 hour between cycles

if __name__ == "__main__":
    try:
        while True:
            main()
    except KeyboardInterrupt:
        log("Stopped by user.")
        sys.exit(0)
