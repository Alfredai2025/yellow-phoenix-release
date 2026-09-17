#!/bin/bash
# Round 2: finish true-5M unique — embed round-2 PubMed, dedup build, HNSW.
set -u
cd /Users/mac/yellow_phoenix
LOG=data/logs/build_5m_round2.log
mkdir -p data/logs
STATUS=data/build_5m_status.txt
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
st()  { echo "$*" >> "$STATUS"; }

say "=== round 2 started ==="
st "ROUND2_STARTED $(date)"

# ── 1. wait for harvest round 2 ───────────────────────────────────────────
while pgrep -f "harvest_europepmc.py" > /dev/null; do sleep 20; done
PUB=$(sqlite3 data/phoenix_arxiv_1m.db \
  "SELECT COUNT(*) FROM papers WHERE source='pubmed_medline'")
MAXR=$(sqlite3 data/phoenix_arxiv_1m.db "SELECT MAX(rowid) FROM papers")
say "harvest round 2 done: pubmed=$PUB maxrowid=$MAXR"
st "HARVEST2 pubmed=$PUB maxrowid=$MAXR"

# ── 2. embed round-2 pubmed rows ──────────────────────────────────────────
say "embedding round-2 pubmed rows..."
.venv/bin/python scripts/embed_new_batch.py data/phoenix_arxiv_1m.db \
  "rowid > 4960130 AND source='pubmed_medline'" \
  data/embeddings_pubmed_new2.npy data/new_registry_pubmed2.jsonl phoenix_pubmed2 \
  >> "$LOG" 2>&1
RC=$?
say "embed round2 rc=$RC"
st "EMBED2 rc=$RC"
if [ "$RC" != 0 ]; then exit 1; fi

# ── 3. dedup build (overwrites data/real_5m.ism) ──────────────────────────
say "building deduped real_5m.ism..."
.venv/bin/python scripts/build_5m_real_dedup.py data/real_5m.ism >> "$LOG" 2>&1
RC=$?
say "dedup ISM rc=$RC"
st "ISM_DEDUP rc=$RC"
if [ "$RC" != 0 ]; then exit 1; fi

# ── 4. HNSW (overwrites data/real_5m_hnsw.bin) ────────────────────────────
say "building deduped real_5m_hnsw.bin..."
YP_GENESIS_PHRASE=YP_DEV_BUILD_2026_08_19 \
  cargo run --release --bin build_hnsw_from_flat_ism -- \
  data/real_5m.ism data/real_5m_hnsw.bin 32 200 >> "$LOG" 2>&1
RC=$?
say "HNSW rc=$RC"

# ── 5. verify alignment ───────────────────────────────────────────────────
if [ "$RC" = 0 ]; then
  YP_GENESIS_PHRASE=YP_DEV_BUILD_2026_08_19 \
    cargo run --release --bin verify_hnsw_alignment -- \
    data/real_5m.ism data/real_5m_hnsw.bin 2>&1 | tail -6 >> "$LOG"
  say "verify done"
fi

ls -la data/real_5m.ism data/real_5m_hnsw.bin >> "$LOG" 2>&1
say "=== round 2 finished ==="
st "ROUND2_DONE rc=$RC $(date)"
