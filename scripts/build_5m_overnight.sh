#!/bin/bash
# Overnight orchestrator: true-5M real-papers pipeline.
# Steps: wait OAI embed -> EPMC harvest -> embed new PubMed -> build ISM -> build HNSW.
# Logs to data/logs/build_5m_overnight.log; writes STATUS lines for morning review.
set -u
cd /Users/mac/yellow_phoenix
LOG=data/logs/build_5m_overnight.log
mkdir -p data/logs
STATUS=data/build_5m_status.txt
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
st()  { echo "$*" >> "$STATUS"; }

say "=== 5M overnight build started ==="
st "STARTED $(date)"

# ── 0. wait for the 439K OAI embedding to finish ─────────────────────────
say "waiting for OAI embedding (embed_new_batch.py) to finish..."
while pgrep -f "embed_new_batch.py" > /dev/null; do sleep 30; done
REG=$(wc -l < data/new_registry.jsonl | tr -d ' ')
say "OAI embedding done: registry lines=$REG (expect 439400)"
if [ "$REG" -lt 430000 ]; then
  say "ERROR: OAI registry short ($REG). Aborting; resume manually."
  st "FAILED oai_embed registry=$REG"
  exit 1
fi

# ── 1. Europe PMC harvest (needs +1.45M new pubmed rows) ─────────────────
say "starting Europe PMC harvest..."
python3 scripts/harvest_europepmc.py --start 1996-01-01 --target 1450000 \
  --workers 4 --db data/phoenix_arxiv_1m.db >> "$LOG" 2>&1
PUBTOTAL=$(sqlite3 data/phoenix_arxiv_1m.db \
  "SELECT COUNT(*) FROM papers WHERE source='pubmed_medline'")
say "harvest done: pubmed_medline total=$PUBTOTAL"
st "HARVEST pubmed_total=$PUBTOTAL"

# ── 2. embed the new PubMed rows (rowid > 3465152) ───────────────────────
say "embedding new PubMed rows (this is the long step, ~3h)..."
.venv/bin/python scripts/embed_new_batch.py data/phoenix_arxiv_1m.db \
  "rowid > 3465152 AND source='pubmed_medline'" \
  data/embeddings_pubmed_new.npy data/new_registry_pubmed.jsonl phoenix_pubmed \
  >> "$LOG" 2>&1
say "pubmed embedding step exited rc=$?"
ls -la data/embeddings_pubmed_new.npy >> "$LOG" 2>&1
st "EMBED_PUBMED rc=$?"

# ── 3. build the merged ITQ ISM file ──────────────────────────────────────
say "building real_5m.ism (streamed ITQ encode)..."
.venv/bin/python scripts/build_5m_real.py data/real_5m.ism >> "$LOG" 2>&1
RC=$?
say "ISM build rc=$RC"
ls -la data/real_5m.ism >> "$LOG" 2>&1
st "ISM rc=$RC"
if [ "$RC" != 0 ]; then exit 1; fi

# ── 4. build the HNSW graph ───────────────────────────────────────────────
say "building real_5m_hnsw.bin (single-threaded, hours)..."
YP_GENESIS_PHRASE=YP_DEV_BUILD_2026_08_19 \
  cargo run --release --bin build_hnsw_from_flat_ism -- \
  data/real_5m.ism data/real_5m_hnsw.bin 32 200 >> "$LOG" 2>&1
RC=$?
say "HNSW build rc=$RC"
ls -la data/real_5m_hnsw.bin >> "$LOG" 2>&1
st "HNSW rc=$RC"

say "=== 5M overnight build finished ==="
if [ "$RC" = 0 ]; then
  st "DONE $(date)"
else
  st "FAILED hnsw rc=$RC $(date)"
fi
