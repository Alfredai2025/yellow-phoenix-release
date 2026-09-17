#!/bin/bash
# Serial PubMed baseline downloader with validation + retries.
# Usage: download_pubmed_serial.sh <start> <end>
set -u
START=${1:-51}
END=${2:-100}
DIR=/Users/mac/yellow_phoenix/data/pubmed_baseline
mkdir -p "$DIR"
cd "$DIR"
ok=0; fixed=0; failed=0
for i in $(seq "$START" "$END"); do
  n=$(printf "%04d" "$i")
  f="pubmed26n${n}.xml.gz"
  want="https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/$f"
  if [ -f "$f" ] && gzip -t "$f" 2>/dev/null; then
    ok=$((ok+1)); continue
  fi
  good=0
  for try in 1 2 3 4 5; do
    curl -sS --retry 3 --retry-delay 2 -o "$f" "$want"
    if [ -f "$f" ] && gzip -t "$f" 2>/dev/null; then good=1; break; fi
    sleep $((try*3))
  done
  if [ "$good" = 1 ]; then
    fixed=$((fixed+1)); echo "[ok] $f ($(stat -f%z "$f") bytes, try $try)"
  else
    failed=$((failed+1)); echo "[FAIL] $f"
  fi
  sleep 1
done
echo "DONE: ok=$ok downloaded=$fixed failed=$failed"
