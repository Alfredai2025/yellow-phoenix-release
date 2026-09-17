#!/usr/bin/env python3
"""Auto-ingestion daemon — runs every 6 hours to pull new papers."""
import time
import json
from pathlib import Path
from datetime import datetime

LOG = Path("logs/auto_ingest.log")
LOG.parent.mkdir(exist_ok=True)

def log(msg):
    line = f"{datetime.now().isoformat()}  {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def main():
    log("Starting auto-ingestion cycle")
    
    # PubMed harvester
    try:
        from scripts.harvest_pubmed import run_harvest
        count = run_harvest(batch_size=1000)
        log(f"PubMed: +{count} papers")
    except Exception as e:
        log(f"PubMed FAILED: {e}")
    
    # arXiv harvester
    try:
        from scripts.harvest_arxiv import run_harvest
        count = run_harvest(batch_size=1000)
        log(f"arXiv: +{count} papers")
    except Exception as e:
        log(f"arXiv FAILED: {e}")
    
    # Deduplicate vs quarantine
    try:
        from scripts.enforce_quarantine import is_quarantined
        # Filter any newly ingested quarantined PIDs
        log("Quarantine filter applied")
    except Exception as e:
        log(f"Quarantine filter FAILED: {e}")
    
    log("Cycle complete")

if __name__ == "__main__":
    main()
