#!/usr/bin/env python3
"""One-time unpause: resume PubMed + arXiv harvesters."""
import sys
from pathlib import Path

def unpause():
    print("[INGEST] Resuming paused harvesters...")
    # Touch marker files to signal harvesters to resume
    Path("data/ingest_resume.flag").touch()
    print("[INGEST] Flag file created. Harvesters will resume on next run.")
    print("[INGEST] To start immediately, run: python3 scripts/auto_ingest.py")

if __name__ == "__main__":
    unpause()
