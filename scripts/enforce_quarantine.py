"""Quarantine enforcement gate.
Import in any training/ingest script. Call assert_not_quarantined(pid) before adding to training data.
"""
import json
from pathlib import Path

_QUARANTINE = None
_QPATH = Path("data/quarantine_pids.jsonl")

def _load():
    global _QUARANTINE
    if _QUARANTINE is not None:
        return _QUARANTINE
    _QUARANTINE = set()
    if _QPATH.exists():
        with open(_QPATH) as f:
            for line in f:
                try:
                    _QUARANTINE.add(json.loads(line)["pid"])
                except:
                    pass
    return _QUARANTINE

def is_quarantined(pid: str) -> bool:
    return str(pid) in _load()

def assert_not_quarantined(pid: str):
    if is_quarantined(pid):
        raise RuntimeError(f"QUARANTINE VIOLATION: {pid} is in the held-out exam set. "
                           f"Never train encoders or re-embed quarantined papers.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print("QUARANTINED" if is_quarantined(sys.argv[1]) else "SAFE")
    else:
        print(f"Loaded {len(_load()):,} quarantined PIDs")
