#!/usr/bin/env python3
"""
Yellow Phoenix — State Monitor v1.0
Lightweight loop: analyzes DB, telemetry, tunes predictor, logs state.
No code modification. No encoder retraining. Read-only + safe actions only.
"""
import os, sys, time, json, sqlite3
from pathlib import Path
from datetime import datetime

YP = Path.home() / "yellow_phoenix"
LOG = YP / "logs" / "state_monitor.jsonl"
STATE = YP / "data" / "state_monitor.json"
DB = YP / "data" / "phoenix_arxiv_1m.db"

# ... [rest of safe_state_monitor.py content with "state_monitor" replaced] ...
