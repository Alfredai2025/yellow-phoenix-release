# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Yellow Phoenix Deep Audit & Gap Scanner — Phase 6 Complete"""
import os, re, json, ast
from pathlib import Path
from collections import defaultdict

BASE = Path.home() / "yellow_phoenix"
report = {
    "scan_timestamp": "2026-08-12T00:10:00+08:00",
    "branch": "phoenix-bench-windows-v2",
    "gaps": [],
    "markers": defaultdict(list),
    "empty_files": [],
    "stub_modules": [],
    "unwired_in_agent": [],
    "ffi_missing": [],
    "rust_unimplemented": [],
    "python_todo_count": 0,
    "rust_todo_count": 0,
}

MARKERS = re.compile(r'(TODO|FIXME|HACK|XXX|STUB|WIRE_ME|UNIMPLEMENTED|PLACEHOLDER|INCOMPLETE|BROKEN|DEPRECATED|NOT? (?:YET|IMPLEMENTED)|TEMP(?:ORARY)?|SHIM|BYPASS|WORKAROUND|PHASE\s*6)', re.IGNORECASE)

# ── 1. Scan all source for markers ──
for ext in ("*.py", "*.rs"):
    for f in sorted(BASE.rglob(ext)):
        if any(x in str(f) for x in ("__pycache__", "target/", ".git/", ".bak", "snapshot")):
            continue
        try:
            text = f.read_text(errors="ignore")
            lines = text.splitlines()
        except:
            continue
        for i, line in enumerate(lines, 1):
            m = MARKERS.search(line)
            if m:
                report["markers"][str(f.relative_to(BASE))].append({
                    "line": i,
                    "marker": m.group(1),
                    "code": line.strip()[:140]
                })

# ── 2. Find empty / stub Python files ──
for f in sorted(BASE.rglob("*.py")):
    if "__pycache__" in str(f):
        continue
    text = f.read_text(errors="ignore").strip()
    if not text:
        report["empty_files"].append(str(f.relative_to(BASE)))
    elif len(text) < 100 and ("pass" in text or "..." in text):
        report["stub_modules"].append(str(f.relative_to(BASE)))

# ── 3. Find Rust unimplemented! / todo! ──
for f in sorted(BASE.rglob("*.rs")):
    if "target/" in str(f):
        continue
    text = f.read_text(errors="ignore")
    if "unimplemented!" in text or "todo!" in text:
        report["rust_unimplemented"].append(str(f.relative_to(BASE)))

# ── 4. Check agent.py wiring ──
agent_text = (BASE / "yp_autonomic" / "agent.py").read_text(errors="ignore")
for mod_dir in ("actuators", "sensors", "cortex", "immune", "replication"):
    d = BASE / "yp_autonomic" / mod_dir
    if not d.exists():
        continue
    for f in sorted(d.glob("*.py")):
        if f.name.startswith("_") or f.name == "auto_rewire.py.bak.safety":
            continue
        mod_name = f.stem
        if mod_name not in agent_text:
            report["unwired_in_agent"].append(f"{mod_dir}/{f.name}")

# ── 5. FFI gaps from runtime warnings ──
bridge_text = (BASE / "yp_bridge.py").read_text(errors="ignore")
for m in re.finditer(r'Missing FFI function:.*?(yp_[a-zA-Z0-9_]+)', bridge_text):
    if m.group(1) not in report["ffi_missing"]:
        report["ffi_missing"].append(m.group(1))

# ── 6. Count totals ──
report["python_todo_count"] = sum(len(v) for v in report["markers"].values())
report["rust_todo_count"] = len(report["rust_unimplemented"])

# ── 7. Classify gaps by severity ──
for fname, items in report["markers"].items():
    for item in items:
        severity = "LOW"
        if any(x in item["code"].lower() for x in ["broken", "crash", "fail", "oom", "leak"]):
            severity = "CRITICAL"
        elif any(x in item["code"].lower() for x in ["todo", "fixme", "unimplemented"]):
            severity = "MEDIUM"
        elif "phase 6" in item["code"].lower():
            severity = "MEDIUM"
        
        report["gaps"].append({
            "file": fname,
            "line": item["line"],
            "severity": severity,
            "marker": item["marker"],
            "description": item["code"]
        })

# ── Write reports ──
json_path = BASE / "reports" / "deep_audit_20260812.json"
json_path.write_text(json.dumps(report, indent=2, default=str))

# Human-readable summary
summary = f"""YELLOW PHOENIX DEEP AUDIT REPORT
=================================
Generated: {report['scan_timestamp']}
Branch: {report['branch']}

SUMMARY
-------
Total TODO/FIXME markers: {report['python_todo_count']}
Rust unimplemented! blocks: {report['rust_todo_count']}
Empty/stub Python files: {len(report['empty_files'])}
Unwired modules (not in agent.py): {len(report['unwired_in_agent'])}
Missing FFI symbols: {len(report['ffi_missing'])}

CRITICAL GAPS
-------------
"""
criticals = [g for g in report["gaps"] if g["severity"] == "CRITICAL"]
for g in criticals[:20]:
    summary += f"  [{g['severity']}] {g['file']}:{g['line']} — {g['description']}\n"

summary += f"""\nMEDIUM GAPS (first 30)
----------------------\n"""
mediums = [g for g in report["gaps"] if g["severity"] == "MEDIUM"]
for g in mediums[:30]:
    summary += f"  [{g['severity']}] {g['file']}:{g['line']} — {g['description']}\n"

summary += f"""\nUNWIRED MODULES
---------------\n"""
for m in report["unwired_in_agent"]:
    summary += f"  {m}\n"

summary += f"""\nMISSING FFI SYMBOLS
-------------------\n"""
for sym in report["ffi_missing"]:
    summary += f"  {sym}\n"""

summary_path = BASE / "reports" / "deep_audit_20260812.txt"
summary_path.write_text(summary)

print(f"JSON:  {json_path}")
print(f"TEXT:  {summary_path}")
print(f"\n=== SUMMARY ===")
print(f"Markers: {report['python_todo_count']} | Rust stubs: {report['rust_todo_count']} | Unwired: {len(report['unwired_in_agent'])} | FFI gaps: {len(report['ffi_missing'])}")
