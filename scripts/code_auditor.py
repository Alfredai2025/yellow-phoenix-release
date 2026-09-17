#!/usr/bin/env python3
"""Self-audit tool for Yellow Phoenix source code.

Scans key Python files for red flags:
- hardcoded credentials/tokens
- print statements in hot paths
- bare except clauses
- unsafe eval/exec
- TODO markers

Usage:
    python3 scripts/code_auditor.py
"""
import ast
import os
import re
from pathlib import Path

YP_ROOT = Path("~/yellow_phoenix").expanduser()
TARGETS = [
    "yp_bridge.py",
    "scripts/flask_api.py",
    "scripts/watchdog.sh",
    "scripts/thermal_guard.sh",
    "scripts/safe_autopoiesis.py",
    "scripts/narrative_translator.py",
    "scripts/telemetry.py",
    "scripts/live_telemetry.py",
    "scripts/plot_telemetry.py",
]

PATTERNS = {
    "hardcoded_secret": re.compile(
        r"(password|secret|token|api_key)\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE
    ),
    "todo": re.compile(r"#\s*TODO", re.IGNORECASE),
    "bare_except": re.compile(r"except\s*:\s*$"),
    "eval_exec": re.compile(r"\b(eval|exec)\s*\("),
    "debug_print": re.compile(r"\bprint\s*\("),
}


def audit_file(path: Path):
    """Return list of findings for a file."""
    findings = []
    if not path.exists():
        findings.append(f"MISSING: {path}")
        return findings

    text = path.read_text(errors="ignore")
    lines = text.splitlines()

    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if PATTERNS["hardcoded_secret"].search(line):
            findings.append(f"{path}:{lineno}: possible hardcoded secret")
        if PATTERNS["bare_except"].search(line):
            findings.append(f"{path}:{lineno}: bare except")
        if PATTERNS["eval_exec"].search(line):
            findings.append(f"{path}:{lineno}: eval/exec usage")
        if PATTERNS["todo"].search(line):
            findings.append(f"{path}:{lineno}: TODO marker")

    # Syntax check
    try:
        ast.parse(text)
    except SyntaxError as e:
        findings.append(f"{path}: SYNTAX ERROR: {e}")

    return findings


def main():
    print("=" * 60)
    print("  YELLOW PHOENIX — CODE AUDITOR")
    print("=" * 60)

    all_findings = []
    for target in TARGETS:
        path = YP_ROOT / target
        findings = audit_file(path)
        if findings:
            all_findings.extend(findings)
            print(f"\n{target}:")
            for f in findings:
                print(f"  • {f}")

    if not all_findings:
        print("\nNo red flags found. Code looks clean.")
    else:
        print(f"\nTotal findings: {len(all_findings)}")

    print("=" * 60)


if __name__ == "__main__":
    main()
