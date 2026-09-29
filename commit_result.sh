#!/bin/bash
# commit_result.sh "<title>" [result-file...]
# JOSS living-history helper: copies optional result files into results/,
# adds a CHANGELOG entry, commits and pushes. Small, honest, dated.
set -euo pipefail
cd "$(dirname "$0")"
TITLE="${1:?usage: commit_result.sh \"<title>\" [result-file...]}"; shift || true
DATE=$(date +%F)
mkdir -p results
for f in "$@"; do
  [ -f "$f" ] && cp "$f" "results/${DATE}_$(basename "$f")"
done
export CR_TITLE="$TITLE"
python3 - <<'EOF'
import os, pathlib
title = os.environ["CR_TITLE"]
p = pathlib.Path("CHANGELOG.md")
t = p.read_text()
entry = f"- {title} ({__import__('datetime').date.today().isoformat()}).\n"
head, sep, rest = t.partition("### Added\n")
if sep:
    t = head + sep + entry + rest
else:
    t = t.replace("## [Unreleased]\n", "## [Unreleased]\n\n### Added\n" + entry + "\n", 1)
p.write_text(t)
EOF
git add -A
git commit -m "$TITLE"
git push
echo "committed + pushed: $TITLE"
