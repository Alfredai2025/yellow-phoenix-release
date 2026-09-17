#!/usr/bin/env python3
"""
Query expansion: ONLY triggers when query is short OR not an exact title match.
"""
import json, os, re

_EXPANSIONS = None

def _load():
    global _EXPANSIONS
    if _EXPANSIONS is not None:
        return
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'query_expansions.json')
    with open(path, 'r') as f:
        _EXPANSIONS = json.load(f)

def expand_query(text, skip_if_in=None, max_added=3):
    """
    Expand query ONLY if:
    - It's short (< 4 content words), OR
    - It's NOT in the skip_if_in set (title hash map)
    """
    _load()
    query_lower = text.lower().strip()
    
    # If exact title match exists, NEVER expand (preserves fast path)
    if skip_if_in and query_lower in skip_if_in:
        return text
    # Also check 60-char prefix
    if skip_if_in and query_lower[:60] in skip_if_in:
        return text
    
    words = re.findall(r"[a-zA-Z]+", query_lower)
    words = [w for w in words if len(w) > 2]
    
    # Only expand short queries
    if len(words) >= 5:
        return text
    
    added = set()
    for w in words:
        for related in _EXPANSIONS.get(w, []):
            if related not in words and related not in added:
                added.add(related)
                if len(added) >= max_added:
                    break
        if len(added) >= max_added:
            break
    
    if added:
        return text + ' ' + ' '.join(sorted(added))
    return text

if __name__ == '__main__':
    import sys
    for line in sys.stdin:
        print(expand_query(line.strip()))
