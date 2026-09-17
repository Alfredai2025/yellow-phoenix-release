#!/usr/bin/env python3
"""
Lightweight spelling correction using title vocabulary.
"""
import json, os, re

_VOCAB = None
_FREQ = None

def _load():
    global _VOCAB, _FREQ
    if _VOCAB is not None:
        return
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'spell_vocab.json')
    with open(path, 'r') as f:
        data = json.load(f)
    _VOCAB = set(data['vocab'])
    _FREQ = data['freq']

def _edits1(word):
    """All edits that are one edit away from word."""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def correct_word(word):
    """Return the most likely correction for word."""
    _load()
    word = word.lower()
    if word in _VOCAB:
        return word
    # Generate candidates
    candidates = _edits1(word) & _VOCAB
    if not candidates:
        # Try edits2 for really bad typos (slower, limit scope)
        for e1 in _edits1(word):
            candidates |= _edits1(e1) & _VOCAB
            if len(candidates) > 20:
                break
    if not candidates:
        return word  # No correction found
    # Pick highest frequency
    return max(candidates, key=lambda w: _FREQ.get(w, 0))

def correct_text(text):
    """Correct all words in text."""
    _load()
    words = re.findall(r"[a-zA-Z]+", text)
    corrected = []
    for w in words:
        cw = correct_word(w)
        corrected.append(cw)
    # Reconstruct preserving non-alpha
    result = []
    i = 0
    for token in re.split(r"([a-zA-Z]+)", text):
        if token.isalpha():
            result.append(corrected[i])
            i += 1
        else:
            result.append(token)
    return ''.join(result)

if __name__ == '__main__':
    import sys
    for line in sys.stdin:
        print(correct_text(line.strip()))
