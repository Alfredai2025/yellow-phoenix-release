#!/usr/bin/env python3
"""Graded recall eval for the 5.1M title/author search paths.

Answers: "if a user types a paper's own title, does title search find it —
and at what rank?" and the same for an author's surname. Semantic already
has a full n=1000 float-GT eval; these two modes had latency numbers only.

Method: 500 random docs (seed 42, title > 10 chars / authors non-empty).
  Title mode: query = the doc's full title (AND of tokens, as the app builds
  it); also a 3-token variant as a realistic partial-title query.
  Author mode: query = one randomly chosen author surname.
Both go through the SAME adaptive SQL as YPMetadataStore (count-gated
bm25 / bounded two-phase). Metrics: hit@1/5/10/20, median rank (capped 100).

Usage: python3 scripts/eval_title_author_recall.py [n]
"""
import os, random, re, sqlite3, sys, time

DB = os.path.expanduser("~/yellow_phoenix/data/papers_5m.db")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
BUDGET, CAP = 6_000, 20_000
TOKEN = re.compile(r"[a-z0-9]+")

db = sqlite3.connect(DB)
db.execute("PRAGMA mmap_size=268435456")


def quoted(query: str) -> str:
    # Mirrors YPMetadataStore.quotedMatchQuery: every maximal alnum run is a
    # term (hyphens separate — quoted phrases are unsupported on
    # detail='column' tables), diacritics folded to match unicode61.
    import unicodedata
    q = "".join(c for c in unicodedata.normalize("NFKD", query.lower())
                if not unicodedata.combining(c))
    return " ".join(f'"{t}"' for t in TOKEN.findall(q))


def pin_and_phrase(query: str, ids, kmax: int):
    """Mirrors YPMetadataStore.pinExactAndPhraseBoost: exact normalized-title
    match pins to #1; multi-word queries: phrase-containing titles jump ahead."""
    import unicodedata as ud
    def norm(s):
        s = "".join(c for c in ud.normalize("NFKD", s.lower())
                    if not ud.combining(c))
        return " ".join(re.findall(r"[a-z0-9]+", s))
    nq = norm(query)
    if not nq:
        return ids[:kmax]
    metas = {r[0]: r[1] or "" for r in db.execute(
        f"SELECT id, title FROM meta WHERE id IN ({','.join('?'*len(ids))})", ids)} if ids else {}
    pinned, rest = [], []
    for i in ids:
        if not pinned and norm(metas.get(i, "")) == nq:
            pinned.append(i)
        else:
            rest.append(i)
    if " " in nq and rest:
        boosted = [i for i in rest if nq in norm(metas.get(i, ""))]
        rest = boosted + [i for i in rest if i not in set(boosted)]
    return (pinned + rest)[:kmax]


def adaptive(table: str, query: str, kmax: int = 100):
    """Replicates YPMetadataStore.searchFTS/searchAuthors SQL. Returns ranked ids."""
    mq = quoted(query)
    if not mq:
        return []
    n = db.execute(f"SELECT COUNT(*) FROM {table} WHERE {table} MATCH ?",
                   (mq,)).fetchone()[0]
    if n <= BUDGET:
        ids = [r[0] for r in db.execute(
            f"SELECT rowid FROM {table} WHERE {table} MATCH ?"
            f" ORDER BY bm25({table}) LIMIT ?", (mq, kmax * 3))]
        return pin_and_phrase(query, ids, kmax) if table == "fts" else ids[:kmax]
    order = ("LENGTH(m.title) ASC, m.year DESC" if table == "fts"
             else "LENGTH(m.authors) ASC, m.year DESC")
    ids = [r[0] for r in db.execute(
        f"WITH c AS (SELECT rowid FROM {table} WHERE {table} MATCH ?"
        f" ORDER BY rowid DESC LIMIT {CAP})"
        f" SELECT m.id FROM c JOIN meta m ON m.id = c.rowid"
        f" ORDER BY {order} LIMIT ?",
        (mq, kmax * 3))]
    return pin_and_phrase(query, ids, kmax) if table == "fts" else ids[:kmax]


def rank_of(ids, target):
    try:
        return ids.index(target) + 1
    except ValueError:
        return None


def summarize(name, ranks, n):
    hits = {k: sum(1 for r in ranks if r is not None and r <= k) / n
            for k in (1, 5, 10, 20)}
    got = [r for r in ranks if r is not None]
    med = sorted(got)[len(got) // 2] if got else "-"
    print(f"{name:28s} n={n}  hit@1 {hits[1]:.1%}  hit@5 {hits[5]:.1%}  "
          f"hit@10 {hits[10]:.1%}  hit@20 {hits[20]:.1%}  "
          f"found {len(got)/n:.1%}  median rank {med}")


rng = random.Random(42)
total = db.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
docs = []
while len(docs) < N:
    i = rng.randrange(total)
    row = db.execute("SELECT title, authors FROM meta WHERE id=? AND "
                     "title IS NOT NULL AND LENGTH(title)>10 AND "
                     "authors IS NOT NULL AND authors!=''", (i,)).fetchone()
    if row:
        docs.append((i, row[0], row[1]))

t0 = time.time()
full_ranks, part_ranks, auth_ranks = [], [], []
for j, (pid, title, authors) in enumerate(docs):
    ids = adaptive("fts", title)
    full_ranks.append(rank_of(ids, pid))
    toks = title.split()
    if len(toks) >= 3:
        mids = adaptive("fts", " ".join(toks[:3]))
        part_ranks.append(rank_of(mids, pid))
    surnames = [e.split()[-1] for e in authors.split("; ") if e.split()]
    if surnames:
        a_ids = adaptive("authors_fts", rng.choice(surnames))
        auth_ranks.append((rank_of(a_ids, pid)))
    if (j + 1) % 100 == 0:
        print(f"  {j+1}/{N} ({time.time()-t0:.0f}s)", flush=True)

print()
summarize("title: full title", full_ranks, N)
summarize("title: first 3 tokens", part_ranks, len(part_ranks))
summarize("author: random surname", auth_ranks, len(auth_ranks))
