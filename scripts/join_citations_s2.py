#!/usr/bin/env python3
"""Join citation counts (cited_by) into papers_5m.db via Semantic Scholar.

Why: match-set containment for author search is 100%, but ranking common
surnames is a lottery without an authority signal (eval Sep 2026). S2's batch
endpoint accepts 1000 external ids per POST (arXiv / PMID / DOI) and returns
citationCount — 5.1M papers ≈ 5,100 calls ≈ ~25 min at a polite 4 req/s.

  export S2_API_KEY=...   # free key: https://www.semanticscholar.org/product/api
  python3 scripts/join_citations_s2.py

Idempotent/resumable: progress checkpoints to data/_build_papers5m/citations.checkpoint;
re-running skips completed ranges. After it finishes, the app automatically
uses citation-aware ordering (YPMetadataStore.hasCitations).

Adds: meta.cited_by INTEGER (NULL = unknown). Coverage stats printed at end.
"""
import json, os, re, sqlite3, sys, time, urllib.request

DB = os.path.expanduser("~/yellow_phoenix/data/papers_5m.db")
CKPT = os.path.join(os.path.expanduser("~/yellow_phoenix/data"),
                    "_build_papers5m", "citations.checkpoint")
BATCH = 1000
REQS_PER_SEC = float(os.environ.get("S2_RPS", "4"))


def s2_id(ext_id: str):
    e = (ext_id or "").strip()
    if not e:
        return None
    if e.startswith("arxiv1m:"):
        return "ARXIV:" + e.split(":", 1)[1]
    if e.startswith("arxiv_"):
        return "ARXIV:" + e[len("arxiv_"):]
    if re.fullmatch(r"\d+", e):
        return "PMID:" + e
    if e.startswith("10."):
        return "DOI:" + e
    return None  # e.g. "paper_..." internal ids — no external resolver


def main():
    api_key = os.environ.get("S2_API_KEY", "")
    if not api_key:
        print("WARNING: S2_API_KEY not set — running unauthenticated at 0.3 req/s")
        print("(anonymous pool = 100 req / 5 min per IP; expect ~9h for 5.1M).")
        print("Get a free key for 1 req/s: https://www.semanticscholar.org/product/api")
        rps = 0.3
    else:
        rps = REQS_PER_SEC
    db = sqlite3.connect(DB)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    cols = [r[1] for r in db.execute("PRAGMA table_info(meta)")]
    if "cited_by" not in cols:
        db.execute("ALTER TABLE meta ADD COLUMN cited_by INTEGER")
        db.commit()
        print("added meta.cited_by")

    done = set()
    if os.path.exists(CKPT):
        done = set(json.load(open(CKPT)))
        print(f"resuming: {len(done):,} batches already done")

    rows = db.execute("SELECT id, ext_id FROM meta").fetchall()
    jobs = [(i, s2_id(e)) for i, e in rows]
    eligible = [j for j in jobs if j[1]]
    print(f"total {len(rows):,} rows, {len(eligible):,} S2-eligible ids")

    url = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=citationCount"
    t0, calls, last = time.time(), 0, 0.0
    pending = []
    for lo in range(0, len(eligible), BATCH):
        if lo in done:
            continue
        chunk = eligible[lo:lo + BATCH]
        body = json.dumps({"ids": [s for _, s in chunk]}).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json",
                                              **({"x-api-key": api_key} if api_key else {})})
        # rate limit
        dt = time.time() - last
        if dt < 1.0 / rps:
            time.sleep(1.0 / rps - dt)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                last = time.time()
                break
            except urllib.error.HTTPError as ex:
                wait = int(ex.headers.get("Retry-After", "30")) if ex.code == 429 else 5 * (attempt + 1)
                print(f"  batch {lo}: HTTP {ex.code} — retry in {wait}s", flush=True)
                time.sleep(wait)
            except Exception as ex:
                wait = 5 * (attempt + 1)
                print(f"  batch {lo}: {ex} — retry in {wait}s", flush=True)
                time.sleep(wait)
        else:
            print(f"  batch {lo}: FAILED after retries, skipping")
            continue
        for (pid, _), item in zip(chunk, data):
            cc = item.get("citationCount") if item else None
            pending.append((cc if cc is not None else 0, pid))
        done.add(lo)
        calls += 1
        if calls % 25 == 0:
            db.executemany("UPDATE meta SET cited_by=? WHERE id=?", pending)
            db.commit()
            json.dump(sorted(done), open(CKPT, "w"))
            pending.clear()
            rate = calls / (time.time() - t0)
            eta = (len(eligible) // BATCH - len(done)) / max(rate, 1e-9) / 60
            print(f"  {len(done)}/{len(eligible)//BATCH} batches "
                  f"({rate:.1f} calls/s, eta {eta:.0f} min)", flush=True)
    if pending:
        db.executemany("UPDATE meta SET cited_by=? WHERE id=?", pending)
        db.commit()
        json.dump(sorted(done), open(CKPT, "w"))

    known = db.execute("SELECT COUNT(*), AVG(cited_by) FROM meta WHERE cited_by IS NOT NULL").fetchone()
    print(f"DONE in {(time.time()-t0)/60:.1f} min. cited_by known for "
          f"{known[0]:,} rows, mean {known[1]:.1f} citations.")


if __name__ == "__main__":
    main()
