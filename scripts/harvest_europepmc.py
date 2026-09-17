#!/usr/bin/env python3
"""Harvest real PubMed abstracts from Europe PMC (EMBL-EBI) into phoenix db.

Europe PMC REST API explicitly supports text-mining/bulk use. Records are
deduped with INSERT OR IGNORE on pmid (id column), same as the NCBI baseline
parser, tagged pubmed_medline.

Strategy: walk publication dates forward from --start in fixed windows,
cursor-paginated, N polite parallel workers, until --target new rows are in
the db (or dates exhaust).

Usage:
    python3 scripts/harvest_europepmc.py --start 1996-01-01 --target 1450000 \
        --workers 4 --db data/phoenix_arxiv_1m.db
"""
import argparse
import json
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta

API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
INSERT = ("INSERT OR IGNORE INTO papers "
          "(id, source, title, authors, year, categories, abstract) "
          "VALUES (?,?,?,?,?,?,?)")
WINDOW_DAYS = 5
PAGE_SIZE = 1000


BAD_TITLES = {"[not available].", "[not available]"}


def clean_title(t: str) -> str:
    t = (t or "").replace("\n", " ").strip()
    if t.lower() in BAD_TITLES or len(t) < 5:
        return ""
    return t


def fetch_window(start: date, end: date, conn, lock, counter: dict, stop: threading.Event, log):
    """Cursor-paginate one [start, end] window and insert new papers."""
    query = f"SRC:MED AND FIRST_PDATE:[{start.isoformat()} TO {end.isoformat()}]"
    cursor = "*"
    url_base = API + "?" + urllib.parse.urlencode({
        "query": query, "format": "json", "pageSize": PAGE_SIZE,
        "resultType": "core",
    })
    pages = 0
    while not stop.is_set():
        url = url_base + "&cursorMark=" + urllib.parse.quote(cursor, safe="")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=120) as resp:
                    payload = json.load(resp)
                break
            except Exception as e:
                if attempt == 3:
                    log(f"    [!] {start} fetch failed: {e}")
                    return
                time.sleep(5 * (attempt + 1))
        else:
            return

        results = payload.get("resultList", {}).get("result", [])
        batch = []
        for r in results:
            pmid = (r.get("pmid") or "").strip()
            title = clean_title(r.get("title"))
            if not pmid or not title:
                continue
            authors = (r.get("authorString") or "").replace("\n", " ")[:500]
            abstract = (r.get("abstractText") or "").replace("\n", " ")[:8000]
            year = (r.get("pubYear") or "")[:4]
            batch.append((pmid, "pubmed_medline", title, authors, year, "", abstract))
        if batch:
            with lock:
                conn.execute("BEGIN IMMEDIATE")
                conn.executemany(INSERT, batch)
                counter["new"] += conn.execute("SELECT changes()").fetchone()[0]
                conn.commit()
        pages += 1
        if pages % 10 == 0:
            log(f"    {start}..{end}: page {pages}, new total {counter['new']:,}")
        nxt = payload.get("nextCursorMark")
        if not results or not nxt or nxt == cursor:
            return
        cursor = nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1996-01-01")
    ap.add_argument("--target", type=int, default=1_450_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--db", default="data/phoenix_arxiv_1m.db")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db, timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    before = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE source='pubmed_medline'").fetchone()[0]

    lock = threading.Lock()
    stop = threading.Event()
    counter = {"new": 0}
    lock_print = threading.Lock()

    def log(msg):
        with lock_print:
            print(msg, flush=True)

    def worker(wid, windows):
        tconn = sqlite3.connect(args.db, timeout=60)
        tconn.execute("PRAGMA busy_timeout=60000")
        try:
            for (s, e) in windows:
                if stop.is_set():
                    return
                fetch_window(s, e, tconn, lock, counter, stop, log)
                with lock:
                    done = counter["new"] >= args.target
                if done:
                    stop.set()
                    return
        finally:
            tconn.close()

    # Build window list forward from --start; enough to cover target.
    # ~14K MED/day (2024) => 1.45M needs ~104 days; build 4000 days of
    # windows (generous for older lower-volume years).
    start = date.fromisoformat(args.start)
    windows = []
    d = start
    for _ in range(800):
        windows.append((d, d + timedelta(days=WINDOW_DAYS - 1)))
        d += timedelta(days=WINDOW_DAYS)

    buckets = [windows[i::args.workers] for i in range(args.workers)]
    log(f"[+] harvesting {len(windows)} windows from {start} with "
        f"{args.workers} workers, target +{args.target:,} new rows")

    threads = [threading.Thread(target=worker, args=(i, b), daemon=True)
               for i, b in enumerate(buckets) if b]
    t0 = time.time()
    for t in threads:
        t.start()
    # monitor: refresh counter from db periodically
    while any(t.is_alive() for t in threads):
        time.sleep(15)
        with lock:
            cur = conn.execute(
                "SELECT COUNT(*) FROM papers WHERE source='pubmed_medline'"
            ).fetchone()[0]
            counter["new"] = cur - before
            if counter["new"] >= args.target:
                stop.set()
        log(f"  [mon] new rows: {counter['new']:,} / {args.target:,} "
            f"({(time.time()-t0)/60:.1f} min)")
    for t in threads:
        t.join(timeout=5)

    after = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE source='pubmed_medline'").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    maxrow = conn.execute("SELECT MAX(rowid) FROM papers").fetchone()[0]
    log(f"[+] done: pubmed {before:,} -> {after:,} (+{after-before:,}), "
        f"db total {total:,}, max rowid {maxrow:,}, "
        f"elapsed {(time.time()-t0)/60:.1f} min")
    conn.close()


if __name__ == "__main__":
    main()
