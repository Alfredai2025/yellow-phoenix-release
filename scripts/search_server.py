#!/usr/bin/env python3
"""Local HTTP search server for Yellow Phoenix iOS app."""
import json
import os
import sqlite3
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

try:
    from yp_bridge import YPEngine
except ImportError as e:
    print(f"[FATAL] Cannot import yp_bridge: {e}")
    sys.exit(1)

engine = None
db_conn = None
node_to_pid: dict = {}


def _rust_id(paper_id: str) -> int:
    """FNV-1a 64-bit hash — matches Rust hash_id_to_u64 exactly."""
    FNV_BASIS = 0xCBF29CE484222325
    FNV_PRIME = 0x100000001B3
    h = FNV_BASIS
    for b in str(paper_id).encode("utf-8"):
        h ^= b
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def init_engine():
    """Lazy-load the YPEngine, HNSW index, and metadata mappings."""
    global engine, db_conn, node_to_pid
    if engine is not None:
        return engine

    print("[INIT] Loading YPEngine ...")
    engine = YPEngine()

    hnsw_path = os.path.expanduser("~/yellow_phoenix/data/binary_hnsw_arxiv1m_m16.bin")
    if os.path.exists(hnsw_path):
        try:
            loaded = engine.load_hnsw(hnsw_path)
            print(f"[INIT] HNSW load result: {loaded}")
        except Exception as e:
            print(f"[WARN] HNSW load failed: {e}")
    else:
        print(f"[WARN] HNSW index not found at {hnsw_path}; server will use fallback stubs")

    # Connect to unified arXiv metadata DB and build node_id -> pid map
    db_path = os.path.expanduser("~/yellow_phoenix/data/phoenix_arxiv_1m.db")
    if os.path.exists(db_path):
        db_conn = sqlite3.connect(db_path, check_same_thread=False)
        db_conn.row_factory = sqlite3.Row
        print(f"[INIT] DB connected: {db_path}")

        try:
            rows = db_conn.execute(
                "SELECT id FROM papers WHERE yp_hash512 IS NOT NULL AND yp_hash512 != ''"
            ).fetchall()
            node_to_pid = {_rust_id(row["id"]): row["id"] for row in rows}
            print(f"[INIT] Built node_id -> pid map for {len(node_to_pid):,} papers")
        except Exception as e:
            print(f"[WARN] Failed to build node_id map: {e}")
    else:
        print(f"[WARN] DB not found at {db_path}; titles will be IDs only")

    return engine


def doc_count() -> int:
    try:
        if engine and engine._rust_hnsw is not None:
            return len(engine._rust_hnsw)
    except Exception:
        pass
    return 0


def lookup_paper(pid: str):
    """Fetch real metadata from SQLite DB by numeric pid."""
    if db_conn is None:
        return None
    try:
        cur = db_conn.cursor()
        cur.execute(
            "SELECT title, authors, abstract, year FROM papers WHERE id = ? LIMIT 1",
            (pid,),
        )
        row = cur.fetchone()
        if row:
            authors = []
            if row["authors"]:
                try:
                    authors = json.loads(row["authors"])
                except Exception:
                    authors = [row["authors"]]
            return {
                "title": row["title"] or f"Paper {pid}",
                "authors": authors if isinstance(authors, list) else [],
                "abstract": row["abstract"] or "",
                "year": row["year"] or 0,
            }
    except Exception as e:
        print(f"[WARN] DB lookup failed for {pid}: {e}")
    return None


def node_to_meta(node_id, score):
    """Convert an HNSW node id into a metadata dict."""
    pid = node_to_pid.get(int(node_id))
    meta = lookup_paper(pid) if pid else None
    if meta is None:
        meta = {
            "title": f"Paper {node_id}",
            "authors": [],
            "abstract": "",
            "year": 0,
        }
    meta["score"] = round(float(score), 6)
    return meta


class YPSearchHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {self.address_string()} - {fmt % args}")

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            eng = init_engine()
            self._send_json(200, {
                "status": "ok",
                "docs_loaded": doc_count(),
                "engine_ready": getattr(eng, '_hnsw_loaded', False),
            })
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/search":
            self._send_json(404, {"error": "Not found"})
            return

        content_len = int(self.headers.get("Content-Length", 0))
        if content_len == 0:
            self._send_json(400, {"error": "Empty body"})
            return

        body = self.rfile.read(content_len).decode("utf-8")
        try:
            req = json.loads(body)
        except json.JSONDecodeError as e:
            self._send_json(400, {"error": f"Invalid JSON: {e}"})
            return

        query = req.get("query", "").strip()
        top_k = int(req.get("top_k", 20))
        if not query:
            self._send_json(400, {"error": "Missing or empty 'query' field"})
            return

        eng = init_engine()
        try:
            results = eng.search(query, top_k=top_k)
        except Exception as e:
            self._send_json(500, {"error": f"Search failed: {e}"})
            return

        hits = []
        for r in results:
            score = 0.0
            node_id = None

            if isinstance(r, dict):
                score = float(r.get("score", 0.0))
                node_id = r.get("id", r.get("node_id"))
            elif isinstance(r, (list, tuple)) and len(r) >= 2:
                score = float(r[0])
                node_id = r[1]
            else:
                node_id = str(r)

            try:
                node_id = int(node_id)
            except (ValueError, TypeError):
                pass

            hits.append(node_to_meta(node_id, score))

        self._send_json(200, {"hits": hits})


def run_server(port=8765):
    init_engine()
    server = HTTPServer(("0.0.0.0", port), YPSearchHandler)
    print(f"[READY] YP Search Server on http://0.0.0.0:{port}")
    print(f"        POST /search   JSON {{query, top_k}}")
    print(f"        GET  /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Server stopped.")
        server.shutdown()


if __name__ == "__main__":
    run_server()
