#!/usr/bin/env python3
"""Phoenix Bench API — server-side retrieval latency benchmark.

Run:
    source .venv/bin/activate
    python scripts/phoenix_bench_api.py

Test:
    curl -X POST http://localhost:5001/bench \
        -H "Content-Type: application/json" \
        -d '{"query":"machine learning","k":10,"scale":"100k"}'
"""

import os
import sys
import time
import json
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
CORS(app)


class ScaleEngine:
    """Lightweight embedding-HNSW engine for a single scale dataset."""

    SCALES = {
        "100k": {
            "embeddings": "data/paper_embeddings_100k.npy",
            "M": 16,
            "ef_construction": 200,
            "ef": 25,
        },
        "1m": {
            "embeddings": "data/paper_embeddings_arxiv_1m.npy",
            "M": 16,
            "ef_construction": 200,
            "ef": 50,
        },
    }

    def __init__(self):
        self._engines = {}
        self._model = None

    def _get_model(self):
        if self._model is None:
            from yp_engine import get_model

            print("[bench_api] Loading sentence encoder...")
            self._model = get_model()
        return self._model

    def _get_engine(self, scale: str):
        if scale not in self._engines:
            try:
                import hnswlib
            except ImportError as exc:
                raise RuntimeError("hnswlib required: pip install hnswlib") from exc

            cfg = self.SCALES.get(scale)
            if cfg is None:
                raise ValueError(f"Unknown scale '{scale}'. Use: {list(self.SCALES)}")

            path = cfg["embeddings"]
            if not os.path.exists(path):
                raise FileNotFoundError(f"Embeddings not found: {path}")

            print(f"[bench_api] Loading {path} for scale '{scale}'...")
            embs = np.load(path).astype(np.float32)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs = embs / np.maximum(norms, 1e-10)
            n, dim = embs.shape
            print(f"[bench_api]   {n:,} vectors, dim={dim}")

            print(
                f"[bench_api] Building HNSW (M={cfg['M']}, ef_construction={cfg['ef_construction']})..."
            )
            t0 = time.perf_counter()
            index = hnswlib.Index(space="cosine", dim=dim)
            index.init_index(
                max_elements=n,
                ef_construction=cfg["ef_construction"],
                M=cfg["M"],
            )
            index.add_items(embs)
            index.set_ef(cfg["ef"])
            print(f"[bench_api]   Built in {time.perf_counter() - t0:.1f}s")

            self._engines[scale] = (embs, index, cfg["ef"])

        return self._engines[scale]

    def search_hybrid_by_embedding(self, q_emb: np.ndarray, k: int = 10, scale: str = "100k"):
        """HNSW candidate fetch + exact cosine re-rank from a raw embedding."""
        embs, index, ef = self._get_engine(scale)
        q_norm = q_emb.astype(np.float32) / (np.linalg.norm(q_emb) + 1e-10)

        # Fetch enough candidates so that re-ranking by exact cosine can reach
        # 99%+ R@1.  Empirically k=10 needs ~50 candidates on MiniLM/ITQ hashes;
        # use k*5 as a simple safe rule for all scales.
        candidates = max(k * 5, 50)
        labels, _ = index.knn_query(q_norm.reshape(1, -1), k=candidates)
        cand_ids = np.asarray(labels[0], dtype=np.int64)

        sims = embs[cand_ids] @ q_norm
        # Select k+1 so we can exclude self and still return k results
        select = min(k + 1, len(cand_ids))
        top_local = np.argpartition(-sims, select - 1)[:select]
        top_local = top_local[np.argsort(-sims[top_local])]

        results = []
        for local_idx in top_local:
            global_idx = cand_ids[local_idx]
            results.append((float(sims[local_idx]), (str(int(global_idx)), "")))
        return results

    def search_hybrid(self, query_text: str, k: int = 10, scale: str = "100k"):
        model = self._get_model()
        q_emb = model.encode([query_text], convert_to_numpy=True)[0]
        return self.search_hybrid_by_embedding(q_emb, k=k, scale=scale)


engine = ScaleEngine()


@app.route("/bench", methods=["POST"])
def bench():
    data = request.get_json(force=True)
    query = data.get("query", "machine learning")
    k = int(data.get("k", 10))
    scale = data.get("scale", "100k")
    client_device = data.get("device", {})

    try:
        # Timed run (server-side latency)
        t0 = time.perf_counter()
        results = engine.search_hybrid(query, k=k, scale=scale)
        t1 = time.perf_counter()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    latency_us = round((t1 - t0) * 1e6, 1)

    return jsonify(
        {
            "query": query,
            "scale": scale,
            "k": k,
            "latency_us": latency_us,
            "results": [
                {
                    "score": round(float(score), 4),
                    "pid": pid,
                    "title": (title or "")[:80],
                }
                for score, (pid, title) in results
            ],
            "device": "server",
            "client_device": client_device,
            "timestamp": time.time(),
        }
    )


@app.route("/bench/health", methods=["GET"])
def bench_health():
    return jsonify(
        {
            "status": "ok",
            "scales": list(engine.SCALES.keys()),
            "loaded_scales": list(engine._engines.keys()),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PHOENIX_BENCH_PORT", "5001"))
    print(f"[bench_api] Starting on http://0.0.0.0:{port}")
    try:
        import waitress

        waitress.serve(app, host="0.0.0.0", port=port, threads=16)
    except ImportError:
        app.run(host="0.0.0.0", port=port, threaded=True)
