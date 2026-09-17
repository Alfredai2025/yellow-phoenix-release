#!/usr/bin/env python3
"""
Minimal Flask API exposing YPEngine.search() to the browser.
Run: python3 scripts/flask_api.py
Test: curl -X POST http://localhost:5000/search -H "Content-Type: application/json" -d '{"query":"neural network","top_k":5}'
"""
import os
import sys
import time
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, ".")
from yp_bridge import YPEngine
from yp_autonomic.agent import AutonomicAgent
from yp_autonomic.orchestrator import AutonomicOrchestrator

app = Flask(__name__)
CORS(app)  # allow browser to call from any origin

print("[api] Loading YPEngine (this may take 10s on cold start)...")
t0 = time.time()
engine = YPEngine(db_path="data/phoenix_arxiv_1m.db")
print(f"[api] Engine ready in {time.time()-t0:.2f}s")

_autonomic_agent: AutonomicAgent = None


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json() or {}
    query = data.get('query', '')
    top_k = data.get('top_k', 5)

    if not query:
        return jsonify({'error': 'query required'}), 400

    t0 = time.time()
    try:
        res = engine.search_with_sah(query, k=top_k, use_cascade=True, use_ghosts=False)
        results = res.get('results', [])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    elapsed_ms = (time.time() - t0) * 1000

    # Detect which path was used
    path = res.get('source', 'geometric' if len(query.split()) < 5 else 'hash')

    # Format results for JSON
    formatted = []
    for r in results:
        if isinstance(r, tuple) and len(r) == 2:
            score, payload = r
            if isinstance(payload, tuple) and len(payload) == 2:
                pid, title = payload
                formatted.append({'score': float(score), 'id': pid, 'title': title})
            else:
                formatted.append({'score': float(score), 'payload': str(payload)})
        else:
            formatted.append({'raw': str(r)})

    return jsonify({
        'query': query,
        'path': path,
        'time_ms': round(elapsed_ms, 2),
        'result_count': len(formatted),
        'results': formatted[:top_k]
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'engine_loaded': True,
        'dynamic_buckets': engine.rust.lib.yp_mesh_dynamic_bucket_count() if hasattr(engine, 'rust') else 0,
        'autonomic': _autonomic_agent.status() if _autonomic_agent else AutonomicOrchestrator().status()
    })


@app.route('/mesh/status', methods=['GET'])
def mesh_status():
    return jsonify({
        'dynamic_buckets': engine.rust.lib.yp_mesh_dynamic_bucket_count() if hasattr(engine, 'rust') else 0,
        'papers_loaded': 12702
    })


@app.route('/mesh/rebucket', methods=['POST'])
def mesh_rebucket():
    try:
        moved = engine._run_rebucket_cycle()
        return jsonify({
            'moved': moved,
            'dynamic_buckets': engine.rust.lib.yp_mesh_dynamic_bucket_count() if hasattr(engine, 'rust') else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/mesh/snapshot', methods=['POST'])
def mesh_snapshot():
    try:
        path = engine.save_snapshot()
        if path:
            return jsonify({'saved': path})
        return jsonify({'error': 'snapshot failed'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/mesh/restore', methods=['POST'])
def mesh_restore():
    data = request.get_json() or {}
    path = data.get('path', '')
    if not path or not os.path.exists(path):
        return jsonify({'error': 'valid snapshot path required'}), 400
    try:
        ok = engine.load_snapshot(path)
        return jsonify({'restored': ok})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/mesh/history', methods=['GET'])
def mesh_history():
    try:
        hist = engine.temporal_history()
        return jsonify({'history': hist})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/mesh/trending', methods=['GET'])
def mesh_trending():
    try:
        since = request.args.get('since', 0, type=int)
        trends = engine.temporal_trending(since)
        return jsonify({'trending': trends})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/autonomic/status", methods=["GET"])
def autonomic_status():
    """Return autonomic agent status."""
    if _autonomic_agent is None:
        orch = AutonomicOrchestrator()
        status = orch.status()
        return jsonify({"autonomic": status})
    return jsonify({"autonomic": _autonomic_agent.status()})


@app.route("/autonomic/start", methods=["POST"])
def autonomic_start():
    """Start the autonomic agent (if not running)."""
    global _autonomic_agent
    if _autonomic_agent and _autonomic_agent.running:
        return jsonify({"error": "Already running"}), 409

    from yp_autonomic.sensors.system_hygiene import SystemHygieneSensor
    from yp_autonomic.sensors.database_health import DatabaseHealthSensor
    from yp_autonomic.sensors.service_health import ServiceHealthSensor
    from yp_autonomic.sensors.resource_pressure import ResourcePressureSensor
    from yp_autonomic.sensors.encoder_drift import EncoderDriftSensor
    from yp_autonomic.sensors.improvement import ImprovementSensor
    from yp_autonomic.sensors.safety_guardian import SafetyGuardianSensor
    from yp_autonomic.sensors.prevention import PreventionSensor
    from yp_autonomic.actuators import ProcessActuator, DatabaseActuator, NotifyActuator

    sensors = [
        SystemHygieneSensor(), DatabaseHealthSensor(), ServiceHealthSensor(), ResourcePressureSensor(),
        EncoderDriftSensor(), ImprovementSensor(), SafetyGuardianSensor(), PreventionSensor(),
    ]
    actuators = {
        "process": ProcessActuator(),
        "database": DatabaseActuator(),
        "notify": NotifyActuator(),
    }
    _autonomic_agent = AutonomicAgent(sensors=sensors, actuators=actuators)

    import threading
    t = threading.Thread(target=_autonomic_agent.run, daemon=True, name="yp-autonomic")
    t.start()
    return jsonify({"started": True, "sensors": [s.name for s in sensors]})


@app.route("/autonomic/stop", methods=["POST"])
def autonomic_stop():
    """Stop the autonomic agent."""
    global _autonomic_agent
    if _autonomic_agent is None or not _autonomic_agent.running:
        return jsonify({"error": "Not running"}), 409
    _autonomic_agent.stop()
    _autonomic_agent = None
    return jsonify({"stopped": True})


if __name__ == '__main__':
    import waitress
    print("[api] Starting on http://localhost:5001")
    print("[api] Test: curl -X POST http://localhost:5001/search -H 'Content-Type: application/json' -d '{\"query\":\"neural network\"}'")
    waitress.serve(app, host='127.0.0.1', port=5001, threads=16)
