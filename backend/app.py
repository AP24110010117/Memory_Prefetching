"""Local web backend for the memory prefetching dashboard."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

from flask import Flask, jsonify, send_from_directory

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT.parent / "frontend"
app = Flask(__name__, static_folder=None)
run_lock = threading.Lock()
run_state = {"running": False, "return_code": None, "message": "Ready", "run_id": 0}


@app.get("/")
def home():
    return send_from_directory(FRONTEND, "index.html")


@app.get("/data.js")
def dashboard_data_js():
    return send_from_directory(FRONTEND, "data.js")


@app.get("/results/graphs/<path:filename>")
def result_graph(filename):
    return send_from_directory(ROOT / "results" / "graphs", filename)


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "service": "Memory Prefetching API", "pipeline": run_state})


@app.get("/api/results")
def results():
    data_file = FRONTEND / "data.js"
    try:
        raw = data_file.read_text(encoding="utf-8").strip()
        prefix = "window.PF = "
        if not raw.startswith(prefix):
            raise ValueError("Dashboard data has an unexpected format")
        return jsonify(json.loads(raw[len(prefix):].rstrip(";")))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return jsonify({"error": f"Dashboard results are not ready: {exc}"}), 503


def _run_pipeline():
    try:
        result = subprocess.run(
            [sys.executable, "-u", "main.py"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", timeout=3600,
        )
        if result.returncode == 0:
            export = subprocess.run(
                [sys.executable, "export_data.py"], cwd=ROOT, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=120,
            )
            if export.returncode:
                raise RuntimeError(export.stderr[-1200:] or "Could not refresh dashboard data")
            run_state.update(running=False, return_code=0, message="Pipeline completed and dashboard refreshed")
        else:
            run_state.update(running=False, return_code=result.returncode,
                             message=(result.stdout or "Pipeline failed")[-2000:])
    except Exception as exc:
        run_state.update(running=False, return_code=1, message=str(exc))
    finally:
        run_lock.release()


@app.post("/api/run")
def start_pipeline():
    if not run_lock.acquire(blocking=False):
        return jsonify({"started": False, "message": "A pipeline run is already in progress"}), 409
    run_state.update(running=True, return_code=None, message="Generating data, training models, and refreshing results…",
                     run_id=run_state["run_id"] + 1)
    threading.Thread(target=_run_pipeline, daemon=True).start()
    return jsonify({"started": True, **run_state}), 202


@app.get("/api/run/status")
def pipeline_status():
    return jsonify(run_state)


if __name__ == "__main__":
    print("Memory Prefetching dashboard: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
