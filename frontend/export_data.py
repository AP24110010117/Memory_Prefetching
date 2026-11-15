"""
export_data.py - bridge between the Python pipeline and the frontend.

Reads results/*.csv, results/*.txt and config.py, then writes frontend/data.js
(window.PF = {...}). Re-run after `python main.py` to refresh the dashboard:

    python frontend/export_data.py
"""
import json, re, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

R = ROOT / "results"
rec = lambda name: pd.read_csv(R / name).round(4).to_dict("records")

# headline numbers parsed from experiment_summary.txt
txt = (R / "experiment_summary.txt").read_text(encoding="utf-8")
def num(label):
    m = re.search(re.escape(label) + r"\s*:\s*([-\d,\.]+)", txt)
    return float(m.group(1).replace(",", "")) if m else None
summary = {
    "best_model": re.search(r"Best Model Selected\s*:\s*(.+)", txt).group(1).strip(),
    "test_acc": num("Test Accuracy (20% heldout)"),
    "unseen_acc": num("Unseen Workload Accuracy"),
    "macro_f1": num("Macro F1-Score"),
    "base_hit": num("Baseline L1 Hit Rate"),
    "ml_hit": num("Adaptive L1 Hit Rate"),
    "base_amat": num("Baseline AMAT"),
    "ml_amat": num("Adaptive AMAT"),
    "coverage": num("Standard Prefetch Coverage"),
    "precision": num("Useful Precision"),
    "pollution": num("Cache Pollution Rate"),
    "page_blocks": num("Cross-Page Blocks"),
    "seg_blocks": num("Segment Overflow Blocks"),
}

# decision breakdown from the cycle-by-cycle simulation trace
tr = pd.read_csv(R / "prefetch_metrics.csv", usecols=["segment", "decision", "byte_address"])
dec = tr.groupby(["segment", "decision"]).size().unstack(fill_value=0)
decisions = {seg: {k: int(v) for k, v in row.items()} for seg, row in dec.iterrows()}

# does the trace stay inside the segments declared in config.py?
inside = {}
for seg, s in config.SEGMENTS.items():
    t = tr[tr.segment == seg].byte_address
    lo, hi = (s["base"] - s["size_kb"] * 1024, s["base"]) if seg == "STACK" else (s["base"], s["base"] + s["size_kb"] * 1024)
    inside[seg] = round(float(((t >= lo) & (t <= hi)).mean()) * 100, 1) if len(t) else None

data = {
    "summary": summary,
    "models": rec("model_comparison.csv"),
    "sweep": rec("threshold_sweep.csv"),
    "ablation": rec("ablation_results.csv"),
    "importance": rec("feature_importance.csv")[:12],
    "perclass": rec("evaluation_metrics.csv"),
    "boundary": rec("boundary_validation_metrics.csv"),
    "decisions": decisions,
    "seg_inside_pct": inside,
    "segments": {k: {"base": hex(v["base"]), "kb": v["size_kb"], "type": v["type"]} for k, v in config.SEGMENTS.items()},
    "cache": config.CACHE_CONFIG,
    "page_bytes": config.PAGE_SIZE_BYTES,
    "graphs": sorted(p.name for p in (R / "graphs").glob("*.png")),
}
out = Path(__file__).resolve().parent / "data.js"
out.write_text("window.PF = " + json.dumps(data, default=float) + ";\n", encoding="utf-8")
print(f"wrote {out}  ({out.stat().st_size/1024:.1f} KB)")
