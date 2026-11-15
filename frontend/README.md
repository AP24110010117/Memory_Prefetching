# Frontend dashboard

Open `frontend/index.html` in any browser (double-click, no server or install needed).

| File | Purpose |
|---|---|
| `index.html` | The dashboard: Overview, Live simulator, Results, Page & segment safety |
| `data.js` | Auto-generated from `results/` and `config.py` (do not edit by hand) |
| `export_data.py` | Regenerates `data.js` from the Python pipeline outputs |

## Keeping it in sync with the backend

```bash
python main.py                    # re-run the pipeline (updates results/)
python frontend/export_data.py    # refresh frontend/data.js
```

Then reload `index.html`. The Results tab also shows the PNGs from `results/graphs/`, so keep the folder layout unchanged.

## What is real and what is a demo

* **Overview / Results / Safety tabs** show the real numbers produced by the trained models and cache simulator.
* **Live simulator** runs in the browser with a simple delta-history predictor and instant prefetch fill. It illustrates page, segment, cache and confidence gating, but it is not the trained Random Forest / MLP.
