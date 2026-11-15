"""
adaptive_prefetching.py — Confidence-Based Adaptive Prefetching Engine.

Workflow:
  1. model.predict_proba() yields confidence for each stride class.
  2. Prefetch issued only if confidence >= threshold and predicted delta != 0.
  3. Memory safety validation:
     - Page boundary check (blocks cross-page prefetch)
     - Segment boundary check (blocks segment underflow/overflow)
  4. Cache occupancy guard (avoids pollution when L1 is saturated >= 95%).
  5. Multi-level routing (high confidence + small stride -> L1, else L2).
  6. Threshold (tau) tuned strictly on VALIDATION set, tested on TEST set.
  7. Accurate computer architecture metrics:
     Coverage = Useful Prefetches / Baseline Demand Misses <= 100%.
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.cache_simulator import MultiLevelCache
from simulation.page_manager import validate_prefetch_candidate


class AdaptivePrefetcher:
    """Confidence-based adaptive prefetch engine."""

    def __init__(self, model, scaler, threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD):
        self.model     = model
        self.scaler    = scaler
        self.threshold = threshold

    def simulate_trace(self, df: pd.DataFrame, feat_cols: list,
                       pq_size: int = 16) -> tuple:
        """
        Run cycle-accurate cache + prefetch simulation on trace `df`.
        Simulates:
          1. Demand-only baseline cache
          2. Adaptive prefetching cache
        Returns (results_df, baseline_metrics, adaptive_metrics).
        """
        X_raw  = df[feat_cols].values.astype(float)
        X_sc   = self.scaler.transform(X_raw)
        probs  = self.model.predict_proba(X_sc)
        preds  = probs.argmax(1)
        confs  = probs.max(1)
        deltas = [config.CLASS_TO_DELTA[c] for c in preds]

        bl_cache = MultiLevelCache(pq_size)   # Baseline: Demand-only
        ad_cache = MultiLevelCache(pq_size)   # Adaptive Prefetch

        blocks    = df["block_address"].values
        cycles    = df["cycle"].values if "cycle" in df.columns else np.arange(len(df)) * 10
        page_offs = df["page_offset"].values
        dist_page = df["dist_to_page_end"].values
        segments  = df["segment"].values if "segment" in df.columns else ["DATA"] * len(df)

        decisions = []
        issued_ds = []
        tgt_lvls  = []

        boundary_stats = {
            "page_blocked": 0,
            "seg_blocked": 0,
            "per_seg_blocked": {s: 0 for s in config.SEGMENTS}
        }

        for i in range(len(df)):
            blk  = int(blocks[i])
            cyc  = int(cycles[i])
            d    = deltas[i]
            conf = float(confs[i])
            d_pe = int(dist_page[i])
            p_of = int(page_offs[i])
            seg  = str(segments[i])

            # 1. Baseline demand access (without prefetch)
            bl_cache.demand_access(blk, cyc)

            # 2. Adaptive prefetcher demand access
            ad_cache.demand_access(blk, cyc)

            # 3. Prefetch candidate evaluation
            if conf >= self.threshold and d != 0:
                # Memory boundary safety (page & segment checks)
                is_safe, safety_reason = validate_prefetch_candidate(
                    block=blk, delta=d, page_offset=p_of,
                    dist_to_page_end=d_pe, segment_name=seg
                )

                if not is_safe:
                    decisions.append(safety_reason)
                    issued_ds.append(0)
                    tgt_lvls.append("NONE")
                    if safety_reason == "BLOCKED_PAGE_BOUNDARY":
                        boundary_stats["page_blocked"] += 1
                    else:
                        boundary_stats["seg_blocked"] += 1
                    if seg in boundary_stats["per_seg_blocked"]:
                        boundary_stats["per_seg_blocked"][seg] += 1
                    continue

                # L1 occupancy guard (suppress if L1 is heavily congested)
                if ad_cache.l1.occupancy() >= 0.95:
                    decisions.append("SUPPRESSED_L1_FULL")
                    issued_ds.append(0)
                    tgt_lvls.append("NONE")
                    continue

                tgt_blk = blk + d
                # Routing: high confidence and localized stride -> L1, else L2
                tgt = "L1" if (conf >= 0.80 and abs(d) <= 2) else "L2"
                ok, reason = ad_cache.issue_prefetch(tgt_blk, cyc, target=tgt)
                if ok:
                    decisions.append(f"PREFETCH_{tgt}")
                    issued_ds.append(d)
                    tgt_lvls.append(tgt)
                else:
                    decisions.append(f"SUPPRESSED_{reason}")
                    issued_ds.append(0)
                    tgt_lvls.append("NONE")
            else:
                reason = "LOW_CONF" if d != 0 else "ZERO_DELTA"
                decisions.append(f"NO_PREFETCH_{reason}")
                issued_ds.append(0)
                tgt_lvls.append("NONE")

        # Baseline metrics (demand misses = baseline DRAM accesses)
        baseline_demand_misses = bl_cache.get_demand_misses()
        bl_m = bl_cache.metrics(baseline_demand_misses=baseline_demand_misses)
        ad_m = ad_cache.metrics(baseline_demand_misses=baseline_demand_misses)
        ad_m["boundary_stats"] = boundary_stats

        res = df.copy()
        res["pred_class"]   = preds
        res["pred_delta"]   = deltas
        res["confidence"]   = confs
        res["decision"]     = decisions
        res["issued_delta"] = issued_ds
        res["target_level"] = tgt_lvls

        return res, bl_m, ad_m

    def tune_threshold(self, df_val: pd.DataFrame, feat_cols: list,
                       candidates=None) -> tuple:
        """
        Select best threshold using VALIDATION SET ONLY.
        Maximises Utility = Useful Prefetches - 1.5 * Pollution.
        Returns (best_tau, sweep_df).
        """
        if candidates is None:
            candidates = config.CONFIDENCE_SWEEP

        records   = []
        best_tau  = self.threshold
        best_util = -1e9

        print("\n[Adaptive Prefetcher] Tuning confidence threshold on validation set …")
        for tau in candidates:
            self.threshold = tau
            _, bl_m, ad_m = self.simulate_trace(df_val, feat_cols)
            util = ad_m["pf_useful"] - 1.5 * ad_m["pf_unused"]
            records.append({
                "threshold":           tau,
                "useful_precision_pct":ad_m["useful_precision_pct"],
                "coverage_pct":        ad_m["coverage_pct"],
                "pollution_rate_pct":  ad_m["pollution_rate_pct"],
                "overall_hit_rate_pct":ad_m["overall_hit_rate_pct"],
                "amat_cycles":         ad_m["amat_cycles"],
                "utility":             util,
            })
            if util > best_util:
                best_util = util
                best_tau  = tau

        df_sweep = pd.DataFrame(records)
        self.threshold = best_tau
        print(f"  Best Threshold tau* = {best_tau:.2f} (Max Utility = {best_util:.1f})")
        return best_tau, df_sweep
