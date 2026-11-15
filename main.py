"""
main.py — Complete ML-Based Memory Prefetching End-to-End Pipeline.

Pipeline Steps:
  1. Multi-pattern dataset generation (Seed 42) & causal feature engineering.
  2. Automated data leakage & information boundary verification audit.
  3. Strict 70 / 10 / 20 chronological partition.
  4. Train all models (DT, RF, GB, Compact MLP 32-16, Next-Line Baseline).
  5. Select best model strictly using Validation Accuracy.
  6. Final evaluation on independent 20% test set.
  7. Out-of-distribution evaluation on 100% unseen workload (Seed 999).
  8. Paging and segmentation boundary safety verification suite.
  9. Permutation feature importance and ablation experiments.
  10. Tune confidence threshold (tau) on validation set.
  11. Adaptive prefetching & cache hierarchy simulation (L1/L2/L3/DRAM).
  12. Correct metrics verification (Coverage <= 100%, AMAT, hit rate, timeliness).
  13. Generate and save all 15 publication-grade visual graphs.
  14. Write comprehensive experiment summary report.
"""

import os, sys, time, warnings
warnings.filterwarnings("ignore")

# Force unbuffered UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

from src.generate_dataset   import build_dataset, get_feature_columns
from src.leakage_checker    import run_leakage_checks
from src.train_models       import train_all
from src.evaluate_model     import evaluate_on_test
from src.evaluate_unseen    import evaluate_unseen_workload
from src.feature_importance import compute_permutation_importance, run_ablation
from src.adaptive_prefetching import AdaptivePrefetcher
from simulation.page_manager import run_boundary_test_suite
import src.visualize as viz

import numpy as np
import pandas as pd


def banner(text):
    print(f"\n{'='*75}")
    print(f"  {text}")
    print(f"{'='*75}")


def main():
    t_start = time.time()

    banner("STEP 1: Dataset Generation & 70/10/20 Chronological Splitting")
    df_train, df_val, df_test, feat_cols = build_dataset(save=True)

    banner("STEP 2: Automated Data Leakage & Information Boundary Audit")
    run_leakage_checks(df_train, feat_cols)

    # Prepare raw arrays for ablation & modeling
    X_tr_raw = df_train[feat_cols].values.astype(float)
    y_tr     = df_train["target_class"].values.astype(int)
    X_va_raw = df_val[feat_cols].values.astype(float)
    y_va     = df_val["target_class"].values.astype(int)

    banner("STEP 3 & 4: Training All Models & Tuning Compact MLP (32–16)")
    results, best_name, scaler, X_tr, y_tr_s, X_va, y_va_s, X_te, y_te_s = \
        train_all(feat_cols, verbose=True)

    best_model = results[best_name]["model"]

    banner("STEP 5: Final Evaluation on Independent 20% Held-Out Test Set")
    test_eval_metrics = evaluate_on_test(best_model, X_te, y_te_s, model_name=best_name)

    banner("STEP 6: Out-of-Distribution Evaluation on 100% Unseen Workload (Seed 999)")
    unseen_metrics = evaluate_unseen_workload(best_model, scaler, feat_cols, seed=config.UNSEEN_SEED)

    banner("STEP 7: Memory Paging & Segmentation Boundary Safety Verification")
    df_boundary_tests = run_boundary_test_suite()
    print("\nBoundary Test Results:")
    for _, row in df_boundary_tests.iterrows():
        print(f"  [{row['status']}] {row['test_case']:32s} -> {row['actual_decision']}")

    df_boundary_tests.to_csv(config.RESULTS_DIR / "boundary_validation_metrics.csv", index=False)
    boundary_report_path = config.RESULTS_DIR / "boundary_validation_report.txt"
    with open(boundary_report_path, "w", encoding="utf-8") as f_:
        f_.write("=== MEMORY PAGING & SEGMENTATION BOUNDARY VERIFICATION REPORT ===\n\n")
        f_.write(df_boundary_tests.to_string(index=False))
        f_.write(f"\n\nTotal Boundary Tests: {len(df_boundary_tests)}\n")
        f_.write(f"Passed: {(df_boundary_tests['status'] == 'PASSED').sum()}\n")
        f_.write(f"Failed: {(df_boundary_tests['status'] == 'FAILED').sum()}\n")
    print(f"  Saved boundary verification report -> {boundary_report_path}")

    banner("STEP 8: Permutation Feature Importance & Ablation Analysis")
    df_imp = compute_permutation_importance(best_model, X_va, y_va_s, feat_cols, n_repeats=5)
    df_imp.to_csv(config.RESULTS_DIR / "feature_importance.csv", index=False)
    print("\nTop 5 Most Informative Features:")
    for _, row in df_imp.head(5).iterrows():
        print(f"  {row['feature']:25s}: {row['importance_mean']*100:.3f}% mean drop")

    df_ab = run_ablation(X_tr_raw, y_tr, X_va_raw, y_va, feat_cols)
    df_ab.to_csv(config.RESULTS_DIR / "ablation_results.csv", index=False)

    banner("STEP 9: Adaptive Prefetching Simulation & Accurate Cache Metrics")
    prefetcher = AdaptivePrefetcher(best_model, scaler)

    # 1. Tune confidence threshold on validation set
    best_tau, df_sweep = prefetcher.tune_threshold(df_val, feat_cols)
    df_sweep.to_csv(config.RESULTS_DIR / "threshold_sweep.csv", index=False)

    # 2. Simulate on independent test set
    print(f"\n[Adaptive Prefetcher] Simulating test set with tuned tau* = {best_tau:.2f} …")
    res_df, bl_m, ad_m = prefetcher.simulate_trace(df_test, feat_cols)
    res_df.to_csv(config.RESULTS_DIR / "prefetch_metrics.csv", index=False)

    print("\n" + "="*70)
    print("  CACHE & PREFETCHING PERFORMANCE EVALUATION (TEST SET)")
    print("="*70)
    print(f"  Baseline L1 Hit Rate        : {bl_m['l1_hit_rate_pct']:.2f}%")
    print(f"  Adaptive L1 Hit Rate        : {ad_m['l1_hit_rate_pct']:.2f}%  (+{ad_m['l1_hit_rate_pct'] - bl_m['l1_hit_rate_pct']:.2f}%)")
    print(f"  Baseline Overall Hit Rate   : {bl_m['overall_hit_rate_pct']:.2f}%")
    print(f"  Adaptive Overall Hit Rate   : {ad_m['overall_hit_rate_pct']:.2f}%  (+{ad_m['overall_hit_rate_pct'] - bl_m['overall_hit_rate_pct']:.2f}%)")
    print(f"  Baseline AMAT               : {bl_m['amat_cycles']:.2f} cycles")
    print(f"  Adaptive AMAT               : {ad_m['amat_cycles']:.2f} cycles  (-{bl_m['amat_cycles'] - ad_m['amat_cycles']:.2f} cycles)")
    print(f"  Prefetch Coverage (<=100%)  : {ad_m['coverage_pct']:.2f}%")
    print(f"  Useful Precision            : {ad_m['useful_precision_pct']:.2f}%")
    print(f"  Timely Precision            : {ad_m['timely_precision_pct']:.2f}%")
    print(f"  Cache Pollution Rate        : {ad_m['pollution_rate_pct']:.2f}%")
    print(f"  Prefetches Requested        : {ad_m['pf_requested']:,}")
    print(f"  Prefetches Issued (Net)     : {ad_m['pf_issued']:,}")
    print(f"  Timely Useful Hits          : {ad_m['pf_timely']:,}")
    print(f"  Late Useful Hits            : {ad_m['pf_late']:,}")
    print(f"  Unused / Evicted            : {ad_m['pf_unused']:,}")
    print(f"  Redundant Suppressed        : {ad_m['pf_redundant']:,}")
    print(f"  Page Hazard Crosses Blocked : {ad_m['boundary_stats']['page_blocked']:,}")
    print(f"  Segment Overflows Blocked   : {ad_m['boundary_stats']['seg_blocked']:,}")
    print("="*70)

    banner("STEP 10: Generating All 15 Publication-Quality Plots")
    # 1. Learning curves
    mlp_hist = results["Compact MLP (32-16)"]["history"]
    viz.plot_loss(mlp_hist)
    viz.plot_accuracy(mlp_hist)
    viz.plot_combined(mlp_hist)
    viz.plot_loss_vs_epoch(mlp_hist)
    viz.plot_acc_vs_epoch(mlp_hist)

    # 2. Confusion matrix & model comparison
    viz.plot_confusion_matrix(test_eval_metrics["confusion_matrix"])
    viz.plot_model_comparison(results)
    viz.plot_per_class(test_eval_metrics["per_class_p"],
                       test_eval_metrics["per_class_r"],
                       test_eval_metrics["per_class_f"])

    # 3. Feature importance & ablation
    viz.plot_feature_importance(df_imp)
    viz.plot_ablation(df_ab)

    # 4. Cache & prefetch performance
    viz.plot_threshold_sweep(df_sweep)
    viz.plot_cache_comparison(bl_m, ad_m)
    viz.plot_prefetch_breakdown(ad_m)
    viz.plot_amat(bl_m, ad_m)
    viz.plot_boundary_validation(df_boundary_tests, ad_m["boundary_stats"])

    banner("STEP 11: Writing Comprehensive Experiment Summary")
    summary_path = config.RESULTS_DIR / "experiment_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f_:
        f_.write("=======================================================================\n")
        f_.write("  ML-BASED MEMORY PREFETCHING SYSTEM — COMPREHENSIVE EXPERIMENT REPORT \n")
        f_.write("=======================================================================\n\n")
        f_.write(f"Best Model Selected        : {best_name}\n")
        f_.write(f"Validation Accuracy        : {results[best_name]['val_acc']*100:.2f}%\n")
        f_.write(f"Test Accuracy (20% heldout): {test_eval_metrics['accuracy']*100:.2f}%\n")
        f_.write(f"Unseen Workload Accuracy   : {unseen_metrics['unseen_accuracy']*100:.2f}%\n")
        f_.write(f"Macro F1-Score             : {test_eval_metrics['macro_f1']*100:.2f}%\n\n")
        f_.write(f"Baseline L1 Hit Rate       : {bl_m['l1_hit_rate_pct']:.2f}%\n")
        f_.write(f"Adaptive L1 Hit Rate       : {ad_m['l1_hit_rate_pct']:.2f}% (+{ad_m['l1_hit_rate_pct'] - bl_m['l1_hit_rate_pct']:.2f}%)\n")
        f_.write(f"Baseline AMAT              : {bl_m['amat_cycles']:.2f} cycles\n")
        f_.write(f"Adaptive AMAT              : {ad_m['amat_cycles']:.2f} cycles (-{bl_m['amat_cycles'] - ad_m['amat_cycles']:.2f} cycles)\n")
        f_.write(f"Standard Prefetch Coverage : {ad_m['coverage_pct']:.2f}% (<= 100% verified)\n")
        f_.write(f"Useful Precision           : {ad_m['useful_precision_pct']:.2f}%\n")
        f_.write(f"Timely Precision           : {ad_m['timely_precision_pct']:.2f}%\n")
        f_.write(f"Cache Pollution Rate       : {ad_m['pollution_rate_pct']:.2f}%\n")
        f_.write(f"Cross-Page Blocks          : {ad_m['boundary_stats']['page_blocked']:,}\n")
        f_.write(f"Segment Overflow Blocks    : {ad_m['boundary_stats']['seg_blocked']:,}\n\n")
        f_.write("Model Comparison Table:\n")
        f_.write(pd.read_csv(config.RESULTS_DIR / "model_comparison.csv").to_string(index=False))
        f_.write("\n\nAll graphs generated in results/graphs/\n")
    print(f"  Saved experiment summary -> {summary_path}")

    elapsed = time.time() - t_start
    banner(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.1f}s")


if __name__ == "__main__":
    main()
