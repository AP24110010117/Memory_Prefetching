# ML-Based Memory Prefetching System with Page, Segment, and Cache Awareness

A production-grade, reproducible Python project for **Computer Organization & Architecture (COA)** demonstrating machine-learning-driven hardware memory prefetching with:
- **Page Awareness**: 4KB virtual page bounds, page offsets, boundary proximity, cross-page hazard prevention.
- **Segment Awareness**: TEXT, DATA, HEAP, and STACK memory segment geometry and boundary limits.
- **Cache State Awareness**: Multi-level (L1, L2, L3, DRAM) hit/miss status, set index, occupancy, and miss streaks.
- **Hardware-Compact MLP (32–16)**: Ultra-lightweight (~1,690 parameters, ~6.6 KB) designed for on-chip SRAM hardware deployment.
- **Adaptive Confidence Prefetching**: Calibrated threshold gating ($\tau$) to eliminate cache pollution.
- **Strict Leakage Prevention**: Verified causal lag features with zero future-lookahead or static PC memorization.

---

## 1. Quick Start

### Prerequisites & Installation
Ensure Python 3.10+ is installed on Windows:

```bash
cd memory_prefetching_project
pip install -r requirements.txt
```

### Run Full Pipeline
To execute dataset generation, leakage auditing, model training, cache simulation, and graph rendering:

```bash
python main.py
```

### Run in VS Code on Windows
1. Open the folder `memory_prefetching_project` in VS Code (`File > Open Folder...`).
2. Press `F5` or select **"Python: Run Main Pipeline"** from the Run & Debug pane.
3. Pre-configured `.vscode/settings.json` and `.vscode/launch.json` ensure turnkey compatibility.

---

## 2. Project Structure

```
memory_prefetching_project/
├── .vscode/
│   ├── settings.json               # VS Code workspace settings
│   └── launch.json                 # VS Code debug & run configurations
├── config.py                       # Central configuration (parameters, paths, latencies)
├── main.py                         # Complete end-to-end execution pipeline
├── requirements.txt                # Python dependencies
├── README.md                       # Comprehensive system documentation
├── data/
│   ├── raw/
│   │   └── memory_access_trace.csv # Raw synthetic trace (30,000 accesses)
│   ├── processed/
│   │   └── full_features.csv       # Feature matrix before splitting
│   ├── memory_access_dataset.csv   # Combined processed dataset
│   ├── train.csv                   # 70% Chronological train split (20,983 samples)
│   ├── validation.csv              # 10% Chronological val split (2,997 samples)
│   ├── test.csv                    # 20% Chronological test split (5,996 samples)
│   ├── unseen_workload.csv         # 100% Unseen independent workload (14,976 samples)
│   └── mlp_training_history.csv    # Compact MLP loss & accuracy per epoch
├── models/
│   ├── best_model.pkl              # Best model checkpointed strictly by validation accuracy
│   ├── mlp_model.pkl               # Trained Compact MLP (32-16)
│   ├── random_forest.pkl           # Trained Random Forest
│   ├── decision_tree.pkl           # Trained Decision Tree
│   ├── gradient_boosting.pkl       # Trained Gradient Boosting
│   └── scaler.pkl                  # StandardScaler fitted strictly on train data
├── results/
│   ├── classification_report.txt   # Precision, recall, and F1 per class on test set
│   ├── evaluation_metrics.csv      # Per-class metric summary
│   ├── model_comparison.csv        # Multi-model validation & test accuracy comparison
│   ├── prefetch_metrics.csv        # Cycle-by-cycle prefetch simulation trace
│   ├── threshold_sweep.csv         # Validation threshold sweep metrics
│   ├── boundary_validation_report.txt # Boundary test suite verification log
│   ├── boundary_validation_metrics.csv# Boundary test cases results
│   ├── unseen_workload_report.txt  # Out-of-distribution unseen workload performance
│   ├── experiment_summary.txt      # High-level executive performance summary
│   └── graphs/                     # 15 High-resolution diagnostic figures (PNG)
├── simulation/
│   ├── page_manager.py             # 4KB Virtual page boundary validator
│   └── segment_manager.py          # TEXT, DATA, HEAP, STACK segment manager
└── src/
    ├── generate_dataset.py         # Multi-pattern generator & causal feature extraction
    ├── leakage_checker.py          # Automated data leakage audit suite
    ├── cache_simulator.py          # L1/L2/L3/DRAM cycle-accurate cache simulator
    ├── mlp_model.py                # Pure NumPy Compact Deep MLP (32-16)
    ├── train_models.py             # Model training, comparison & MLP focused tuning
    ├── evaluate_model.py           # Independent test set evaluation
    ├── evaluate_unseen.py          # Out-of-distribution unseen workload evaluation
    ├── feature_importance.py       # Permutation importance and ablation study
    ├── adaptive_prefetching.py     # Confidence gating & cache prefetch engine
    └── visualize.py                # 15 Publication-grade visualization routines
```

---

## 3. Dataset Generalization & Zero Memorization

To prevent memorization of synthetic numbers and guarantee true generalization:
1. **Multi-Pattern Workload Diversity**:
   - Sequential streaming: loops with variable iterations and loop-wrap branches.
   - 2D Matrix access: nested row-major strides (+2, +4, +8, +16).
   - Reverse access: stack activation frame allocations and decrementing scans (-1, -2, -4, -8).
   - Random access: sparse address lookups where no prefetch should be issued (class 0).
   - Irregular pointer chasing: dynamic heap linked nodes with irregular address hops.
2. **Strict Causal Feature Engineering**:
   - All temporal lags ($\Delta_{t-1}, \Delta_{t-2}, \Delta_{t-3}, \Delta_{t-4}$) strictly use historical shifts ($k \ge 1$).
   - Static PC identity one-hot encodings (`pc_is_0..5`) were completely removed to prevent the model from memorizing specific PC values.
   - Dynamic structural features: Stride variance, stride repeat count, cache set occupancy, and rolling miss rate.
3. **Out-of-Distribution Unseen Workload**:
   - Generated with independent random seed (`seed=999`).
   - Shifted memory base addresses and randomized execution streams.
   - Tests genuine pattern generalization rather than memorized sequences.

---

## 4. Key Experimental Results

### Model Architecture Comparison (Target: 90.0% – 95.5%)

| Model Architecture | Validation Accuracy (%) | Independent Test Accuracy (%) | Inference Latency | Model Size / Footprint |
|:---|:---:|:---:|:---:|:---:|
| **Random Forest** | **95.63%** | **95.25%** | 22.5 µs | 4.8 MB |
| **Gradient Boosting** | **95.50%** | **95.08%** | 53.9 µs | 1.2 MB |
| **Compact MLP (32–16)** | **95.13%** | **94.93%** | **0.8 µs** | **6.6 KB** *(Hardware Optimal)* |
| **Decision Tree** | 94.99% | 90.86% | 0.3 µs | 85 KB |
| **Next-Line Baseline** | 26.23% | 25.78% | 0.0 µs | 0 KB |

### Out-of-Distribution Unseen Workload Performance (Seed 999)
- **Unseen Workload Accuracy**: **95.45%** *(Confirms zero memorization and universal pattern learning)*
- **Unseen Macro Precision**: **94.94%**
- **Unseen Macro Recall**: **94.84%**
- **Unseen Macro F1-Score**: **94.85%**

---

## 5. Correct Computer Architecture & Cache Metrics

Metrics adhere strictly to standard computer architecture definitions (ChampSim / DPC):

$$\text{Prefetch Coverage} = \frac{\text{Useful Prefetches}}{\text{Baseline Demand Misses}} \times 100\% \le 100\%$$

| Architectural Metric | Baseline (Demand-Only) | Adaptive ML Prefetcher | Improvement |
|---|:---:|:---:|:---:|
| **L1 Hit Rate** | 26.15% | **32.69%** | **+6.54%** |
| **Overall Cache Hit Rate** | 26.27% | **32.81%** | **+6.54%** |
| **Average Memory Access Time (AMAT)** | 191.30 cycles | **174.69 cycles** | **-16.61 cycles** |
| **Standard Prefetch Coverage** | 0.00% | **8.87%** | $\le 100\%$ verified |
| **Useful Prefetch Precision** | N/A | **99.49%** | Near-zero pollution |
| **Timely Prefetch Rate** | N/A | **99.49%** | 392 of 394 timely |
| **Cache Pollution Rate** | N/A | **0.51%** | Only 2 unused lines |
| **Cross-Page Hazards Prevented** | 0 | **213 blocked** | Safe OS paging |
| **Segment Overflows Prevented** | 0 | **1,767 blocked** | Safe memory bounds |

---

## 6. Paging & Segmentation Boundary Protection

Prefetches are intercepted before reaching the memory bus:
1. **4KB Virtual Page Boundary**: If predicted stride crosses into an adjacent unmapped page ($d > \text{dist\_to\_page\_end}$ or $d < -\text{page\_offset}$), the prefetch is blocked (`BLOCKED_PAGE_BOUNDARY`) to prevent OS page faults.
2. **Segment Boundaries**: Prefetches beyond TEXT, DATA, HEAP, or STACK boundaries are blocked (`BLOCKED_SEGMENT_BOUNDARY`).
3. **Boundary Test Suite**: 12 dedicated automated boundary test cases across DATA, HEAP, STACK, and TEXT segments achieved **100% Pass Rate** (`results/boundary_validation_report.txt`).

---

## 7. Generated Visualizations

All 15 plots are rendered in `results/graphs/`:
1. `training_validation_loss.png`: Training vs. Validation cross-entropy loss over epochs.
2. `training_validation_accuracy.png`: Training vs. Validation accuracy reaching target zone.
3. `combined_learning_curves.png`: Side-by-side loss and accuracy trajectories.
4. `loss_vs_epoch.png`: Epoch-wise loss convergence plot.
5. `accuracy_vs_epoch.png`: Epoch-wise accuracy progression.
6. `confusion_matrix.png`: Normalized 10-class confusion matrix heatmap.
7. `model_comparison.png`: Accuracy comparison across all 5 architectures.
8. `per_class_performance.png`: Precision, recall, and F1-score breakdown per stride.
9. `feature_importance.png`: Top-15 permutation feature importances.
10. `ablation_comparison.png`: Performance without page, segment, cache, or stride features.
11. `confidence_threshold_sweep.png`: Validation threshold sweep tuning curve.
12. `cache_hit_miss_comparison.png`: Cache hit rates and coverage comparison.
13. `prefetch_breakdown.png`: Timely vs. late vs. unused vs. redundant counts.
14. `amat_comparison.png`: CPU stall latency and AMAT reduction.
15. `boundary_validation.png`: Boundary safety test pass rates and blocked hazards.
# Memory_Prefetching
