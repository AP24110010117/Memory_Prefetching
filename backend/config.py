"""
Centralized Configuration for ML-Based Memory Prefetching System.
All parameters are defined here. Import config in every module.
"""

import os
from pathlib import Path

# ─── Project Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent
DATA_DIR       = PROJECT_ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
MODELS_DIR     = PROJECT_ROOT / "models"
RESULTS_DIR    = PROJECT_ROOT / "results"
GRAPHS_DIR     = RESULTS_DIR / "graphs"
SRC_DIR        = PROJECT_ROOT / "src"
SIM_DIR        = PROJECT_ROOT / "simulation"

# Auto-create all directories
for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, GRAPHS_DIR, SRC_DIR, SIM_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 42
UNSEEN_SEED = 999  # Independent seed for strictly unseen workload evaluation

# ─── Dataset Parameters ──────────────────────────────────────────────────────
TOTAL_ACCESSES   = 30000          # Synthetic trace length
DATASET_CSV      = DATA_DIR / "memory_access_dataset.csv"
TRAIN_CSV        = DATA_DIR / "train.csv"
VAL_CSV          = DATA_DIR / "validation.csv"
TEST_CSV         = DATA_DIR / "test.csv"
UNSEEN_CSV       = DATA_DIR / "unseen_workload.csv"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20

# ─── Virtual Memory Parameters ───────────────────────────────────────────────
PAGE_SIZE_BYTES  = 4096           # 4 KB pages
BLOCK_SIZE_BYTES = 64             # 64-byte cache line
BLOCKS_PER_PAGE  = PAGE_SIZE_BYTES // BLOCK_SIZE_BYTES  # 64 blocks/page

# ─── Memory Segments ─────────────────────────────────────────────────────────
SEGMENTS = {
    "TEXT":  {"id": 0, "base": 0x00400000, "size_kb": 512,  "type": "Sequential/Loops"},
    "DATA":  {"id": 1, "base": 0x10000000, "size_kb": 2048, "type": "Arrays/Matrices"},
    "HEAP":  {"id": 2, "base": 0x60000000, "size_kb": 2048, "type": "Dynamic/Pointers"},
    "STACK": {"id": 3, "base": 0x7FFF00000000, "size_kb": 512, "type": "Frames/Reverse"},
}

# ─── Cache Hierarchy ─────────────────────────────────────────────────────────
CACHE_CONFIG = {
    "L1": {"capacity_kb": 32,   "num_ways": 8,  "line_bytes": 64, "latency_cycles": 4},
    "L2": {"capacity_kb": 256,  "num_ways": 8,  "line_bytes": 64, "latency_cycles": 14},
    "L3": {"capacity_kb": 2048, "num_ways": 16, "line_bytes": 64, "latency_cycles": 40},
    "DRAM": {"latency_cycles": 200},
    "PREFETCH_QUEUE": {"capacity": 16},
}

# ─── Delta/Stride Classes ────────────────────────────────────────────────────
# These are the cache-block strides the ML model predicts
DELTA_CLASSES   = [1, 2, 4, 8, 16, -1, -2, -4, -8, 0]
DELTA_TO_CLASS  = {d: i for i, d in enumerate(DELTA_CLASSES)}
CLASS_TO_DELTA  = {i: d for i, d in enumerate(DELTA_CLASSES)}
CLASS_NAMES     = [
    "Stride +1", "Stride +2", "Stride +4", "Stride +8", "Stride +16",
    "Stride -1", "Stride -2", "Stride -4", "Stride -8", "No-Prefetch"
]
NUM_CLASSES = len(DELTA_CLASSES)

# ─── Compact MLP (32–16) Hyperparameters ────────────────────────────────────
# Compact architecture (32 -> 16) designed for ~6 KB on-chip SRAM budget
MLP_HIDDEN_SIZES   = [32, 16]       # Compact MLP (32-16)
MLP_EPOCHS         = 45
MLP_BATCH_SIZE     = 128
MLP_LEARNING_RATE  = 0.0035
MLP_L2_REG         = 2.0e-4
MLP_DROPOUT_RATE   = 0.06
MLP_PATIENCE       = 10             # Early stopping patience

# ─── Adaptive Prefetching ────────────────────────────────────────────────────
DEFAULT_CONFIDENCE_THRESHOLD = 0.65
CONFIDENCE_SWEEP = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# ─── Feature Groups (Excludes raw PCs to prevent memorization) ───────────────
PAGE_FEATURES    = ["page_offset", "dist_to_page_end", "is_page_boundary",
                    "page_access_freq", "page_cross_risk"]
SEGMENT_FEATURES = ["seg_is_text", "seg_is_data", "seg_is_heap", "seg_is_stack",
                    "dist_to_seg_end", "seg_id"]
CACHE_FEATURES   = ["cache_set_idx", "cache_hit_lag1", "cache_hit_lag2",
                    "l1_occupancy", "recent_miss_rate"]
STRIDE_FEATURES  = ["delta_lag_1", "delta_lag_2", "delta_lag_3", "delta_lag_4",
                    "delta_diff_1", "delta_diff_2", "is_match_1_2",
                    "delta_repeat_count", "delta_window_std", "delta_rolling_mean",
                    "has_zero_lag"]
CONTEXT_FEATURES = ["access_type", "access_freq", "stream_access_freq"]
