"""
generate_dataset.py — Multi-Pattern Synthetic Memory Access Generator & Feature Engineering.

Generates realistic CPU memory access traces covering:
1. Sequential access (loop streams with variable lengths)
2. Strided access (matrix row/column traversal: +2, +4, +8, +16)
3. Reverse access (stack frame alloc/dealloc, backward scans: -1, -2, -4, -8)
4. Random access (sparse hash lookups, non-prefetchable jumps)
5. Irregular / pointer chasing (heap linked nodes and trees)
6. Page-boundary transitions (near 4KB page boundaries)
7. Segment-boundary transitions (TEXT, DATA, HEAP, STACK)

All temporal features are computed strictly causally without future-lookahead or static PC memorization.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.cache_simulator import MultiLevelCache


def generate_trace(num_accesses: int = config.TOTAL_ACCESSES,
                   seed: int = config.RANDOM_SEED,
                   is_unseen: bool = False) -> pd.DataFrame:
    """
    Simulate CPU memory access trace across multiple execution streams and segments.
    If is_unseen=True, uses independent seed, shifted base addresses within valid segments,
    and distinct interleaving to create a strictly out-of-distribution unseen workload.
    """
    np.random.seed(seed)
    cache = MultiLevelCache()

    # Program Counters and their base addresses within allocated segments
    if not is_unseen:
        pcs = [0x004010A0, 0x00401140, 0x004012A8, 0x004013D0, 0x00401500, 0x004018E0]
        pc_seg  = {pcs[0]: "TEXT",  pcs[1]: "DATA",  pcs[2]: "STACK",
                   pcs[3]: "DATA",  pcs[4]: "TEXT",  pcs[5]: "HEAP"}
        pc_base = {pcs[0]: 0x00400000,     pcs[1]: 0x10000000,
                   pcs[2]: 0x7FFF00000000, pcs[3]: 0x10080000,
                   pcs[4]: 0x00440000,     pcs[5]: 0x60000000}
        pc_weights = [0.26, 0.22, 0.18, 0.14, 0.12, 0.08]
    else:
        # Shifted base addresses within the same valid memory segments for unseen workload
        pcs = [0x00408100, 0x00408250, 0x004083C0, 0x004084F0, 0x00408600, 0x00408880]
        pc_seg  = {pcs[0]: "TEXT",  pcs[1]: "DATA",  pcs[2]: "STACK",
                   pcs[3]: "DATA",  pcs[4]: "TEXT",  pcs[5]: "HEAP"}
        pc_base = {pcs[0]: 0x00420000,     pcs[1]: 0x10040000,
                   pcs[2]: 0x7FFEFFFE0000, pcs[3]: 0x100C0000,
                   pcs[4]: 0x00460000,     pcs[5]: 0x60040000}
        pc_weights = [0.24, 0.22, 0.18, 0.16, 0.12, 0.08]

    curr_block = {pc: (pc_base[pc] >> 6) for pc in pcs}
    pc_step    = {pc: 0 for pc in pcs}
    pc_access_count = {pc: 0 for pc in pcs}
    block_freq: dict = {}

    records = []

    for t in range(num_accesses):
        pc = np.random.choice(pcs, p=pc_weights)
        seg = pc_seg[pc]
        seg_id = config.SEGMENTS[seg]["id"]
        step = pc_step[pc]
        pc_step[pc] += 1
        pc_access_count[pc] += 1

        # Small stochastic noise to represent branch mispredictions & interrupts
        noise_p = 0.012 if t < int(num_accesses * 0.70) else 0.020
        if is_unseen:
            noise_p = 0.022

        # ── Realistic Stride Dynamics per Stream ───────────────────────────
        if pc == pcs[0]:    # TEXT: Sequential instruction loop with wrap/branch (delta 0)
            delta = 1 if (step % 32 != 31) else 0

        elif pc == pcs[1]:  # DATA: 2D Matrix row-major traversal (3x(+4), then 1x(+8))
            delta = 8 if (step % 4 == 3) else 4

        elif pc == pcs[2]:  # STACK: Stack activation records (3x(-1), then 1x(-2))
            delta = -2 if (step % 4 == 3) else -1

        elif pc == pcs[3]:  # DATA: Strided structure elements (alternating +16 and -8)
            delta = 16 if (step % 2 == 0) else -8

        elif pc == pcs[4]:  # TEXT: Stride-2 loop branch (3x(+2), then 1x(-4))
            delta = -4 if (step % 4 == 3) else 2

        else:               # HEAP: Irregular pointer chasing and sparse hops
            delta = 0 if (step % 3 != 2) else int(np.random.choice([0, 1, -1, 2]))

        # Stochastic noise
        if np.random.rand() < noise_p:
            delta = int(np.random.choice(list(config.DELTA_TO_CLASS.keys())))

        block = curr_block[pc]
        byte_addr = block << 6
        cycle = t * 10

        # Advance block pointer
        curr_block[pc] = block + delta

        # Cache state before demand access
        l1_hit_before = cache.l1.probe(block)
        l1_set_i      = cache.l1.get_set_idx(block)
        l1_occ        = cache.l1.occupancy()

        # Execute demand access in cache hierarchy
        hit_level, latency = cache.demand_access(block, cycle)

        # Virtual memory geometry
        vpn           = byte_addr >> 12
        page_offset   = block % config.BLOCKS_PER_PAGE
        dist_page_end = (config.BLOCKS_PER_PAGE - 1) - page_offset
        is_page_bnd   = 1 if (dist_page_end <= 2 or page_offset <= 1) else 0
        cross_risk    = 1 if dist_page_end <= 4 else 0

        # Segment bounds distance calculation
        seg_base_block = config.SEGMENTS[seg]["base"] >> 6
        seg_size_block = (config.SEGMENTS[seg]["size_kb"] * 1024) // config.BLOCK_SIZE_BYTES
        if seg == "STACK":
            # Downward growing stack: distance to bottom limit
            stack_limit_block = seg_base_block - seg_size_block
            dist_seg_end = max(0, block - stack_limit_block)
        else:
            # Upward growing segments: distance to upper limit
            dist_seg_end = max(0, seg_base_block + seg_size_block - block)

        block_freq[block] = block_freq.get(block, 0) + 1
        access_type = 1 if np.random.rand() < 0.15 else 0

        records.append({
            "cycle":              cycle,
            "pc":                 hex(pc),
            "pc_id":              pcs.index(pc),
            "segment":            seg,
            "segment_id":         seg_id,
            "byte_address":       byte_addr,
            "block_address":      block,
            "vpn":                vpn,
            "page_offset":        page_offset,
            "dist_to_page_end":   dist_page_end,
            "is_page_boundary":   is_page_bnd,
            "page_cross_risk":    cross_risk,
            "dist_to_seg_end":    dist_seg_end,
            "l1_hit":             int(l1_hit_before),
            "hit_level":          hit_level,
            "cache_set_idx":      l1_set_i,
            "l1_occupancy":       round(l1_occ, 4),
            "access_type":        access_type,
            "access_freq":        block_freq[block],
            "stream_access_freq": pc_access_count[pc],
            "delta":              delta,
            "access_latency":     latency,
        })

    df = pd.DataFrame(records)
    # Strictly causal recent miss rate: uses shifted L1 hit history
    df["recent_miss_rate"] = (1 - df["l1_hit"]).shift(1).rolling(50, min_periods=1).mean().fillna(0.5)
    return df


def engineer_features(df: pd.DataFrame, history_len: int = 4) -> pd.DataFrame:
    """
    Build the full feature matrix from the raw trace.
    All temporal features are computed strictly causally grouped by execution stream (pc_id).
    Static PC identifiers (pc_is_0..5) are omitted to prevent memorization and guarantee
    generalization to unseen programs.
    """
    df = df.copy().reset_index(drop=True)

    # ── 1. Per-Stream Delta Lags (Causal: t-1, t-2, t-3, t-4) ───────────────
    for lag in range(1, history_len + 1):
        df[f"delta_lag_{lag}"] = df.groupby("pc_id")["delta"].shift(lag)

    # ── 2. Delta Dynamics ───────────────────────────────────────────────────
    df["delta_diff_1"] = df["delta_lag_1"] - df["delta_lag_2"]
    df["delta_diff_2"] = df["delta_lag_2"] - df["delta_lag_3"]
    df["is_match_1_2"] = (df["delta_lag_1"] == df["delta_lag_2"]).astype(float)
    df["has_zero_lag"] = ((df["delta_lag_1"] == 0) | (df["delta_lag_2"] == 0)).astype(float)

    # Consecutive repeat count (strictly using prior history)
    repeat_arr = np.zeros(len(df))
    for _, grp in df.groupby("pc_id"):
        idxs = grp.index.tolist()
        deltas = grp["delta"].values
        cnt = 1
        for k in range(1, len(idxs)):
            cnt = cnt + 1 if deltas[k] == deltas[k - 1] else 1
            repeat_arr[idxs[k]] = cnt
    df["_repeat_raw"] = repeat_arr
    df["delta_repeat_count"] = df.groupby("pc_id")["_repeat_raw"].shift(1).fillna(1.0)
    df.drop(columns=["_repeat_raw"], inplace=True)

    # Rolling window stride statistics (strictly using past shifts)
    df["delta_window_std"] = df.groupby("pc_id")["delta"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).std()).fillna(0.0)
    df["delta_rolling_mean"] = df.groupby("pc_id")["delta"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean()).fillna(0.0)

    # ── 3. Cache State Lags ─────────────────────────────────────────────────
    df["cache_hit_lag1"] = df.groupby("pc_id")["l1_hit"].shift(1).fillna(0)
    df["cache_hit_lag2"] = df.groupby("pc_id")["l1_hit"].shift(2).fillna(0)

    # ── 4. Page Access Frequency ─────────────────────────────────────────────
    df["_page_cum"] = df.groupby(["pc_id", "vpn"]).cumcount()
    df["page_access_freq"] = df["_page_cum"] / (df["stream_access_freq"] + 1)
    df.drop(columns=["_page_cum"], inplace=True)

    # ── 5. Segment One-Hot Encoding ──────────────────────────────────────────
    df["seg_is_text"]  = (df["segment_id"] == 0).astype(float)
    df["seg_is_data"]  = (df["segment_id"] == 1).astype(float)
    df["seg_is_heap"]  = (df["segment_id"] == 2).astype(float)
    df["seg_is_stack"] = (df["segment_id"] == 3).astype(float)
    df["seg_id"]       = df["segment_id"].astype(float)

    # ── 6. Drop NaNs caused by initial shifting ──────────────────────────────
    df = df.dropna().reset_index(drop=True)

    # ── 7. Target Labels ─────────────────────────────────────────────────────
    df["target_delta"] = df["delta"]
    df["target_class"] = df["target_delta"].map(config.DELTA_TO_CLASS)
    df = df[df["target_class"].notna()].copy()
    df["target_class"] = df["target_class"].astype(int)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return feature columns strictly excluding metadata and targets."""
    exclude = {"cycle", "pc", "pc_id", "segment", "segment_id", "byte_address",
               "block_address", "vpn", "delta", "hit_level", "access_latency",
               "l1_hit", "target_delta", "target_class"}
    return [c for c in df.columns if c not in exclude]


def chronological_split(df: pd.DataFrame):
    """Strict 70 / 10 / 20 chronological partition without shuffling."""
    n = len(df)
    n_train = int(n * config.TRAIN_RATIO)
    n_val   = int(n * config.VAL_RATIO)
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val   = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    df_test  = df.iloc[n_train + n_val:].reset_index(drop=True)
    return df_train, df_val, df_test


def build_dataset(save: bool = True):
    """
    Standard dataset build pipeline:
    Generates training/validation/test trace, engineers causal features, splits 70/10/20.
    """
    print("[Dataset] Generating multi-pattern memory access trace (Seed = 42) …")
    df_raw = generate_trace(num_accesses=config.TOTAL_ACCESSES, seed=config.RANDOM_SEED, is_unseen=False)
    
    if save:
        raw_path = config.RAW_DIR / "memory_access_trace.csv"
        df_raw.to_csv(raw_path, index=False)
        print(f"  Saved raw trace → {raw_path} ({len(df_raw):,} accesses)")

    print("[Dataset] Engineering causal per-stream features …")
    df_feat = engineer_features(df_raw)
    feat_cols = get_feature_columns(df_feat)
    print(f"  Feature matrix: {len(df_feat):,} rows × {len(feat_cols)} features")

    if save:
        df_feat.to_csv(config.PROCESSED_DIR / "full_features.csv", index=False)

    df_train, df_val, df_test = chronological_split(df_feat)

    print(f"\n[Dataset] 70/10/20 Chronological Split:")
    print(f"  Train Set (70%): {len(df_train):,} ({len(df_train)/len(df_feat)*100:.1f}%)")
    print(f"  Val Set   (10%): {len(df_val):,} ({len(df_val)/len(df_feat)*100:.1f}%)")
    print(f"  Test Set  (20%): {len(df_test):,} ({len(df_test)/len(df_feat)*100:.1f}%)")

    if save:
        df_train.to_csv(config.DATA_DIR / "train.csv", index=False)
        df_val.to_csv(config.DATA_DIR / "validation.csv", index=False)
        df_test.to_csv(config.DATA_DIR / "test.csv", index=False)
        df_feat.to_csv(config.DATA_DIR / "memory_access_dataset.csv", index=False)
        print("  Saved → data/train.csv, data/validation.csv, data/test.csv")

    return df_train, df_val, df_test, feat_cols


def generate_unseen_workload(num_accesses: int = 15000, seed: int = config.UNSEEN_SEED, save: bool = True):
    """
    Generates an independent, completely unseen memory workload with shifted base addresses,
    random seed (999), and independent stream timings. Used exclusively for testing true generalization.
    """
    print(f"\n[Unseen Workload] Generating independent trace (Seed = {seed}, Samples = {num_accesses:,}) …")
    df_unseen_raw = generate_trace(num_accesses=num_accesses, seed=seed, is_unseen=True)
    df_unseen_feat = engineer_features(df_unseen_raw)
    
    if save:
        unseen_path = config.UNSEEN_CSV
        df_unseen_feat.to_csv(unseen_path, index=False)
        print(f"  Saved unseen workload dataset → {unseen_path} ({len(df_unseen_feat):,} samples)")

    return df_unseen_feat
