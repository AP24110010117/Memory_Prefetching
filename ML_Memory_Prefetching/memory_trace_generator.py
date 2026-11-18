"""Synthetic trace generation and trace-level dataset splitting."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate_trace(pattern="mixed", length=500, address_space=128, seed=42,
                   trace_id="trace_0"):
    if length < 1 or address_space < 2:
        raise ValueError("length must be >= 1 and address_space must be >= 2")
    rng = np.random.default_rng(seed)
    pattern = pattern.lower()
    if pattern == "sequential":
        blocks = np.arange(length) % address_space
    elif pattern == "random":
        blocks = rng.integers(0, address_space, size=length)
    elif pattern == "repeated":
        hot = max(2, min(8, address_space))
        blocks = rng.choice(np.arange(hot), size=length,
                            p=np.arange(hot, 0, -1) / np.arange(hot, 0, -1).sum())
    elif pattern == "loop":
        loop = np.arange(min(12, address_space))
        blocks = np.resize(loop, length)
    elif pattern == "mixed":
        chunks = []
        while sum(len(x) for x in chunks) < length:
            mode = rng.choice(["sequential", "random", "repeated", "loop"])
            n = min(int(rng.integers(20, 70)), length - sum(len(x) for x in chunks))
            chunks.append(generate_trace(mode, n, address_space, int(rng.integers(1, 1_000_000)), trace_id)["block"].to_numpy())
        blocks = np.concatenate(chunks)[:length]
    else:
        raise ValueError("pattern must be sequential, random, repeated, loop, or mixed")

    # Attach illustrative page and segment metadata. These are inputs from
    # companion modules in a real integration; this generator only synthesizes them.
    page_size_blocks = 16
    rows = []
    for i, block in enumerate(blocks.astype(int)):
        seg_id = int(block % 3)
        rows.append({
            "trace_id": trace_id, "step": i, "block": int(block),
            "address": int(block) * 4, "page_id": int(block // page_size_blocks),
            "page_number": int(block // page_size_blocks),
            "page_size": page_size_blocks * 4,
            "segment_id": seg_id,
            "segment_type": ["Code", "Data", "Stack"][seg_id],
            "segment_offset": int(block % page_size_blocks) * 4
        })
    return pd.DataFrame(rows)


def generate_dataset(output_path="data/synthetic_traces.csv", traces_per_pattern=12,
                     length=500, address_space=128, seed=42):
    patterns = ["sequential", "random", "repeated", "loop", "mixed"]
    frames = []
    for p_idx, pattern in enumerate(patterns):
        for j in range(traces_per_pattern):
            tid = f"{pattern}_{j:02d}"
            frames.append(generate_trace(pattern, length, address_space,
                                         seed + p_idx * 1000 + j, tid))
    data = pd.concat(frames, ignore_index=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return data


def split_traces(data, train_ratio=0.6, validation_ratio=0.2, seed=42):
    """Split by complete trace IDs to prevent adjacent-row leakage."""
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1 or train_ratio + validation_ratio >= 1:
        raise ValueError("Ratios must be positive and sum to less than 1")
    ids = np.array(sorted(data["trace_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_train = max(1, int(len(ids) * train_ratio))
    n_val = max(1, int(len(ids) * validation_ratio))
    if n_train + n_val >= len(ids):
        raise ValueError("Need at least 3 distinct traces for train/validation/test")
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train+n_val])
    test_ids = set(ids[n_train+n_val:])
    return (
        data[data.trace_id.isin(train_ids)].copy(),
        data[data.trace_id.isin(val_ids)].copy(),
        data[data.trace_id.isin(test_ids)].copy()
    )
