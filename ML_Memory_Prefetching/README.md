# ML-Based Memory Prefetching System – Cache Memory Module

A modular B.Tech project implementation using Python, scikit-learn Random Forest,
NumPy, Pandas, Matplotlib and Streamlit.

## 1. Architecture

- `memory_trace_generator.py`: creates synthetic sequential, random, repeated,
  loop and mixed traces; splits complete trace IDs into train/validation/test.
- `ml_predictor.py`: builds next-block training examples using only past accesses
  and trains/evaluates a Random Forest classifier.
- `cache_simulator.py`: byte-address to block conversion, FIFO/LRU cache,
  demand hits/misses, prefetch usefulness and pollution accounting.
- `prefetcher.py`: sequential baseline and confidence-gated ML prefetch requests.
- `performance_metrics.py`: identical-trace simulation and metric calculations.
- `app.py`: interactive Streamlit dashboard.

## 2. macOS / VS Code installation

Install Python 3.10 or newer. Open this folder in VS Code, then open Terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

In VS Code, select the interpreter at `.venv/bin/python`.

Generate the bundled dataset (optional; the app can generate it if missing):

```bash
python -c "from memory_trace_generator import generate_dataset; generate_dataset()"
```

Run the dashboard:

```bash
streamlit run app.py
```

The browser opens locally. Stop with `Ctrl+C`.

## 3. Example input

A block sequence:

```text
0,1,2,3,4,5,2,3,4,5,6,7,8,9,4,5,6,7
```

Blocks are memory-block identifiers, not byte addresses. The simulator also
supports byte-address conversion with `CacheSimulator.address_to_block(address)`.

## 4. CSV schema for imported traces

Required columns: `trace_id`, `step`, `block`.
Optional companion-module metadata: `address`, `page_id`, `page_number`,
`page_size`, `segment_id`, `segment_type`, `segment_offset`.

Each trace must have a unique `trace_id`; `step` is its chronological order.
Do not randomly mix rows from the same trace across training and test splits.
The `split_traces()` function splits by complete trace ID.

## 5. Page and segment integration

The cache does not implement virtual memory or segment translation. It accepts
metadata supplied by companion modules:

```python
metadata = {
    "page_id": page_record.page_id,
    "page_number": page_record.page_number,
    "page_size": page_record.page_size,
    "segment_id": segment_record.segment_id,
    "segment_type": segment_record.segment_type,  # Code/Data/Stack
    "segment_offset": segment_record.offset
}
cache.access_block(physical_address // block_size, metadata=metadata)
```

Agree with teammates whether `block` means physical block number or whether the
cache receives physical byte addresses. This implementation's `block` column
means block number; `address` means byte address. Page/segment metadata is
preserved as metadata and can be used by the predictor.

## 6. Metric formulas

- Hit ratio = hits / demand accesses.
- Miss ratio = misses / demand accesses.
- Prefetch accuracy = useful prefetched blocks / accepted prefetch insertions.
- Prefetch coverage = useful prefetched blocks / misses in a shadow no-prefetch baseline using the same trace, cache size and replacement policy.
- Cache pollution = unused prefetched blocks evicted before use / accepted prefetch insertions.
- Average memory access time = total demand latency / demand accesses.
- Prefetch timeliness = correct predictions that were inserted before demand / accepted prefetches.

A prefetched line is useful if a later demand access hits it. Unused prefetched
lines still resident at simulation end are included in `unused_prefetches`, but
not in the eviction-based `cache_pollution` numerator.

## 7. Evaluation notes and limitations

- The ML target is the next block in the trace; features contain only the
  preceding three or fewer historical accesses plus metadata of the last
  observed access.
- Trace-level splitting reduces leakage from adjacent windows. Synthetic
  patterns may still be easier than real applications; report this limitation.
- Random Forest classification can only predict block IDs represented in its
  training classes. Unseen block IDs cannot be emitted by the classifier.
- Basic prefetching predicts current block + 1 and is a simple sequential
  baseline, not an oracle.
- Latency units are an explicit cost model (default cache=1, memory=100), not
  measured processor timing.
- The current cache is fully associative. Set associativity, write policies,
  multi-level caches, coherence and realistic DRAM queues are outside this
  module's scope.
- Prediction accuracy is evaluated separately from cache benefit. A correct
  prediction may still arrive too late or be evicted before use.

## 8. Reproducibility

Synthetic dataset generation uses fixed seeds. To regenerate:

```bash
python -c "from memory_trace_generator import generate_dataset; generate_dataset('data/synthetic_traces.csv', traces_per_pattern=12, length=500, seed=42)"
```
