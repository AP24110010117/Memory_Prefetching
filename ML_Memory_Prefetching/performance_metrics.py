"""Metric calculations and fair configuration evaluation."""
import pandas as pd
from cache_simulator import CacheSimulator
from prefetcher import AdaptivePrefetcher


def simulate_trace(trace, cache_size=8, block_size=4, policy="LRU",
                   mode="normal", predictor=None, confidence_threshold=0.55,
                   memory_latency=100.0, cache_latency=1.0):
    cache = CacheSimulator(cache_size, block_size, policy)
    # Shadow baseline counts misses without prefetching, enabling a standard
    # coverage denominator even when prefetching reduces actual demand misses.
    baseline_cache = CacheSimulator(cache_size, block_size, policy)
    prefetcher = AdaptivePrefetcher(predictor, confidence_threshold)
    history = []
    prediction_correct = 0
    prediction_count = 0
    timely_prefetches = 0
    for _, row in trace.sort_values("step").iterrows():
        block = int(row["block"])
        metadata = {
            "page_id": int(row.get("page_id", 0)),
            "page_number": int(row.get("page_number", row.get("page_id", 0))),
            "page_size": int(row.get("page_size", 0)),
            "segment_id": int(row.get("segment_id", 0)),
            "segment_type": row.get("segment_type", "Unknown"),
            "segment_offset": int(row.get("segment_offset", 0))
        }

        # Make a prediction BEFORE consuming the current demand block.
        # The current block is therefore never included in its own features.
        if mode == "basic" and history:
            candidate = prefetcher.basic_candidate(history[-1])
            if candidate != block:
                prefetcher.request(candidate, 1.0, cache, metadata)
        elif mode == "ml" and predictor is not None and len(history) >= 3:
            candidate, confidence = prefetcher.ml_candidate(history, metadata)
            prediction_count += 1
            prediction_correct += int(candidate == block)
            decision = prefetcher.request(candidate, confidence, cache, metadata)
            if candidate == block and decision.accepted:
                timely_prefetches += 1

        cache.access_block(block, metadata, memory_latency, cache_latency)
        baseline_cache.access_block(block, metadata, memory_latency, cache_latency)
        history.append(block)

    metrics = cache.finalize()
    baseline_misses = baseline_cache.misses
    metrics["prefetch_coverage"] = (
        cache.useful_prefetches / baseline_misses if baseline_misses else 0.0
    )
    metrics.update({
        "mode": mode, "policy": policy,
        "prefetch_requests_attempted": prefetcher.requests,
        "prefetch_accepted": prefetcher.accepted,
        "prefetch_rejected": prefetcher.rejected,
        "prediction_accuracy_on_steps": prediction_correct / prediction_count if prediction_count else 0.0,
        "prefetch_timeliness": timely_prefetches / prefetcher.accepted if prefetcher.accepted else 0.0,
        "cache_contents": cache.snapshot()
    })
    return metrics


def compare_configurations(test_trace, cache_size=8, block_size=4,
                           policy="LRU", predictor=None, confidence_threshold=0.55):
    results = []
    for mode in ["normal", "basic", "ml"]:
        result = simulate_trace(
            test_trace, cache_size, block_size, policy, mode,
            predictor if mode == "ml" else None, confidence_threshold
        )
        results.append(result)
    return pd.DataFrame(results)


def metric_definitions():
    return {
        "Hit ratio": "Cache hits / total demand accesses",
        "Miss ratio": "Cache misses / total demand accesses",
        "Prefetch accuracy": "Useful prefetched blocks / blocks inserted by prefetcher",
        "Prefetch coverage": "Useful prefetched blocks / demand cache misses",
        "Cache pollution": "Unused prefetched blocks evicted before use / prefetch insertions",
        "Prefetch timeliness": "Correct predictions available before demand / accepted prefetches",
        "Average memory access time": "Total demand access latency units / demand accesses"
    }
