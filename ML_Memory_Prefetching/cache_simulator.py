"""Set-associative-ready (currently fully associative) cache simulator.

Addresses are byte addresses. They are converted to memory block numbers using
block_size. Cache capacity is measured in cache lines (cache_size_blocks).
"""
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class CacheLine:
    block: int
    prefetched: bool = False
    useful: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class CacheSimulator:
    def __init__(self, cache_size_blocks: int = 8, block_size: int = 4,
                 policy: str = "LRU"):
        if cache_size_blocks < 1:
            raise ValueError("cache_size_blocks must be at least 1")
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        policy = policy.upper()
        if policy not in {"FIFO", "LRU"}:
            raise ValueError("policy must be FIFO or LRU")
        self.capacity = int(cache_size_blocks)
        self.block_size = int(block_size)
        self.policy = policy
        self.lines: OrderedDict[int, CacheLine] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.prefetch_evictions_unused = 0
        self.prefetch_insertions = 0
        self.useful_prefetches = 0
        self.unused_prefetches = 0
        self.demand_accesses = 0
        self.demand_latency_units = 0.0
        self.prefetch_buffer = set()

    def address_to_block(self, address: int) -> int:
        if int(address) < 0:
            raise ValueError("Memory addresses must be non-negative")
        return int(address) // self.block_size

    def _touch(self, block: int):
        if self.policy == "LRU":
            self.lines.move_to_end(block)

    def _insert(self, block: int, prefetched: bool = False, metadata=None):
        if block in self.lines:
            return False
        if len(self.lines) >= self.capacity:
            old_block, old_line = self.lines.popitem(last=False)
            self.evictions += 1
            if old_line.prefetched and not old_line.useful:
                self.unused_prefetches += 1
                self.prefetch_evictions_unused += 1
        self.lines[block] = CacheLine(
            block=block, prefetched=prefetched, useful=False,
            metadata=dict(metadata or {})
        )
        if prefetched:
            self.prefetch_insertions += 1
        return True

    def access_block(self, block: int, metadata=None, memory_latency: float = 100.0,
                     cache_latency: float = 1.0) -> bool:
        """Perform one demand access. Returns True for hit, False for miss."""
        block = int(block)
        if block < 0:
            raise ValueError("Block numbers must be non-negative")
        self.demand_accesses += 1
        if block in self.lines:
            self.hits += 1
            line = self.lines[block]
            if line.prefetched and not line.useful:
                line.useful = True
                self.useful_prefetches += 1
            self._touch(block)
            self.demand_latency_units += cache_latency
            return True
        self.misses += 1
        self.demand_latency_units += memory_latency
        self._insert(block, prefetched=False, metadata=metadata)
        return False

    def prefetch_block(self, block: int, metadata=None) -> bool:
        """Insert a predicted block if absent. Returns whether inserted."""
        block = int(block)
        if block < 0 or block in self.lines or block in self.prefetch_buffer:
            return False
        return self._insert(block, prefetched=True, metadata=metadata)

    def snapshot(self):
        return [
            {"block": b, "prefetched": line.prefetched, "useful": line.useful,
             **line.metadata}
            for b, line in self.lines.items()
        ]

    def finalize(self):
        # Count prefetched lines still resident but never consumed as unused.
        resident_unused = sum(
            1 for line in self.lines.values()
            if line.prefetched and not line.useful
        )
        self.unused_prefetches = self.prefetch_evictions_unused + resident_unused
        return self.metrics()

    def metrics(self):
        total = self.hits + self.misses
        return {
            "total_accesses": total,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hits / total if total else 0.0,
            "miss_ratio": self.misses / total if total else 0.0,
            "evictions": self.evictions,
            "prefetch_requests": self.prefetch_insertions,
            "useful_prefetches": self.useful_prefetches,
            "unused_prefetches": self.unused_prefetches,
            "prefetch_accuracy": self.useful_prefetches / self.prefetch_insertions
                if self.prefetch_insertions else 0.0,
            "prefetch_coverage": self.useful_prefetches / self.misses
                if self.misses else 0.0,
            "cache_pollution": self.prefetch_evictions_unused / self.prefetch_insertions
                if self.prefetch_insertions else 0.0,
            "average_memory_access_time": self.demand_latency_units / total if total else 0.0,
        }
