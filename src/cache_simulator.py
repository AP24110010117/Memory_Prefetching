"""
cache_simulator.py — Multi-Level Cache Hierarchy Simulator with Correct Prefetch Metrics.

Models:
  L1  :  32 KB, 8-way set-associative, 64B lines, 4-cycle hit latency
  L2  : 256 KB, 8-way set-associative, 64B lines, 14-cycle hit latency
  L3  :   2 MB,16-way set-associative, 64B lines, 40-cycle hit latency
  DRAM: 200-cycle access latency

Metrics adhere strictly to standard computer architecture definitions:
  - Coverage: Useful Prefetches / Baseline Demand Misses <= 100%
  - Useful: Timely + Late prefetches that directly eliminate a demand miss
  - Late: Demanded while in-flight (partially mitigates miss penalty)
  - Unused / Pollution: Evicted before being referenced by CPU demand
  - Redundant: Line already cached or in-flight (suppressed)
"""

import collections
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CacheSet:
    """Single set in a set-associative cache (LRU replacement)."""

    def __init__(self, num_ways: int):
        self.num_ways = num_ways
        self.lines: collections.OrderedDict = collections.OrderedDict()

    def probe(self, block: int) -> bool:
        return block in self.lines

    def access(self, block: int, cycle: int):
        """CPU demand access. Returns (hit: bool, meta: dict|None)."""
        if block in self.lines:
            meta = self.lines.pop(block)
            meta["referenced"] += 1
            meta["last_cycle"] = cycle
            self.lines[block] = meta
            return True, meta
        return False, None

    def insert(self, block: int, is_prefetch: bool = False, cycle: int = 0):
        """Insert line; returns evicted metadata or None."""
        evicted = None
        if block in self.lines:
            meta = self.lines.pop(block)
            meta["last_cycle"] = cycle
            self.lines[block] = meta
            return None
        if len(self.lines) >= self.num_ways:
            _, evicted = self.lines.popitem(last=False)
        self.lines[block] = {
            "is_prefetch": is_prefetch,
            "referenced": 0 if is_prefetch else 1,
            "install_cycle": cycle,
            "last_cycle": cycle,
            "consumed": False,
        }
        return evicted


class CacheLevel:
    """One level of the cache hierarchy."""

    def __init__(self, name: str, capacity_kb: int, num_ways: int,
                 line_bytes: int = 64, latency: int = 4):
        self.name = name
        self.line_bytes = line_bytes
        self.num_ways = num_ways
        self.num_lines = (capacity_kb * 1024) // line_bytes
        self.num_sets = self.num_lines // num_ways
        self.latency = latency
        self.sets = [CacheSet(num_ways) for _ in range(self.num_sets)]
        self.demand_accesses = self.hits = self.misses = 0

    def set_idx(self, block: int) -> int:
        return block % self.num_sets

    def probe(self, block: int) -> bool:
        return self.sets[self.set_idx(block)].probe(block)

    def get_set_idx(self, block: int) -> int:
        return self.set_idx(block)

    def occupancy(self) -> float:
        total_ways = sum(s.num_ways for s in self.sets)
        used_ways  = sum(len(s.lines) for s in self.sets)
        return used_ways / max(1, total_ways)

    def access(self, block: int, cycle: int):
        self.demand_accesses += 1
        hit, meta = self.sets[self.set_idx(block)].access(block, cycle)
        if hit:
            self.hits += 1
        else:
            self.misses += 1
        return hit, meta

    def insert(self, block: int, is_prefetch: bool = False, cycle: int = 0):
        return self.sets[self.set_idx(block)].insert(block, is_prefetch, cycle)


class MultiLevelCache:
    """
    Full L1 -> L2 -> L3 -> DRAM hierarchy with prefetch tracking.
    """

    def __init__(self, pq_size: int = 16):
        cfg = config.CACHE_CONFIG
        self.l1 = CacheLevel("L1", **{
            "capacity_kb": cfg["L1"]["capacity_kb"],
            "num_ways":    cfg["L1"]["num_ways"],
            "line_bytes":  cfg["L1"]["line_bytes"],
            "latency":     cfg["L1"]["latency_cycles"],
        })
        self.l2 = CacheLevel("L2", **{
            "capacity_kb": cfg["L2"]["capacity_kb"],
            "num_ways":    cfg["L2"]["num_ways"],
            "line_bytes":  cfg["L2"]["line_bytes"],
            "latency":     cfg["L2"]["latency_cycles"],
        })
        self.l3 = CacheLevel("L3", **{
            "capacity_kb": cfg["L3"]["capacity_kb"],
            "num_ways":    cfg["L3"]["num_ways"],
            "line_bytes":  cfg["L3"]["line_bytes"],
            "latency":     cfg["L3"]["latency_cycles"],
        })
        self.dram_lat = cfg["DRAM"]["latency_cycles"]
        self.pq_size  = pq_size
        self.in_flight: dict = {}   # block -> completion_cycle

        # Access and hit counts
        self.total_accesses = 0
        self.l1_hits = self.l2_hits = self.l3_hits = self.dram_hits = 0
        self.stall_cycles = 0

        # Prefetch tracking metrics
        self.pf_requested = 0
        self.pf_issued    = 0
        self.pf_redundant = 0
        self.pf_dropped   = 0
        self.pf_timely    = 0
        self.pf_late      = 0
        self.pf_unused    = 0

    def _flush_completed(self, cycle: int):
        done = [b for b, c in self.in_flight.items() if c <= cycle]
        for b in done:
            del self.in_flight[b]

    def demand_access(self, block: int, cycle: int):
        """CPU demand access. Returns (level_str, latency)."""
        self.total_accesses += 1
        self._flush_completed(cycle)

        # 1. L1 Cache Check
        hit, meta = self.l1.access(block, cycle)
        if hit:
            self.l1_hits += 1
            self.stall_cycles += self.l1.latency
            if meta and meta["is_prefetch"] and not meta["consumed"]:
                self.pf_timely += 1
                meta["consumed"] = True
            return "L1_HIT", self.l1.latency

        # 2. Check In-Flight Prefetches (Late Prefetch Hit)
        if block in self.in_flight:
            ready = self.in_flight.pop(block)
            stall = max(0, ready - cycle) + self.l1.latency
            self.stall_cycles += stall
            self.pf_late += 1
            evicted = self.l1.insert(block, is_prefetch=True, cycle=cycle)
            self._track_eviction(evicted)
            return "LATE_PREFETCH", stall

        # 3. L2 Cache Check
        hit, meta = self.l2.access(block, cycle)
        if hit:
            self.l2_hits += 1
            stall = self.l1.latency + self.l2.latency
            self.stall_cycles += stall
            if meta and meta["is_prefetch"] and not meta["consumed"]:
                self.pf_timely += 1
                meta["consumed"] = True
            evicted = self.l1.insert(block, is_prefetch=False, cycle=cycle)
            self._track_eviction(evicted)
            return "L2_HIT", stall

        # 4. L3 Cache Check
        hit, meta = self.l3.access(block, cycle)
        if hit:
            self.l3_hits += 1
            stall = self.l1.latency + self.l2.latency + self.l3.latency
            self.stall_cycles += stall
            self.l2.insert(block, is_prefetch=False, cycle=cycle)
            evicted = self.l1.insert(block, is_prefetch=False, cycle=cycle)
            self._track_eviction(evicted)
            return "L3_HIT", stall

        # 5. DRAM Access (L1, L2, L3 Miss)
        self.dram_hits += 1
        stall = self.l1.latency + self.l2.latency + self.l3.latency + self.dram_lat
        self.stall_cycles += stall
        self.l3.insert(block, cycle=cycle)
        self.l2.insert(block, cycle=cycle)
        evicted = self.l1.insert(block, cycle=cycle)
        self._track_eviction(evicted)
        return "DRAM_MISS", stall

    def _track_eviction(self, meta):
        """Track if an installed prefetch was evicted before ever being referenced."""
        if meta and meta["is_prefetch"] and meta["referenced"] == 0:
            self.pf_unused += 1

    def issue_prefetch(self, block: int, cycle: int, target: str = "L2"):
        """Issue a prefetch with complete hardware deduplication and capacity checks."""
        self.pf_requested += 1

        # Deduplication against L1, L2, and active MSHR / in-flight requests
        if self.l1.probe(block):
            self.pf_redundant += 1
            return False, "IN_L1"
        if target == "L2" and self.l2.probe(block):
            self.pf_redundant += 1
            return False, "IN_L2"
        if block in self.in_flight:
            self.pf_redundant += 1
            return False, "IN_FLIGHT"

        # Hardware prefetch queue capacity check
        self._flush_completed(cycle)
        if len(self.in_flight) >= self.pq_size:
            self.pf_dropped += 1
            return False, "QUEUE_FULL"

        # Determine memory fill latency
        if self.l2.probe(block):
            fill = self.l2.latency
        elif self.l3.probe(block):
            fill = self.l3.latency
        else:
            fill = self.dram_lat

        self.pf_issued += 1
        self.in_flight[block] = cycle + fill

        if target == "L1":
            evicted = self.l1.insert(block, is_prefetch=True, cycle=cycle + fill)
        else:
            evicted = self.l2.insert(block, is_prefetch=True, cycle=cycle + fill)
        self._track_eviction(evicted)
        return True, f"ISSUED_{target}"

    def get_demand_misses(self) -> int:
        """Total demand misses that went to DRAM."""
        return self.dram_hits

    def metrics(self, baseline_demand_misses: int = None) -> dict:
        total = max(1, self.total_accesses)
        issued = max(1, self.pf_issued)
        useful = self.pf_timely + self.pf_late

        # Standard Coverage calculation: Useful Prefetches / Baseline Demand Misses <= 100%
        if baseline_demand_misses is not None and baseline_demand_misses > 0:
            coverage = min(100.0, (useful / baseline_demand_misses) * 100.0)
        else:
            # Fallback if no baseline provided
            coverage = min(100.0, (useful / max(1, self.dram_hits + useful)) * 100.0)

        return {
            "total_accesses":          self.total_accesses,
            "l1_hit_rate_pct":         (self.l1_hits / total) * 100.0,
            "l2_hit_rate_pct":         (self.l2_hits / total) * 100.0,
            "l3_hit_rate_pct":         (self.l3_hits / total) * 100.0,
            "overall_hit_rate_pct":    ((self.l1_hits + self.l2_hits + self.l3_hits) / total) * 100.0,
            "dram_miss_rate_pct":      (self.dram_hits / total) * 100.0,
            "amat_cycles":             self.stall_cycles / total,
            "pf_requested":            self.pf_requested,
            "pf_issued":               self.pf_issued,
            "pf_redundant":            self.pf_redundant,
            "pf_dropped":              self.pf_dropped,
            "pf_timely":               self.pf_timely,
            "pf_late":                 self.pf_late,
            "pf_unused":               self.pf_unused,
            "pf_useful":               useful,
            "coverage_pct":            coverage,
            "useful_precision_pct":    min(100.0, (useful / issued) * 100.0),
            "timely_precision_pct":    min(100.0, (self.pf_timely / issued) * 100.0),
            "pollution_rate_pct":      min(100.0, (self.pf_unused / issued) * 100.0),
            "l1_occupancy_pct":        self.l1.occupancy() * 100.0,
        }
