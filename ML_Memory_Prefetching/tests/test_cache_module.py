import unittest
import pandas as pd
from cache_simulator import CacheSimulator
from memory_trace_generator import generate_trace, split_traces
from ml_predictor import RandomForestPredictor
from performance_metrics import simulate_trace


class CacheModuleTests(unittest.TestCase):
    def test_address_conversion(self):
        c = CacheSimulator(2, 4, "FIFO")
        self.assertEqual(c.address_to_block(15), 3)

    def test_fifo_and_lru(self):
        # FIFO evicts block 1; LRU evicts block 2 after touching block 1.
        fifo = CacheSimulator(2, 4, "FIFO")
        lru = CacheSimulator(2, 4, "LRU")
        for b in [1, 2]:
            fifo.access_block(b); lru.access_block(b)
        fifo.access_block(1); lru.access_block(1)
        fifo.access_block(3); lru.access_block(3)
        self.assertNotIn(1, fifo.lines)
        self.assertIn(1, lru.lines)

    def test_ml_training_and_simulation(self):
        traces = pd.concat([
            generate_trace("sequential", 100, 32, 1, "a"),
            generate_trace("loop", 100, 32, 2, "b"),
            generate_trace("mixed", 100, 32, 3, "c"),
            generate_trace("random", 100, 32, 4, "d"),
            generate_trace("repeated", 100, 32, 5, "e"),
        ], ignore_index=True)
        train, val, test = split_traces(traces, seed=4)
        model = RandomForestPredictor(n_estimators=20).fit(train)
        result = simulate_trace(test[test.trace_id == test.trace_id.iloc[0]],
                                mode="ml", predictor=model)
        self.assertEqual(result["total_accesses"], 100)
        self.assertGreaterEqual(result["hit_ratio"], 0)
        self.assertLessEqual(result["hit_ratio"], 1)
        self.assertLessEqual(result["prefetch_timeliness"], 1)


if __name__ == "__main__":
    unittest.main()
