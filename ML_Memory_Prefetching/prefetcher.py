"""Basic and Random-Forest adaptive prefetch policies."""
from dataclasses import dataclass


@dataclass
class PrefetchDecision:
    block: int
    confidence: float
    accepted: bool
    reason: str


class AdaptivePrefetcher:
    def __init__(self, predictor=None, confidence_threshold=0.55,
                 max_prefetch_distance=1):
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.predictor = predictor
        self.confidence_threshold = float(confidence_threshold)
        self.max_prefetch_distance = max_prefetch_distance
        self.requests = 0
        self.accepted = 0
        self.rejected = 0
        self.decisions = []

    def basic_candidate(self, current_block):
        return int(current_block) + 1

    def ml_candidate(self, history, metadata=None):
        if self.predictor is None:
            raise RuntimeError("ML predictor is not configured")
        return self.predictor.predict_next(history, metadata)

    def request(self, block, confidence, cache, metadata=None):
        self.requests += 1
        if confidence < self.confidence_threshold:
            self.rejected += 1
            decision = PrefetchDecision(block, confidence, False, "below confidence threshold")
        elif block in cache.lines:
            self.rejected += 1
            decision = PrefetchDecision(block, confidence, False, "already in cache")
        else:
            inserted = cache.prefetch_block(block, metadata)
            if inserted:
                self.accepted += 1
                decision = PrefetchDecision(block, confidence, True, "inserted")
            else:
                self.rejected += 1
                decision = PrefetchDecision(block, confidence, False, "duplicate or unavailable")
        self.decisions.append(decision)
        return decision
