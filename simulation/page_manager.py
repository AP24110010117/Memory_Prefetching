"""
page_manager.py — Page and Segment Boundary Validation Manager.

Manages:
1. Virtual page boundary safety (4KB page limits).
2. Segment boundary safety (TEXT, DATA, HEAP, STACK limits).
3. Pre-issue validation to prevent speculative page faults and memory corruption.
4. Comprehensive boundary test suite for verification.
"""

import sys, os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def is_valid_page_prefetch(block: int, delta: int,
                           page_offset: int, dist_to_page_end: int) -> bool:
    """
    Returns True if prefetching (block + delta) stays strictly within the current 4KB page.
    Cross-page prefetches without OS-level address translation are unsafe.
    """
    if delta > dist_to_page_end:
        return False  # Crosses page boundary forward
    if delta < -page_offset:
        return False  # Crosses page boundary backward
    return True


def is_valid_segment_prefetch(block: int, delta: int, segment_name: str) -> bool:
    """
    Returns True if prefetching (block + delta) stays strictly within the allocated memory segment.
    """
    target_byte = (block + delta) << 6
    if segment_name not in config.SEGMENTS:
        return True  # Unknown segment; default pass

    seg = config.SEGMENTS[segment_name]
    base = seg["base"]
    limit = base + (seg["size_kb"] * 1024)

    if segment_name == "STACK":
        # Downward-growing stack: base is highest address, grows downward to (base - size)
        stack_bottom = base - (seg["size_kb"] * 1024)
        if target_byte < stack_bottom or target_byte > base:
            return False
    else:
        # Normal segments: grow upward from base to base + size
        if target_byte < base or target_byte >= limit:
            return False

    return True


def validate_prefetch_candidate(block: int, delta: int,
                                page_offset: int, dist_to_page_end: int,
                                segment_name: str) -> tuple:
    """
    Validates candidate prefetch address.
    Returns (is_valid: bool, reason: str).
    """
    if not is_valid_page_prefetch(block, delta, page_offset, dist_to_page_end):
        return False, "BLOCKED_PAGE_BOUNDARY"

    if not is_valid_segment_prefetch(block, delta, segment_name):
        return False, "BLOCKED_SEGMENT_BOUNDARY"

    return True, "SAFE"


def run_boundary_test_suite() -> pd.DataFrame:
    """
    Comprehensive boundary test suite verifying DATA, HEAP, and STACK segments
    under within-bounds, page-crossing, and segment-overflow conditions.
    """
    test_cases = [
        # TEXT Segment
        {"name": "TEXT: Sequential Safe",       "seg": "TEXT",  "delta": 1,   "p_off": 10, "dist": 53, "blk_off": 100,  "expect": "SAFE"},
        {"name": "TEXT: Forward Page Cross",    "seg": "TEXT",  "delta": 4,   "p_off": 62, "dist": 1,  "blk_off": 100,  "expect": "BLOCKED_PAGE_BOUNDARY"},
        {"name": "TEXT: Backward Page Cross",   "seg": "TEXT",  "delta": -4,  "p_off": 2,  "dist": 61, "blk_off": 100,  "expect": "BLOCKED_PAGE_BOUNDARY"},
        # DATA Segment
        {"name": "DATA: Strided Safe",          "seg": "DATA",  "delta": 4,   "p_off": 20, "dist": 43, "blk_off": 500,  "expect": "SAFE"},
        {"name": "DATA: Large Stride Cross",    "seg": "DATA",  "delta": 16,  "p_off": 55, "dist": 8,  "blk_off": 500,  "expect": "BLOCKED_PAGE_BOUNDARY"},
        {"name": "DATA: Segment Overflow",      "seg": "DATA",  "delta": 16,  "p_off": 10, "dist": 53, "blk_off": 16380,"expect": "BLOCKED_SEGMENT_BOUNDARY"},
        # HEAP Segment
        {"name": "HEAP: Pointer Step Safe",     "seg": "HEAP",  "delta": 2,   "p_off": 30, "dist": 33, "blk_off": 1000, "expect": "SAFE"},
        {"name": "HEAP: Page Boundary Hazard",  "seg": "HEAP",  "delta": 8,   "p_off": 60, "dist": 3,  "blk_off": 1000, "expect": "BLOCKED_PAGE_BOUNDARY"},
        {"name": "HEAP: Heap Limit Exceeded",   "seg": "HEAP",  "delta": 16,  "p_off": 20, "dist": 43, "blk_off": 32760,"expect": "BLOCKED_SEGMENT_BOUNDARY"},
        # STACK Segment
        {"name": "STACK: Reverse Safe",         "seg": "STACK", "delta": -1,  "p_off": 25, "dist": 38, "blk_off": -100, "expect": "SAFE"},
        {"name": "STACK: Backward Page Cross",  "seg": "STACK", "delta": -2,  "p_off": 1,  "dist": 62, "blk_off": -100, "expect": "BLOCKED_PAGE_BOUNDARY"},
        {"name": "STACK: Stack Overflow",       "seg": "STACK", "delta": -16, "p_off": 30, "dist": 33, "blk_off": -8185,"expect": "BLOCKED_SEGMENT_BOUNDARY"},
    ]

    results = []
    for tc in test_cases:
        seg = tc["seg"]
        base_blk = config.SEGMENTS[seg]["base"] >> 6
        curr_blk = base_blk + tc["blk_off"]
        ok, reason = validate_prefetch_candidate(
            block=curr_blk,
            delta=tc["delta"],
            page_offset=tc["p_off"],
            dist_to_page_end=tc["dist"],
            segment_name=seg
        )
        passed = (reason == tc["expect"])
        results.append({
            "test_case":        tc["name"],
            "segment":          tc["seg"],
            "stride_delta":     tc["delta"],
            "page_offset":      tc["p_off"],
            "dist_to_page_end": tc["dist"],
            "expected":         tc["expect"],
            "actual_decision":  reason,
            "status":           "PASSED" if passed else "FAILED"
        })

    return pd.DataFrame(results)
