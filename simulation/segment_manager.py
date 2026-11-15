"""
Segment management utilities.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_segment_info(block: int) -> dict:
    """Return segment metadata for the given block address."""
    byte_addr = block << 6
    for seg_name, seg in config.SEGMENTS.items():
        base = seg["base"]
        end  = base + seg["size_kb"] * 1024
        if base <= byte_addr < end:
            seg_base_block = base >> 6
            seg_end_block  = end  >> 6
            return {
                "name":          seg_name,
                "id":            seg["id"],
                "dist_to_end":   max(0, seg_end_block - block),
                "valid":         True,
            }
    return {"name": "UNKNOWN", "id": -1, "dist_to_end": 0, "valid": False}
