"""Transparent visual risk estimate; this is not a physical depth measurement."""
from __future__ import annotations

SEVERITY_CONFIG = {"low_max": 0.38, "medium_max": 0.67}


def score_severity(box_area: float, frame_area: float, confidence: float,
                   bottom_y: float, frame_height: float) -> tuple[float, str]:
    # Area saturates at 8% of the frame. A lower position is treated as closer.
    area_score = min(1.0, box_area / max(frame_area * 0.08, 1.0))
    proximity = min(1.0, max(0.0, (bottom_y / max(frame_height, 1)) - 0.35) / 0.55)
    score = 0.50 * area_score + 0.25 * min(1.0, max(0.0, confidence)) + 0.25 * proximity
    level = "LOW" if score < SEVERITY_CONFIG["low_max"] else (
        "MEDIUM" if score < SEVERITY_CONFIG["medium_max"] else "HIGH")
    return round(score, 3), level
