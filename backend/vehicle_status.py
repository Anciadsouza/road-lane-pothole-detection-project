"""Video-level condition indicators that avoid inventing physical telemetry."""
from __future__ import annotations

from statistics import median


LIGHTING_NIGHT_THRESHOLD = 58.0


def classify_lighting(luminance_samples: list[float]) -> tuple[str, float | None]:
    """Classify the dominant scene lighting from sampled frame luminance."""
    if not luminance_samples:
        return "UNKNOWN", None
    value = float(median(luminance_samples))
    return ("NIGHT" if value < LIGHTING_NIGHT_THRESHOLD else "DAY"), round(value, 1)


def vehicle_speed_status() -> dict:
    """Return unavailable speed; the project has no scale or vehicle telemetry."""
    return {
        "speed_kmh": None,
        "speed_label": "N/A",
        "speed_source": "unavailable",
        "speed_note": "No GPS, CAN-bus speed, calibrated scene scale, or validated visual odometry is available.",
        "driving_direction": "Forward",
    }
