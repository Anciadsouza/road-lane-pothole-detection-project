"""Loads the repository's trained YOLO model for CPU inference."""
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best.pt"


def load_model() -> "YOLO":
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    model.to("cpu")
    return model


def is_pothole_class(model: "YOLO", class_id: int) -> bool:
    names = model.names
    name = names.get(class_id, "") if isinstance(names, dict) else names[class_id]
    return "pothole" in str(name).casefold()
