"""Load the trained pothole model on CUDA when available, otherwise CPU."""
import os
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best.pt"


def get_inference_device() -> str:
    import torch
    requested = os.getenv("YOLO_DEVICE", "auto").strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        requested = "cuda:0"
    if not requested.startswith("cuda:") or not requested[5:].isdigit():
        raise ValueError("YOLO_DEVICE must be auto, cpu, or cuda:<index>.")
    if not torch.cuda.is_available() or int(requested[5:]) >= torch.cuda.device_count():
        raise RuntimeError(f"Requested GPU {requested} is unavailable. Check CUDA PyTorch and the NVIDIA driver.")
    return requested


def inference_runtime(device: str | None = None) -> dict:
    import torch
    device = device or get_inference_device()
    return {"inference_device": device,
            "inference_device_name": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else "CPU",
            "cuda_available": torch.cuda.is_available(), "torch_version": torch.__version__}


def load_model() -> "YOLO":
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    model.to(get_inference_device())
    return model


def is_pothole_class(model: "YOLO", class_id: int) -> bool:
    names = model.names
    name = names.get(class_id, "") if isinstance(names, dict) else names[class_id]
    return "pothole" in str(name).casefold()
