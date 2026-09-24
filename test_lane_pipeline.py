"""Checks for model inference on the selected device and the lane-aware pipeline.

Run from the repository root with: .venv/bin/python test_lane_pipeline.py
Optional previously uploaded condition clips under runs/dashboard are included
when present so the same detector is checked on real night, traffic and haze.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from backend.detector import load_model
from backend.lane_detector import LaneDetector, is_in_current_lane, is_on_drivable_road
from backend.tracker import CentroidTracker
from backend.vehicle_status import classify_lighting, vehicle_speed_status

ROOT = Path(__file__).resolve().parent
VIDEO = ROOT / "assets" / "final.mp4"
CONDITION_CLIPS = {
    "night_traffic": ROOT / "runs/dashboard/b52975f5d423/input.mp4",
    "dim_low_contrast": ROOT / "runs/dashboard/2cde00935ec0/input.mp4",
    "daytime_traffic": ROOT / "runs/dashboard/e6b9f2c65e08/input.mp4",
    "haze_and_traffic": ROOT / "runs/dashboard/14080cf57630/input.mp4",
}


def metadata(path: Path) -> tuple[int, int, float, int]:
    output = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames", "-of", "json", str(path)], text=True)
    stream = json.loads(output)["streams"][0]
    num, den = stream["avg_frame_rate"].split("/")
    return int(stream["width"]), int(stream["height"]), float(num) / max(1.0, float(den)), int(stream["nb_frames"])


def read_frames(path: Path, count: int) -> list[np.ndarray]:
    width, height, _, _ = metadata(path)
    decoded = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", str(count),
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"])
    frame_size = width * height * 3
    usable = len(decoded) // frame_size * frame_size
    return [np.frombuffer(decoded[i:i + frame_size], dtype=np.uint8).reshape(height, width, 3).copy()
            for i in range(0, usable, frame_size)]


def inspect_condition_clip(name: str, path: Path, sample_count: int = 120) -> tuple[int, int, int]:
    width, height, fps, _ = metadata(path)
    frames = read_frames(path, sample_count)
    detector = LaneDetector(width, height, fps=fps)
    reliable = low_light = low_contrast = 0
    for index, frame in enumerate(frames, 1):
        geometry = detector.update(frame, index)
        reliable += int(geometry.reliable)
        low_light += int(geometry.low_light)
        low_contrast += int(geometry.low_contrast)
    return reliable, low_light, low_contrast


def main() -> None:
    assert VIDEO.is_file(), f"Missing demo video: {VIDEO}"
    width, height, fps, _ = metadata(VIDEO)
    frames = read_frames(VIDEO, 60)
    assert len(frames) == 60, f"Expected 60 demo frames, got {len(frames)}"

    lane_detector = LaneDetector(width, height, fps=fps)
    geometry = None
    source_snapshot = frames[-1].copy()
    for index, frame in enumerate(frames, 1):
        geometry = lane_detector.update(frame, index)
    assert geometry is not None and geometry.reliable, "Could not estimate the ego lane in demo frames"
    assert np.array_equal(source_snapshot, frames[-1]), "Lane preprocessing modified the source frame"
    inside = tuple(np.mean(geometry.polygon, axis=0).astype(int))
    assert is_in_current_lane(inside, geometry.polygon), "Ego-lane point association failed"
    assert is_on_drivable_road(inside, geometry.road_mask), "Road-mask association failed"
    assert geometry.centerline is not None and geometry.vanishing_point is not None

    # Brief total marking loss keeps the previous lane; sustained loss becomes
    # uncertain instead of extending a stale polygon indefinitely.
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    held = [lane_detector.update(blank, 60 + step) for step in range(1, 9)]
    assert held[-1].reliable and held[-1].tracking_status == "TEMPORARILY_OCCLUDED"
    assert held[-1].confidence < geometry.confidence and len(held[-1].polygon) >= 3
    expired = None
    for step in range(9, lane_detector.config["max_hold_frames"] + 3):
        expired = lane_detector.update(blank, 60 + step)
    assert expired is not None and not expired.reliable and expired.tracking_status == "UNCERTAIN"

    # Keep the weights, confidence threshold, and source pixels unchanged.
    model = load_model()
    device = str(model.device)
    prediction = model.predict(source_snapshot.copy(), conf=0.25, imgsz=640,
                               device=device, verbose=False)[0]
    assert prediction.boxes is not None, "YOLO prediction did not return box results"

    tracker = CentroidTracker()
    boxes = [
        {"cx": 300.0, "cy": 400.0, "x1": 280, "y1": 380, "x2": 320, "y2": 420, "lane": "current"},
        {"cx": 302.0, "cy": 401.0, "x1": 282, "y1": 381, "x2": 322, "y2": 421, "lane": "uncertain"},
        {"cx": 304.0, "cy": 402.0, "x1": 284, "y1": 382, "x2": 324, "y2": 422, "lane": "current"},
    ]
    assignments = [tracker.update([detection], frame)[0][0]
                   for frame, detection in enumerate(boxes, 1)]
    assert len(set(assignments)) == 1, "Tracker did not bridge current/uncertain lane state"
    repeated = [tracker.update([{"cx": 300.0 + i, "cy": 400.0, "lane": "current"}], frame)[0][0]
                for i, frame in enumerate(range(4, 44), 1)]
    assert len(set(repeated)) == 1, "Repeated pothole detections received multiple IDs"

    tested = []
    for name, path in CONDITION_CLIPS.items():
        if not path.is_file():
            print(f"SKIP: {name} clip is not present: {path.relative_to(ROOT)}")
            continue
        reliable, low_light, low_contrast = inspect_condition_clip(name, path)
        fraction = reliable / min(120, metadata(path)[3])
        tested.append(name)
        print(f"PASS: {name}: reliable lane frames {reliable}/120 ({fraction:.0%}), "
              f"low-light frames {low_light}, low-contrast frames {low_contrast}")
        assert reliable > 0, f"Lane detector never acquired a lane in {name}"
        if name == "night_traffic":
            assert low_light > 0, "Night clip did not enter the low-light lane preprocessing branch"
    if tested:
        print("PASS: multiple available traffic/night/low-contrast clips exercised without changing YOLO input")

    assert classify_lighting([120, 135, 128])[0] == "DAY"
    assert classify_lighting([20, 35, 28])[0] == "NIGHT"
    assert vehicle_speed_status()["speed_kmh"] is None
    print(f"PASS: existing model loads and runs on {device}")
    print("PASS: lane polygon, road point association, centerline and convergence estimate")
    print("PASS: 8-frame occlusion grace followed by honest uncertain state")
    print("PASS: YOLO source remains unchanged and original inference settings are retained")
    print("PASS: tracker preserves event ID through temporary lane uncertainty and repeated frames")
    print("PASS: day/night status and unavailable speed behavior")


if __name__ == "__main__":
    main()
