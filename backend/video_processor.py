"""Frame-by-frame YOLO processing with FFmpeg video I/O and OpenCV overlays."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .detector import inference_runtime, is_pothole_class, load_model
from .forward_alerts import ForwardAlertTimeline
from .lane_detector import (LaneDetector, draw_lane_overlay, is_in_current_lane,
                             is_on_drivable_road)
from .severity import score_severity
from .tracker import CentroidTracker
from .vehicle_status import classify_lighting, vehicle_speed_status


def _probe(path: Path) -> tuple[int, int, float, int]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("FFprobe is required for video processing. Install FFmpeg and retry.")
    result = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames", "-of", "json", str(path)],
        check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    numerator, denominator = stream.get("avg_frame_rate", "24/1").split("/")
    fps = float(numerator) / max(float(denominator), 1)
    raw_count = stream.get("nb_frames")
    count = int(raw_count) if raw_count and str(raw_count).isdigit() else 0
    return width, height, fps, count


def process_video(source: Path, output: Path, progress: Callable[[int, str], None] | None = None) -> dict:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required for video processing. Install FFmpeg and retry.")
    width, height, fps, frame_count = _probe(source)
    model = load_model()
    device = str(model.device)
    lane_detector = LaneDetector(width, height, fps=fps)
    tracker = CentroidTracker()
    forward_alerts = ForwardAlertTimeline(fps, height)
    output.parent.mkdir(parents=True, exist_ok=True)
    decoder = subprocess.Popen([ffmpeg, "-v", "error", "-i", str(source), "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-fps_mode", "passthrough", "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encoder = subprocess.Popen([ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    detections: list[dict] = []
    ignored_records: list[dict] = []
    current_event_ids: dict[int, int] = {}
    frame_index = 0
    raw_count = 0
    ignored_unique: set[int] = set()
    outside_unique: set[int] = set()
    uncertain_unique: set[int] = set()
    confidences: list[float] = []
    lane_confidences: list[float] = []
    lane_status_counts = {"TRACKED": 0, "TEMPORARILY_OCCLUDED": 0, "LOW_CONFIDENCE": 0, "UNCERTAIN": 0}
    last_lane_geometry = None
    lighting_samples: list[float] = []
    frame_bytes = width * height * 3
    try:
        while True:
            raw = decoder.stdout.read(frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise RuntimeError("The input video ended unexpectedly while decoding.")
            frame_index += 1
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
            if frame_index == 1 or frame_index % max(1, round(fps)) == 0:
                thumbnail = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
                luminance_samples = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY)
                lighting_samples.append(float(np.mean(luminance_samples)))
            lane_geometry = lane_detector.update(frame, frame_index)
            last_lane_geometry = lane_geometry
            polygon = lane_geometry.polygon
            lane_confidences.append(lane_geometry.confidence)
            lane_status_counts[lane_geometry.tracking_status] += 1
            # Run unchanged YOLO inference on the decoded road frame before
            # adding any lane or box visualization overlays.
            result = model.predict(frame, conf=0.25, imgsz=640, device=device, verbose=False)[0]
            draw_lane_overlay(frame, lane_geometry)
            frame_detections: list[dict] = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    if not is_pothole_class(model, class_id):
                        continue
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    conf = float(box.conf.item())
                    cx, bottom = (x1 + x2) / 2, y2
                    if not lane_geometry.reliable:
                        lane = "uncertain"
                    elif is_in_current_lane((cx, bottom), polygon):
                        lane = "current"
                    elif is_on_drivable_road((cx, bottom), lane_geometry.road_mask):
                        lane = "adjacent"
                    else:
                        lane = "outside_road"
                    if lane == "current":
                        area = max(0, x2 - x1) * max(0, y2 - y1)
                        score, severity = score_severity(area, width * height, conf, bottom, height)
                    else:
                        score, severity = 0.0, "IGNORED"
                    frame_detections.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                        "cx": cx, "cy": bottom, "confidence": conf, "lane": lane, "score": score, "severity": severity})
            raw_count += len(frame_detections)
            associations = tracker.update(frame_detections, frame_index)
            for d, (track_id, is_new) in zip(frame_detections, associations):
                display_id = track_id
                if d["lane"] == "current" and track_id not in current_event_ids:
                    display_id = len(detections) + 1
                    current_event_ids[track_id] = display_id
                    uncertain_unique.discard(track_id)
                    ignored_unique.discard(track_id)
                    outside_unique.discard(track_id)
                    left_x = np.interp(d["cy"], lane_geometry.left_boundary[:, 1], lane_geometry.left_boundary[:, 0])
                    right_x = np.interp(d["cy"], lane_geometry.right_boundary[:, 1], lane_geometry.right_boundary[:, 0])
                    center_x = (left_x + right_x) / 2
                    position = "center" if abs(d["cx"] - center_x) < width * 0.055 else ("left" if d["cx"] < center_x else "right")
                    detections.append({"id": display_id, "frame": frame_index,
                        "confidence": round(d["confidence"], 4), "severity": d["severity"],
                        "severity_score": d["score"], "lane": "ego", "lane_id": 1,
                        "lane_confidence": round(lane_geometry.confidence, 4), "position": position,
                        "center": [round(d["cx"], 1), round((d["y1"] + d["y2"]) / 2, 1)],
                        "road_contact": [round(d["cx"], 1), round(d["cy"], 1)],
                        "width": d["x2"] - d["x1"], "height": d["y2"] - d["y1"],
                        "box": [d["x1"], d["y1"], d["x2"], d["y2"]]})
                    confidences.append(d["confidence"])
                elif d["lane"] == "current":
                    display_id = current_event_ids.get(track_id, track_id)
                elif d["lane"] == "adjacent" and track_id not in ignored_unique:
                    ignored_unique.add(track_id)
                    uncertain_unique.discard(track_id)
                    ignored_records.append({"track_id": track_id, "frame": frame_index,
                        "confidence": round(d["confidence"], 4), "lane": "adjacent",
                        "box": [d["x1"], d["y1"], d["x2"], d["y2"]]})
                elif d["lane"] == "outside_road" and track_id not in outside_unique:
                    outside_unique.add(track_id)
                    uncertain_unique.discard(track_id)
                    ignored_records.append({"track_id": track_id, "frame": frame_index,
                        "confidence": round(d["confidence"], 4), "lane": "outside_road",
                        "box": [d["x1"], d["y1"], d["x2"], d["y2"]]})
                elif d["lane"] == "uncertain" and is_new and track_id not in current_event_ids:
                    uncertain_unique.add(track_id)

                if d["lane"] == "current":
                    forward_alerts.observe(track_id, display_id, frame_index, d["cy"], d["confidence"])

                # Red is reserved for a detected pothole, irrespective of its
                # lane relevance. The short label carries its association.
                pothole_red = (42, 48, 232)  # BGR
                cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), pothole_red, 2, cv2.LINE_AA)
                lane_label = (f"POTHOLE #{display_id}" if d["lane"] == "current" else
                              f"ADJACENT #{display_id}" if d["lane"] == "adjacent" else
                              f"OUTSIDE ROAD #{display_id}" if d["lane"] == "outside_road" else
                              f"LANE UNCERTAIN #{display_id}")
                label = f"{lane_label}  {d['confidence']:.0%}"
                if d["lane"] == "current":
                    label += f"  {d['severity']}"
                y = max(24, d["y1"] - 9)
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                label_left = max(0, min(d["x1"], width - text_width - 10))
                label_top = max(0, y - text_height - 8)
                background = (45, 27, 31) if d["lane"] == "current" else (34, 39, 44)
                text_color = (255, 235, 235) if d["lane"] == "current" else (215, 221, 226)
                cv2.rectangle(frame, (label_left, label_top), (label_left + text_width + 10, y + baseline), background, -1)
                cv2.putText(frame, label, (label_left + 5, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1, cv2.LINE_AA)

            encoder.stdin.write(frame.tobytes())
            if progress and (frame_index == 1 or frame_index % 3 == 0):
                pct = min(99, round(frame_index * 100 / frame_count)) if frame_count else 0
                progress(pct, f"Analyzing frame {frame_index}" + (f" of {frame_count}" if frame_count else ""))
        encoder.stdin.close()
        encoder_code = encoder.wait()
        decoder_code = decoder.wait()
        if encoder_code != 0:
            raise RuntimeError(encoder.stderr.read().decode("utf-8", "replace") or "FFmpeg could not encode the processed video.")
        if decoder_code != 0:
            raise RuntimeError(decoder.stderr.read().decode("utf-8", "replace") or "FFmpeg could not decode the input video.")
    except Exception:
        decoder.kill()
        encoder.kill()
        output.unlink(missing_ok=True)
        raise

    levels = {level: sum(item["severity"] == level for item in detections) for level in ("HIGH", "MEDIUM", "LOW")}
    if progress:
        progress(100, "Processing complete")
    adjacent_count, outside_count, uncertain_count = len(ignored_unique), len(outside_unique), len(uncertain_unique)
    ego_count = len(detections)
    total_unique = ego_count + adjacent_count + outside_count
    lighting_condition, lighting_luminance = classify_lighting(lighting_samples)
    current_confidence = last_lane_geometry.confidence if last_lane_geometry is not None else 0.0
    current_lane_status = last_lane_geometry.tracking_status if last_lane_geometry is not None else "UNCERTAIN"
    alert_intervals = forward_alerts.finish(frame_index)
    return {"video": source.name, "frame_count": frame_index,
        **inference_runtime(device),
        "fps": round(fps, 4), "forward_alerts": alert_intervals,
        "forward_alert_count": len(alert_intervals),
        "forward_alert_method": "Repeated ego-lane detections with road contact above 72% of image height",
        "distance_estimate_method": "Approximate flat-road pinhole-camera estimate from normalized road-contact position; camera height, downward pitch, and vertical field of view are adjustable in the UI",
        "raw_yolo_detections": raw_count, "total_unique_potholes": total_unique,
        "unique_detections": total_unique,
        "ego_lane": 1, "ego_lane_potholes": ego_count,
        "current_lane_potholes": ego_count, "adjacent_lane_potholes": adjacent_count,
        "outside_road_potholes": outside_count, "uncertain_lane_potholes": uncertain_count,
        "ignored_potholes": adjacent_count + outside_count + uncertain_count,
        "ignored_adjacent_lane_potholes": adjacent_count,
        "lane_status": current_lane_status,
        "lane_status_frames": lane_status_counts,
        "lane_detection_confidence": round(sum(lane_confidences) / len(lane_confidences), 4) if lane_confidences else 0,
        "lane_confidence_current": round(current_confidence, 4),
        "last_valid_lane_frame": last_lane_geometry.last_valid_frame if last_lane_geometry is not None else None,
        "lighting_condition": lighting_condition, "lighting_luminance_median": lighting_luminance,
        "lighting_method": "median grayscale luminance from 1 Hz frame samples",
        **vehicle_speed_status(),
        "high_risk": levels["HIGH"], "medium_risk": levels["MEDIUM"], "low_risk": levels["LOW"],
        "average_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0,
        "detections": detections, "ignored_detections": ignored_records}
