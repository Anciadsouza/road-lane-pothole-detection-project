"""Video-aware lane-marking detector for the forward road camera."""
from __future__ import annotations

from dataclasses import dataclass
import os

import cv2
import numpy as np


# Inspection note: the active processor already detects road markings with Canny
# and probabilistic Hough lines; its polygon is a representation of those fitted
# curves. The former fixed fallback was the remaining hard-coded lane assumption.
# These values tune detection and the calibrated image location of the camera path.
LANE_CONFIG = {
    "roi_top": 0.32,
    "roi_bottom": 0.86,
    "camera_x": 0.50,
    "camera_boundary_margin": 0.010,
    "expected_lane_width_anchor": 0.27,
    "anchor_y": 0.72,
    "canny_low": 42,
    "canny_high": 132,
    "hough_threshold": 36,
    "min_line_length": 38,
    "max_line_gap": 72,
    "min_abs_dx_dy": 0.20,
    "max_abs_dx_dy": 3.2,
    "cluster_anchor_fraction": 0.085,
    "cluster_slope_tolerance": 1.15,
    "smoothing_factor": 0.42,
    "max_boundary_jump_fraction": 0.22,
    "confidence_threshold": 0.32,
    # Keep the last measured lane for a brief period at the clip's frame rate.
    "max_hold_seconds": 0.65,
    "max_hold_frames": 18,  # used when caller does not supply FPS
    "confidence_decay": 0.90,  # retained for configuration compatibility
    "night_mean_threshold": 88.0,
    "low_contrast_range_threshold": 74.0,
    "night_gamma": 1.28,
    "clahe_clip_limit": 2.0,
    "glare_threshold": 248,
    "glare_dilate_size": 7,
    "road_sat_max": 105,
    "road_value_min": 34,
    "road_value_max": 238,
    "sample_count": 12,
    "debug": os.getenv("DEBUG_LANES", "false").casefold() in {"1", "true", "yes", "on"},
}


@dataclass
class LaneGeometry:
    polygon: np.ndarray
    left_boundary: np.ndarray | None
    right_boundary: np.ndarray | None
    candidate_lines: list[tuple[int, int, int, int]]
    edge_map: np.ndarray
    road_mask: np.ndarray
    left_detected: bool
    right_detected: bool
    camera_x: float
    anchor_y: float
    debug: bool
    confidence: float
    reliable: bool
    tracking_status: str
    frame_index: int
    last_valid_frame: int | None
    centerline: np.ndarray | None
    vanishing_point: tuple[int, int] | None
    low_light: bool
    low_contrast: bool


class LaneDetector:
    """Selects the nearest detected road-marking curve on each side of the camera."""

    def __init__(self, width: int, height: int, config: dict | None = None, fps: float = 24.0):
        self.width = width
        self.height = height
        self.config = {**LANE_CONFIG, **(config or {})}
        self.config["max_hold_frames"] = int(self.config.get("max_hold_frames", 18))
        if config is None or "max_hold_frames" not in config:
            self.config["max_hold_frames"] = max(5, min(15, round(float(fps) * self.config["max_hold_seconds"])))
        self.y_norm = np.linspace(self.config["roi_top"], self.config["roi_bottom"], self.config["sample_count"])
        self.camera_x = self.config["camera_x"]
        self._left: np.ndarray | None = None
        self._right: np.ndarray | None = None
        self._missed_left = 0
        self._missed_right = 0
        self._left_confidence = 0.0
        self._right_confidence = 0.0
        self._left_base_confidence = 0.0
        self._right_base_confidence = 0.0
        self._left_fit_quality = 0.0
        self._right_fit_quality = 0.0
        self._frame_index = 0
        self._last_valid_frame: int | None = None
        self._last_lane_width: float | None = None
        self._last_status = "UNCERTAIN"
        self._low_light = False
        self._low_contrast = False
        self._enhanced_gray: np.ndarray | None = None

    def _lane_preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """Create a lane-only enhanced grayscale image and a bright-glare mask."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_top = int(self.config["roi_top"] * self.height)
        roi_bottom = int(self.config["roi_bottom"] * self.height)
        road_gray = gray[roi_top:roi_bottom]
        if road_gray.size:
            median_luminance = float(np.median(road_gray))
            contrast_range = float(np.percentile(road_gray, 90) - np.percentile(road_gray, 20))
        else:
            median_luminance = float(np.median(gray))
            contrast_range = float(np.percentile(gray, 90) - np.percentile(gray, 20))
        low_light = median_luminance < self.config["night_mean_threshold"]
        low_contrast = contrast_range < self.config["low_contrast_range_threshold"]
        enhance = low_light or low_contrast
        # Suppress only near-saturated headlamp cores before enhancement so
        # CLAHE/gamma does not turn glare blobs into long Hough candidates.
        thresholded = cv2.inRange(gray, self.config["glare_threshold"], 255)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded, 8)
        glare = np.zeros_like(gray)
        for component in range(1, count):
            x, y, w, h, area = stats[component]
            # Small/thin saturated lane paint stays eligible; only broad
            # headlamp-like clipped blobs are treated as glare.
            if area >= 36 and w >= 6 and h >= 6:
                glare[labels == component] = 255
        k = max(3, int(self.config["glare_dilate_size"]) | 1)
        glare = cv2.dilate(glare, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        if enhance:
            capped = np.minimum(gray, 242).astype(np.uint8)
            gamma = self.config["night_gamma"] if low_light else 1.0
            lut = np.array([min(255, round(255 * (i / 255.0) ** (1.0 / gamma)))
                            for i in range(256)], dtype=np.uint8)
            enhanced = cv2.LUT(capped, lut)
            enhanced = cv2.createCLAHE(clipLimit=self.config["clahe_clip_limit"],
                                       tileGridSize=(8, 8)).apply(enhanced)
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        else:
            enhanced = cv2.GaussianBlur(gray, (5, 5), 0)
        self._low_light = low_light
        self._low_contrast = low_contrast
        self._enhanced_gray = enhanced
        return enhanced, glare, low_light

    def _edges(self, frame: np.ndarray) -> np.ndarray:
        gray, glare, low_light = self._lane_preprocess(frame)
        enhanced = low_light or self._low_contrast
        low = max(24, int(self.config["canny_low"] * (0.76 if enhanced else 1.0)))
        high = max(low + 40, int(self.config["canny_high"] * (0.82 if enhanced else 1.0)))
        edges = cv2.Canny(gray, low, high)
        edges[glare > 0] = 0
        mask = np.zeros_like(edges)
        y0 = round(self.config["roi_top"] * self.height)
        y1 = round(self.config["roi_bottom"] * self.height)
        cv2.rectangle(mask, (0, y0), (self.width - 1, y1), 255, -1)
        return cv2.bitwise_and(edges, mask)

    def _road_mask(self, frame: np.ndarray) -> np.ndarray:
        """Approximate drivable asphalt from low-chroma pixels in the road ROI."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        value_min = 12 if self._low_light else self.config["road_value_min"]
        value_max = 245 if self._low_light else self.config["road_value_max"]
        mask = cv2.inRange(hsv, (0, 0, value_min),
                           (180, self.config["road_sat_max"], value_max))
        roi = np.zeros_like(mask)
        y0 = round(self.config["roi_top"] * self.height)
        y1 = round(self.config["roi_bottom"] * self.height)
        roi[y0:y1, :] = 255
        mask = cv2.bitwise_and(mask, roi)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    def _segments(self, frame: np.ndarray, edge_map: np.ndarray) -> list[dict]:
        lines = cv2.HoughLinesP(edge_map, 1, np.pi / 180, self.config["hough_threshold"],
            minLineLength=self.config["min_line_length"], maxLineGap=self.config["max_line_gap"])
        if lines is None:
            return []
        gray = self._enhanced_gray if self._enhanced_gray is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_top = self.config["roi_top"] * self.height
        roi_bottom = self.config["roi_bottom"] * self.height
        anchor_y = self.config["anchor_y"] * self.height
        camera_px = self.camera_x * self.width
        segments = []
        for raw in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = [int(v) for v in raw]
            dx, dy = x2 - x1, y2 - y1
            length = float(np.hypot(dx, dy))
            if length < self.config["min_line_length"] or abs(dy) < 12:
                continue
            slope = dx / dy  # x change per vertical pixel.
            if max(y1, y2) < roi_top or min(y1, y2) > roi_bottom:
                continue
            x_anchor = x1 + (anchor_y - y1) * slope
            magnitude = abs(slope)
            if magnitude > self.config["max_abs_dx_dy"]:
                continue
            if magnitude < self.config["min_abs_dx_dy"]:
                # Near-vertical markings converge toward a vanishing point;
                # retain them only when they sit near the camera path.
                if abs(x_anchor - camera_px) > self.width * 0.12:
                    continue
                side = "left" if x_anchor < camera_px else "right"
            else:
                side = "left" if slope < 0 else "right"
            margin = self.config["camera_boundary_margin"] * self.width
            if side == "left" and not 0 <= x_anchor < camera_px - margin:
                continue
            if side == "right" and not camera_px + margin < x_anchor < self.width:
                continue
            count = max(5, int(length / 12))
            xs = np.linspace(x1, x2, count).astype(int).clip(0, self.width - 1)
            ys = np.linspace(y1, y2, count).astype(int).clip(0, self.height - 1)
            brightness = float(np.mean(gray[ys, xs]))
            # Discard dark cracks/shadows; lane paint and road edges are brighter.
            brightness_floor = 76 if self._low_light or self._low_contrast else 112
            if brightness < brightness_floor:
                continue
            distance = abs(camera_px - x_anchor)
            score = length * (0.35 + brightness / 255) / (1 + 1.8 * distance / self.width)
            segments.append({"points": np.array([[x1, y1], [x2, y2]], dtype=np.float32),
                "anchor_x": float(x_anchor), "side": side, "length": length,
                "slope": slope, "score": score,
                "line": (x1, y1, x2, y2)})
        return segments

    def _fit_side(self, segments: list[dict], side: str) -> np.ndarray | None:
        candidates = self._fit_side_candidates(segments, side)
        return candidates[0][0] if candidates else None

    def _fit_side_candidates(self, segments: list[dict], side: str) -> list[tuple[np.ndarray, float, float]]:
        side_segments = [segment for segment in segments if segment["side"] == side]
        if not side_segments:
            return []
        previous = self._left if side == "left" else self._right
        anchor_idx = int(np.argmin(abs(self.y_norm - self.config["anchor_y"])))
        camera_px = self.camera_x * self.width
        if previous is None:
            # Prefer camera-near candidates, while retaining alternatives for
            # joint lane-width and vanishing-geometry checks below.
            preference = lambda item: item["score"] / (1.0 + 5.0 * abs(item["anchor_x"] - camera_px) / self.width)
        else:
            previous_anchor = float(previous[anchor_idx]) * self.width
            preference = lambda item: item["score"] * np.exp(
                -abs(item["anchor_x"] - previous_anchor) / max(self.width * 0.10, 1))
        ordered = sorted(side_segments, key=preference, reverse=True)
        tolerance = self.config["cluster_anchor_fraction"] * self.width
        tried: list[dict] = []
        candidates: list[tuple[np.ndarray, float, float]] = []
        for seed in ordered:
            near_vertical = abs(seed["slope"]) < self.config["min_abs_dx_dy"]
            if any(abs(seed["anchor_x"] - prior["anchor_x"]) <= tolerance * 0.45 and
                   abs(seed["slope"] - prior["slope"]) <= self.config["cluster_slope_tolerance"] * 0.5
                   for prior in tried):
                continue
            tried.append(seed)
            cluster_anchor_tolerance = min(tolerance, self.width * 0.035) if near_vertical else tolerance
            cluster_slope_tolerance = min(self.config["cluster_slope_tolerance"], 0.16) if near_vertical else self.config["cluster_slope_tolerance"]
            cluster = [item for item in side_segments if abs(item["anchor_x"] - seed["anchor_x"]) <= cluster_anchor_tolerance
                       and abs(item["slope"] - seed["slope"]) <= cluster_slope_tolerance]
            curve = self._fit_cluster(cluster, side)
            if curve is not None:
                quality = self._left_fit_quality if side == "left" else self._right_fit_quality
                candidates.append((curve, quality, float(preference(seed))))
                if len(candidates) >= 8:
                    break
        return candidates

    def _select_lane_pair(self, left_candidates: list[tuple[np.ndarray, float, float]],
                          right_candidates: list[tuple[np.ndarray, float, float]]) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not left_candidates or not right_candidates:
            return None, None
        anchor = int(np.argmin(abs(self.y_norm - self.config["anchor_y"])))
        target_width = self._last_lane_width or self.config["expected_lane_width_anchor"]
        options = []
        for left, lq, lp in left_candidates:
            for right, rq, rp in right_candidates:
                anchor_width = float(right[anchor] - left[anchor])
                if not (0.07 <= anchor_width <= 0.52 and left[anchor] < self.camera_x < right[anchor]):
                    continue
                ordered = left + 0.006 < right
                starts = [idx for idx in range(len(ordered)) if bool(np.all(ordered[idx:]))]
                if not starts:
                    continue
                start = starts[0]
                if len(ordered) - start < 4:
                    continue
                bottom_width = float(right[-1] - left[-1])
                if not (0.08 <= bottom_width <= 0.96 and
                        0.72 <= bottom_width / max(anchor_width, 1e-3) <= 3.8):
                    continue
                center_error = abs(float((left[anchor] + right[anchor]) * 0.5) - self.camera_x)
                center_score = float(np.exp(-center_error / 0.20))
                width_score = float(np.exp(-abs(anchor_width - target_width) / 0.13))
                if self._left is not None and self._right is not None:
                    temporal_error = (float(np.mean(np.abs(left - self._left))) +
                                      float(np.mean(np.abs(right - self._right)))) * 0.5
                    temporal_score = float(np.exp(-temporal_error / 0.07))
                else:
                    temporal_score = 0.68
                support_score = min(1.0, (lq + rq) * 0.5)
                # Rank complete pairs: image-center proximity alone cannot
                # select an adjacent lane, and width/history reject stripe pairs.
                score = (0.30 * support_score + 0.23 * center_score +
                         0.31 * width_score + 0.16 * temporal_score)
                options.append((score, left, right, lq, rq))
        if not options:
            return None, None
        _, left, right, left_quality, right_quality = max(options, key=lambda item: item[0])
        self._left_fit_quality, self._right_fit_quality = left_quality, right_quality
        return left, right

    def _fit_cluster(self, cluster: list[dict], side: str) -> np.ndarray | None:
        points = np.concatenate([item["points"] for item in cluster], axis=0)
        keep = ((points[:, 1] >= self.config["roi_top"] * self.height) &
                (points[:, 1] <= self.config["roi_bottom"] * self.height))
        points = points[keep]
        if len(points) < 2:
            return None
        ys = points[:, 1] / self.height
        xs = points[:, 0] / self.width
        degree = min(2, len(np.unique(np.round(ys, 3))) - 1)
        if degree < 1:
            return None
        weights = np.sqrt(np.asarray([item["length"] for item in cluster], dtype=np.float32))
        weights = np.repeat(weights, 2)[keep]
        try:
            coefficients = np.polyfit(ys, xs, degree, w=weights)
            for _ in range(2):
                residual = np.abs(xs - np.polyval(coefficients, ys))
                median = float(np.median(residual))
                mad = float(np.median(np.abs(residual - median)))
                inliers = residual <= max(0.012, median + 2.8 * mad)
                if int(inliers.sum()) < max(4, degree + 1):
                    break
                coefficients = np.polyfit(ys[inliers], xs[inliers], degree, w=weights[inliers])
            curve = np.polyval(coefficients, self.y_norm)
        except (ValueError, np.linalg.LinAlgError):
            return None
        curve = np.clip(curve, 0.0, 1.0)
        camera = self.camera_x
        anchor_idx = int(np.argmin(abs(self.y_norm - self.config["anchor_y"])))
        # Boundaries normally meet near the vanishing point above the ego
        # vehicle. Test side membership at/below the camera anchor, not across
        # that convergence point in the upper image.
        path_margin = self.config["camera_boundary_margin"]
        if side == "left" and np.any(curve[anchor_idx:] >= camera - path_margin):
            return None
        if side == "right" and np.any(curve[anchor_idx:] <= camera + path_margin):
            return None
        delta = np.diff(curve)
        monotonic_tolerance = 0.018
        if side == "left" and np.any(delta > monotonic_tolerance):
            return None
        if side == "right" and np.any(delta < -monotonic_tolerance):
            return None
        if float(np.max(curve) - np.min(curve)) > 0.82:
            return None
        coverage = min(1.0, float(np.ptp(ys)) / max(self.config["roi_bottom"] - self.config["roi_top"], 0.01))
        support = min(1.0, len(cluster) / 4.0)
        quality = min(0.98, 0.25 + 0.34 * coverage + 0.40 * support)
        if side == "left":
            self._left_fit_quality = quality
        else:
            self._right_fit_quality = quality
        return curve.astype(np.float32)

    def _smooth(self, detected: np.ndarray | None, side: str) -> tuple[np.ndarray | None, bool, float]:
        previous = self._left if side == "left" else self._right
        missed = self._missed_left if side == "left" else self._missed_right
        previous_confidence = self._left_base_confidence if side == "left" else self._right_base_confidence
        if detected is not None:
            anchor = int(np.argmin(abs(self.y_norm - self.config["anchor_y"])))
            jump = abs(float(detected[anchor] - previous[anchor])) if previous is not None else 0.0
            if previous is not None and missed <= self.config["max_hold_frames"] and jump > self.config["max_boundary_jump_fraction"]:
                detected = None
            else:
                alpha = self.config["smoothing_factor"]
                smoothed = detected if previous is None else previous * (1 - alpha) + detected * alpha
                marking = self._left_fit_quality if side == "left" else self._right_fit_quality
                temporal = (float(np.exp(-float(np.mean(np.abs(detected - previous))) / 0.04))
                            if previous is not None else 0.72)
                confidence = 0.72 * marking + 0.28 * temporal
                if side == "left":
                    self._left, self._missed_left = smoothed, 0
                    self._left_confidence = self._left_base_confidence = confidence
                else:
                    self._right, self._missed_right = smoothed, 0
                    self._right_confidence = self._right_base_confidence = confidence
                return smoothed, True, confidence
        missed += 1
        if side == "left":
            self._missed_left = missed
        else:
            self._missed_right = missed
        # Grace-period confidence declines gradually instead of vanishing as
        # 0.9**missed (which crossed the old threshold after only ~11 frames).
        hold = max(1, self.config["max_hold_frames"])
        confidence = previous_confidence * max(0.65, 1.0 - 0.35 * missed / hold)
        if missed > self.config["max_hold_frames"]:
            previous, confidence = None, 0.0
            previous_confidence = 0.0
        if side == "left":
            self._left, self._left_confidence, self._left_base_confidence = previous, confidence, previous_confidence
        else:
            self._right, self._right_confidence, self._right_base_confidence = previous, confidence, previous_confidence
        return previous, False, confidence

    def update(self, frame: np.ndarray, frame_index: int | None = None) -> LaneGeometry:
        self._frame_index = int(frame_index) if frame_index is not None else self._frame_index + 1
        edge_map = self._edges(frame)
        road_mask = self._road_mask(frame)
        segments = self._segments(frame, edge_map)
        left_candidates = self._fit_side_candidates(segments, "left")
        right_candidates = self._fit_side_candidates(segments, "right")
        selected_left, selected_right = self._select_lane_pair(left_candidates, right_candidates)
        left, left_detected, left_conf = self._smooth(selected_left, "left")
        right, right_detected, right_conf = self._smooth(selected_right, "right")
        anchor_idx = int(np.argmin(abs(self.y_norm - self.config["anchor_y"])))
        start_idx = None
        if left is not None and right is not None:
            ordered = (left + 0.006) < right
            for candidate in range(len(ordered)):
                if bool(np.all(ordered[candidate:])):
                    start_idx = candidate
                    break
        valid = (left is not None and right is not None and start_idx is not None
                 and len(self.y_norm) - start_idx >= 4)
        measured = bool(left_detected and right_detected and valid)
        width_ok = True
        taper_ok = True
        if measured:
            lane_width = float(right[anchor_idx] - left[anchor_idx])
            width_ok = 0.07 <= lane_width <= 0.52
            width_consistency = float(np.exp(-abs(lane_width - (self._last_lane_width or
                                      self.config["expected_lane_width_anchor"])) / 0.13))
            # The top of a forward-road lane should converge relative to its
            # lower span; this is a plausibility check, not a fixed shape.
            top_width = float(right[start_idx] - left[start_idx])
            bottom_width = float(right[-1] - left[-1])
            taper_ok = top_width <= bottom_width * 1.22 and top_width > 0
            geometry_quality = (0.60 * float(width_ok) + 0.40 * width_consistency)
            geometry_quality *= 0.86 + 0.14 * float(taper_ok)
            marking_quality = min(left_conf, right_conf)
            temporal_quality = 1.0 if self._last_valid_frame is not None else 0.72
            road_samples = []
            for x_norm, y_norm in zip((left[start_idx:] + right[start_idx:]) * 0.5,
                                      self.y_norm[start_idx:]):
                x, y = int(x_norm * self.width), int(y_norm * self.height)
                patch = road_mask[max(0, y - 4):min(self.height, y + 5),
                                  max(0, x - 4):min(self.width, x + 5)]
                if patch.size:
                    road_samples.append(float(np.count_nonzero(patch)) / patch.size)
            road_consistency = float(np.mean(road_samples)) if road_samples else 0.0
            confidence = (0.48 * marking_quality + 0.18 * temporal_quality
                          + 0.20 * geometry_quality + 0.14 * road_consistency)
            if width_ok:
                self._last_lane_width = lane_width if self._last_lane_width is None else (
                    0.75 * self._last_lane_width + 0.25 * lane_width)
            self._last_valid_frame = self._frame_index
        elif valid:
            confidence = min(left_conf, right_conf)
        else:
            confidence = 0.0
        plausible_geometry = not measured or (width_ok and taper_ok)
        reliable = valid and plausible_geometry and confidence >= self.config["confidence_threshold"]
        if reliable:
            if measured:
                status = "TRACKED"
            else:
                status = "TEMPORARILY_OCCLUDED"
        elif valid and confidence >= self.config["confidence_threshold"] * 0.72:
            status = "LOW_CONFIDENCE"
        else:
            status = "UNCERTAIN"
        if not measured and valid and (self._missed_left > 0 or self._missed_right > 0):
            status = "TEMPORARILY_OCCLUDED" if reliable else "LOW_CONFIDENCE"
        self._last_status = status
        left_points = (np.column_stack((left * self.width, self.y_norm * self.height)).round().astype(np.int32)
                       if left is not None else None)
        right_points = (np.column_stack((right * self.width, self.y_norm * self.height)).round().astype(np.int32)
                        if right is not None else None)
        polygon = (np.vstack((left_points[start_idx:], right_points[start_idx:][::-1])).astype(np.int32)
                   if reliable and left_points is not None and right_points is not None else np.empty((0, 2), np.int32))
        center_points = (np.column_stack(((left + right) * 0.5 * self.width,
                                         self.y_norm * self.height)).round().astype(np.int32)[start_idx:]
                         if left is not None and right is not None and valid else None)
        vanishing = None
        if left_points is not None and right_points is not None and start_idx is not None:
            vanishing = (int((left_points[start_idx, 0] + right_points[start_idx, 0]) / 2),
                         int((left_points[start_idx, 1] + right_points[start_idx, 1]) / 2))
        visible_left = left_points[start_idx:] if left_points is not None and start_idx is not None else left_points
        visible_right = right_points[start_idx:] if right_points is not None and start_idx is not None else right_points
        return LaneGeometry(polygon, visible_left, visible_right,
            [segment["line"] for segment in segments], edge_map, road_mask, left_detected, right_detected,
            self.camera_x, self.config["anchor_y"], self.config["debug"], confidence, reliable,
            status, self._frame_index, self._last_valid_frame, center_points, vanishing,
            self._low_light, self._low_contrast)


def is_in_current_lane(point: tuple[float, float], polygon: np.ndarray) -> bool:
    if polygon is None or len(polygon) < 3:
        return False
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


def is_on_drivable_road(point: tuple[float, float], road_mask: np.ndarray, radius: int = 4) -> bool:
    x, y = map(lambda value: int(round(value)), point)
    h, w = road_mask.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = road_mask[y0:y1, x0:x1]
    return bool(patch.size and np.count_nonzero(patch) / patch.size >= 0.30)


def draw_lane_overlay(frame: np.ndarray, geometry: LaneGeometry) -> np.ndarray:
    if geometry.reliable:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [geometry.polygon], (28, 190, 125))
        cv2.addWeighted(overlay, 0.14, frame, 0.86, 0, frame)
    if geometry.left_boundary is not None:
        cv2.polylines(frame, [geometry.left_boundary], False, (50, 220, 170), 3, cv2.LINE_AA)
    if geometry.right_boundary is not None:
        cv2.polylines(frame, [geometry.right_boundary], False, (50, 220, 170), 3, cv2.LINE_AA)
    anchor_y = int(geometry.anchor_y * frame.shape[0])
    camera_x = int(geometry.camera_x * frame.shape[1])
    cv2.line(frame, (camera_x, anchor_y), (camera_x, frame.shape[0] - 1), (210, 205, 95), 1, cv2.LINE_AA)
    state_label = {"TRACKED": "EGO LANE · TRACKED", "TEMPORARILY_OCCLUDED": "EGO LANE · TEMPORARILY OCCLUDED",
                   "LOW_CONFIDENCE": "EGO LANE · LOW CONFIDENCE", "UNCERTAIN": "LANE DETECTION UNCERTAIN"}
    label = f"{state_label[geometry.tracking_status]} · {geometry.confidence:.0%}"
    text_x = (int((geometry.left_boundary[0, 0] + geometry.right_boundary[0, 0]) / 2)
              if geometry.left_boundary is not None and geometry.right_boundary is not None else camera_x)
    label_y = int(geometry.left_boundary[0, 1] - 12) if geometry.left_boundary is not None else anchor_y - 10
    cv2.putText(frame, label, (max(8, text_x - 100), max(25, label_y)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (70, 245, 185), 2, cv2.LINE_AA)

    if geometry.debug:
        if geometry.centerline is not None:
            cv2.polylines(frame, [geometry.centerline], False, (255, 180, 35), 2, cv2.LINE_AA)
        if geometry.vanishing_point is not None:
            cv2.circle(frame, geometry.vanishing_point, 6, (255, 90, 220), -1, cv2.LINE_AA)
        detail = f"STATE {geometry.tracking_status}  CONF {geometry.confidence:.0%}  FRAME {geometry.frame_index}"
        if geometry.last_valid_frame is not None:
            detail += f"  LAST {geometry.last_valid_frame}"
        if geometry.low_light:
            detail += "  LOW-LIGHT PREPROCESS"
        elif geometry.low_contrast:
            detail += "  LOW-CONTRAST PREPROCESS"
        cv2.putText(frame, detail, (12, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (230, 230, 230), 1, cv2.LINE_AA)

    if geometry.debug:
        for x1, y1, x2, y2 in geometry.candidate_lines:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 170, 30), 1, cv2.LINE_AA)
        center_x = int(geometry.camera_x * frame.shape[1])
        cv2.line(frame, (center_x, 0), (center_x, frame.shape[0] - 1), (30, 210, 255), 1, cv2.LINE_AA)
        small = cv2.resize(geometry.edge_map, (frame.shape[1] // 4, frame.shape[0] // 4))
        road = cv2.resize(geometry.road_mask, (frame.shape[1] // 4, frame.shape[0] // 4))
        inset = cv2.cvtColor(np.hstack((small, road)), cv2.COLOR_GRAY2BGR)
        h, w = inset.shape[:2]
        frame[8:8 + h, frame.shape[1] - w - 8:frame.shape[1] - 8] = inset
        cv2.rectangle(frame, (frame.shape[1] - w - 8, 8), (frame.shape[1] - 8, 8 + h), (220, 220, 220), 1)
    return frame
