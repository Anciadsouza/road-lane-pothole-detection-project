"""Small centroid tracker that collapses detections across neighboring frames."""
from __future__ import annotations

from math import hypot, sqrt


class CentroidTracker:
    def __init__(self, max_missing: int = 12, max_distance: float = 85.0):
        self.max_missing = max_missing
        self.max_distance = max_distance
        self.next_id = 1
        self.tracks: dict[int, dict] = {}

    def update(self, detections: list[dict], frame_index: int) -> list[tuple[int, bool]]:
        unmatched = set(range(len(detections)))
        matches: list[tuple[int, bool] | None] = [None] * len(detections)
        # Greedy nearest-centroid matching, restricted to the same lane category.
        for track_id, track in sorted(self.tracks.items(), key=lambda item: item[1]["last_frame"], reverse=True):
            frame_gap = frame_index - track["last_frame"]
            if frame_gap > self.max_missing:
                continue
            candidates = [i for i in unmatched if (
                detections[i]["lane"] == track["lane"] or
                "uncertain" in {detections[i]["lane"], track["lane"]})]
            if not candidates:
                continue
            def metrics(i: int) -> tuple[float, float]:
                d = detections[i]
                distance = hypot(d["cx"] - track["cx"], d["cy"] - track["cy"])
                old_box, new_box = track.get("box"), self._box(d)
                overlap = self._iou(old_box, new_box) if old_box and new_box else 0.0
                gate = self.max_distance * min(1.8, sqrt(max(1, frame_gap)))
                cost = 0.60 * (1.0 - overlap) + 0.40 * min(1.0, distance / max(gate, 1.0))
                return cost, distance
            idx = min(candidates, key=lambda i: metrics(i)[0])
            cost, distance = metrics(idx)
            gate = self.max_distance * min(1.8, sqrt(max(1, frame_gap)))
            if distance <= gate or (track.get("box") and self._iou(track["box"], self._box(detections[idx])) >= 0.08):
                unmatched.remove(idx)
                detection = detections[idx]
                track.update(cx=detection["cx"], cy=detection["cy"], last_frame=frame_index,
                             box=self._box(detection))
                if detection["lane"] != "uncertain":
                    track["lane"] = detection["lane"]
                matches[idx] = (track_id, False)
        for idx in unmatched:
            track_id = self.next_id
            self.next_id += 1
            d = detections[idx]
            self.tracks[track_id] = {"cx": d["cx"], "cy": d["cy"], "last_frame": frame_index,
                                     "lane": d["lane"], "box": self._box(d)}
            matches[idx] = (track_id, True)
        self.tracks = {key: value for key, value in self.tracks.items()
                       if frame_index - value["last_frame"] <= self.max_missing}
        return [match for match in matches if match is not None]

    @staticmethod
    def _box(detection: dict) -> tuple[float, float, float, float] | None:
        keys = ("x1", "y1", "x2", "y2")
        return tuple(float(detection[key]) for key in keys) if all(key in detection for key in keys) else None

    @staticmethod
    def _iou(a: tuple[float, float, float, float] | None,
             b: tuple[float, float, float, float] | None) -> float:
        if not a or not b:
            return 0.0
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        return intersection / max(area_a + area_b - intersection, 1e-9)
