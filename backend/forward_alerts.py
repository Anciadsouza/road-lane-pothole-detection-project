"""Playback intervals for repeatedly observed potholes farther up the ego lane."""
from __future__ import annotations


FAR_CONTACT_MAX_FRACTION = 0.72
MIN_CONFIDENCE = 0.35
MIN_OBSERVATIONS = 2
MAX_GAP_SECONDS = 1.0
DISPLAY_TAIL_SECONDS = 0.9
DISTANCE_SAMPLE_HZ = 5


class ForwardAlertTimeline:
    def __init__(self, fps: float, frame_height: int):
        self.fps = max(float(fps), 1.0)
        self.frame_height = max(int(frame_height), 1)
        self._active: dict[int, dict] = {}
        self._intervals: list[dict] = []

    def observe(self, track_id: int, object_id: int, frame: int, bottom_y: float,
                confidence: float) -> None:
        """Only a visible, distant ego-lane detection reaches this method."""
        if bottom_y / self.frame_height > FAR_CONTACT_MAX_FRACTION or confidence < MIN_CONFIDENCE:
            return
        previous = self._active.get(track_id)
        if previous is None or frame - previous["last_frame"] > round(self.fps * MAX_GAP_SECONDS):
            if previous is not None:
                self._intervals.append(previous)
            previous = {"object_id": object_id, "first_frame": frame,
                        "last_frame": frame, "observations": 0,
                        "confidence": confidence, "distance_samples": []}
            self._active[track_id] = previous
        previous["last_frame"] = frame
        previous["observations"] += 1
        if confidence > previous["confidence"]:
            previous["confidence"] = confidence
        sample_step = max(1, round(self.fps / DISTANCE_SAMPLE_HZ))
        if (not previous["distance_samples"] or
                frame - previous["distance_samples"][-1]["frame"] >= sample_step):
            previous["distance_samples"].append({
                "frame": frame,
                "contact_y_fraction": round(bottom_y / self.frame_height, 5),
            })

    def finish(self, frame_count: int) -> list[dict]:
        intervals = [*self._intervals, *self._active.values()]
        alerts = []
        for interval in intervals:
            if interval["observations"] < MIN_OBSERVATIONS:
                continue
            start = max(0.0, (interval["first_frame"] - 1) / self.fps)
            end = min(frame_count / self.fps,
                      interval["last_frame"] / self.fps + DISPLAY_TAIL_SECONDS)
            alerts.append({"object_id": interval["object_id"],
                           "start_seconds": round(start, 3), "end_seconds": round(end, 3),
                           "first_frame": interval["first_frame"],
                           "last_frame": interval["last_frame"],
                           "confidence": round(interval["confidence"], 4),
                           "distance_samples": [
                               {"time_seconds": round(max(0.0, (sample["frame"] - 1) / self.fps), 3),
                                "contact_y_fraction": sample["contact_y_fraction"]}
                               for sample in interval["distance_samples"]
                           ]})
        return sorted(alerts, key=lambda item: (item["start_seconds"], item["object_id"]))
