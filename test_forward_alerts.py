"""The forward warning follows stable, distant ego-lane observations."""
import unittest

from backend.forward_alerts import ForwardAlertTimeline


class ForwardAlertTests(unittest.TestCase):
    def test_repeated_far_object_creates_playback_interval(self):
        timeline = ForwardAlertTimeline(24, 360)
        timeline.observe(7, 2, 24, 200, 0.62)
        timeline.observe(7, 2, 26, 215, 0.78)
        alerts = timeline.finish(240)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["object_id"], 2)
        self.assertAlmostEqual(alerts[0]["start_seconds"], 23 / 24, places=3)
        self.assertGreater(alerts[0]["end_seconds"], 26 / 24)
        self.assertEqual(alerts[0]["confidence"], 0.78)

    def test_single_detection_or_near_object_does_not_alert(self):
        timeline = ForwardAlertTimeline(24, 360)
        timeline.observe(1, 1, 10, 180, 0.8)
        timeline.observe(2, 2, 20, 315, 0.8)
        timeline.observe(2, 2, 21, 300, 0.8)
        timeline.observe(3, 3, 30, 170, 0.3)
        timeline.observe(3, 3, 31, 175, 0.3)
        self.assertEqual(timeline.finish(240), [])

    def test_gap_creates_separate_intervals_and_tail_is_bounded(self):
        timeline = ForwardAlertTimeline(24, 360)
        for frame in (1, 2, 70, 71):
            timeline.observe(1, 1, frame, 180, 0.8)
        alerts = timeline.finish(72)
        self.assertEqual(len(alerts), 2)
        self.assertLess(alerts[0]["end_seconds"], alerts[1]["start_seconds"])
        self.assertEqual(alerts[1]["end_seconds"], 3.0)

    def test_independent_tracks_are_sorted_by_time(self):
        timeline = ForwardAlertTimeline(24, 360)
        for frame in (48, 49):
            timeline.observe(8, 8, frame, 170, 0.6)
        for frame in (3, 4):
            timeline.observe(2, 2, frame, 190, 0.6)
        self.assertEqual([a["object_id"] for a in timeline.finish(96)], [2, 8])

    def test_short_non_far_gap_does_not_duplicate_same_object(self):
        timeline = ForwardAlertTimeline(24, 360)
        for frame in (62, 63, 76, 77):
            timeline.observe(2, 2, frame, 220, 0.6)
        self.assertEqual(len(timeline.finish(100)), 1)


if __name__ == "__main__":
    unittest.main()
