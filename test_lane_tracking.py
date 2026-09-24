"""Deterministic temporal regression checks; no model weights or FFmpeg needed.

Run: python -m unittest test_lane_tracking -v
Synthetic candidates isolate tracking from image-detection noise.
"""
import unittest
from unittest.mock import patch

import numpy as np
import cv2

from backend.lane_detector import LaneDetector


class LaneTrackingTests(unittest.TestCase):
    def setUp(self):
        self.detector = LaneDetector(1000, 600, config={"max_hold_frames": 10})
        self.left = np.linspace(0.46, 0.22, 12).astype(np.float32)
        self.right = np.linspace(0.54, 0.78, 12).astype(np.float32)
        self.frame = np.full((600, 1000, 3), 100, np.uint8)

    def update(self, left, right, quality=0.9):
        def candidates(segments, side):
            curve = left if side == "left" else right
            return [] if curve is None else [(curve, quality, 1.0)]
        with patch.object(self.detector, "_fit_side_candidates", side_effect=candidates):
            return self.detector.update(self.frame)

    def test_visible_side_updates_and_missing_side_expires(self):
        self.assertTrue(self.update(self.left, self.right).reliable)
        before = self.detector._left.copy()
        width = self.detector._right - before
        geometry = self.update(self.left + 0.02, None)
        self.assertTrue(geometry.left_detected)
        self.assertFalse(geometry.right_detected)
        self.assertEqual(geometry.tracking_status, "TEMPORARILY_OCCLUDED")
        self.assertTrue(np.all(self.detector._left > before))
        np.testing.assert_allclose(self.detector._right - self.detector._left, width, atol=1e-6)
        for _ in range(11):
            geometry = self.update(self.left + 0.02, None)
        self.assertFalse(geometry.reliable)
        self.assertIsNone(geometry.right_boundary)

    def test_single_side_cannot_invent_initial_lane(self):
        for _ in range(5):
            geometry = self.update(self.left, None)
        self.assertFalse(geometry.reliable)
        self.assertEqual(len(geometry.polygon), 0)

    def test_one_frame_jump_is_rejected_and_sustained_shift_recovers(self):
        self.update(self.left, self.right)
        before = self.detector._left.copy()
        geometry = self.update(self.left + 0.08, self.right + 0.08)
        self.assertFalse(geometry.left_detected)
        np.testing.assert_allclose(self.detector._left, before)
        self.update(self.left, self.right)
        for _ in range(15):
            previous = self.detector._left.copy()
            geometry = self.update(self.left + 0.08, self.right + 0.08)
            self.assertLessEqual(float(np.max(np.abs(self.detector._left - previous))), 0.012001)
        self.assertTrue(geometry.reliable)
        self.assertLess(float(np.mean(np.abs(self.detector._left - (self.left + 0.08)))), 0.01)

    def test_inconsistent_alternatives_do_not_confirm_switch(self):
        self.update(self.left, self.right)
        before = self.detector._left.copy()
        for offset in (0.07, -0.07, 0.07, -0.07):
            self.update(self.left + offset, self.right + offset)
        np.testing.assert_allclose(self.detector._left, before)

    def test_blank_frames_expire_then_reacquire(self):
        self.update(self.left, self.right)
        for _ in range(11):
            geometry = self.update(None, None)
        self.assertFalse(geometry.reliable)
        self.assertEqual(len(geometry.polygon), 0)
        self.assertTrue(self.update(self.left, self.right).reliable)

    def test_crossed_pair_is_not_published(self):
        geometry = self.update(self.right, self.left)
        self.assertFalse(geometry.reliable)
        self.assertIsNone(geometry.last_valid_frame)

    def test_low_quality_measurement_moves_boundary_less(self):
        self.update(self.left, self.right)
        self.update(self.left + 0.02, self.right + 0.02, quality=0.1)
        weak_movement = float(np.mean(self.detector._left - self.left))
        self.setUp()
        self.update(self.left, self.right)
        self.update(self.left + 0.02, self.right + 0.02, quality=0.95)
        self.assertGreater(float(np.mean(self.detector._left - self.left)), weak_movement)

    def test_wide_visible_lane_is_accepted(self):
        left = np.linspace(0.46, 0.03, 12).astype(np.float32)
        right = np.linspace(0.54, 0.97, 12).astype(np.float32)
        self.assertTrue(self.update(left, right).reliable)

    def test_visible_boundary_can_exit_image_above_anchor(self):
        frame = self.frame.copy()
        cv2.line(frame, (0, 400), (480, 210), (240, 240, 240), 7)
        cv2.line(frame, (540, 210), (760, 515), (240, 240, 240), 7)
        # Exercise real edge extraction/fitting on both left and right crops.
        for mirrored in (False, True):
            with self.subTest(mirrored=mirrored):
                detector = LaneDetector(1000, 600)
                source = cv2.flip(frame, 1) if mirrored else frame
                geometry = detector.update(source)
                self.assertTrue(geometry.reliable)
                boundary = geometry.right_boundary if mirrored else geometry.left_boundary
                self.assertTrue(np.all(boundary[:, 0] >= 0))
                self.assertTrue(np.all(boundary[:, 0] <= 1000))
                if mirrored:
                    self.assertEqual(int(boundary[:, 0].max()), 1000)
                else:
                    self.assertEqual(int(boundary[:, 0].min()), 0)

    def test_daylight_paint_survives_but_broad_shadow_edge_does_not(self):
        frame = self.frame.copy()
        cv2.rectangle(frame, (240, 190), (360, 510), (150, 150, 150), -1)
        edges = self.detector._edges(frame)
        self.assertEqual(int(np.count_nonzero(edges[220:480, 235:245])), 0)
        cv2.line(frame, (480, 200), (80, 510), (250, 250, 250), 8)
        edges = self.detector._edges(frame)
        self.assertGreater(int(np.count_nonzero(edges)), 200)

    def test_acquisition_threshold_does_not_flicker_an_established_lane(self):
        self.detector.config["acquire_confidence_threshold"] = 0.85
        self.assertTrue(self.update(self.left, self.right).reliable)
        established = self.update(self.left, self.right, quality=0.1)
        self.assertTrue(established.reliable)
        self.assertLess(established.confidence, 0.85)
        self.setUp()
        self.detector.config["acquire_confidence_threshold"] = 0.85
        self.assertFalse(self.update(self.left, self.right, quality=0.1).reliable)


if __name__ == "__main__":
    unittest.main()
