"""Device selection checks independent of the machine's GPU availability."""
import os
import unittest
from unittest.mock import patch

from backend.detector import get_inference_device, inference_runtime


class InferenceDeviceTests(unittest.TestCase):
    def test_auto_prefers_cuda(self):
        with patch.dict(os.environ, {"YOLO_DEVICE": "auto"}), patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(get_inference_device(), "cuda:0")

    def test_auto_falls_back_to_cpu(self):
        with patch.dict(os.environ, {"YOLO_DEVICE": "auto"}), patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(get_inference_device(), "cpu")

    def test_cpu_override(self):
        with patch.dict(os.environ, {"YOLO_DEVICE": "cpu"}), patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(get_inference_device(), "cpu")
            self.assertEqual(inference_runtime()["inference_device_name"], "CPU")

    def test_explicit_missing_gpu_fails(self):
        with patch.dict(os.environ, {"YOLO_DEVICE": "cuda:1"}), patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=1):
            with self.assertRaises(RuntimeError):
                get_inference_device()

    def test_invalid_device_fails(self):
        with patch.dict(os.environ, {"YOLO_DEVICE": "invalid"}):
            with self.assertRaises(ValueError):
                get_inference_device()


if __name__ == "__main__":
    unittest.main()
