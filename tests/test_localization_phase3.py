"""Tests for CAM extraction and box-based localization metrics."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.densenet121 import build_densenet121
from src.localization import box_iou, compute_cam, connected_component_boxes
from src.reference_cnn import ReferenceCNN


class LocalizationTest(unittest.TestCase):
    def test_components_find_separate_eight_connected_regions(self) -> None:
        heatmap = np.zeros((6, 7), dtype=np.float32)
        heatmap[1:3, 1:3] = 0.5
        heatmap[4, 5:7] = 0.8
        self.assertEqual(
            connected_component_boxes(heatmap, threshold=0.4),
            [(1, 1, 2, 2), (5, 4, 2, 1)],
        )

    def test_zero_threshold_does_not_turn_zero_cam_into_full_image(self) -> None:
        boxes = connected_component_boxes(np.zeros((8, 8), dtype=np.float32), 0.0)
        self.assertEqual(boxes, [])

    def test_iou_handles_identical_and_disjoint_boxes(self) -> None:
        self.assertEqual(box_iou((0, 0, 4, 4), (0, 0, 4, 4)), 1.0)
        self.assertEqual(box_iou((0, 0, 2, 2), (4, 4, 2, 2)), 0.0)

    def test_reference_cnn_cam_has_input_dimensions_and_finite_probability(self) -> None:
        torch.manual_seed(9)
        model = ReferenceCNN(channels=(2,) * 10).eval()
        heatmap, probability = compute_cam(model, torch.randn(1, 1, 32, 32))
        self.assertEqual(heatmap.shape, (32, 32))
        self.assertTrue(np.isfinite(heatmap).all())
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_densenet_accepts_grayscale_images(self) -> None:
        model = build_densenet121().eval()
        with torch.inference_mode():
            logits = model(torch.zeros(1, 1, 64, 64))
        self.assertEqual(tuple(logits.shape), (1, 2))
        heatmap, probability = compute_cam(model, torch.zeros(1, 1, 64, 64))
        self.assertEqual(heatmap.shape, (64, 64))
        self.assertTrue(np.isfinite(heatmap).all())
        self.assertTrue(np.isfinite(probability))

    def test_rejects_invalid_cam_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            connected_component_boxes(np.zeros((2, 2)), 1.1)


if __name__ == "__main__":
    unittest.main()
