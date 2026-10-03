"""Tests for result report generation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.report import generate_report


class ReportGenerationTest(unittest.TestCase):
    def test_generates_plots_and_summary_from_metric_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scores = {
                "accuracy": 0.8,
                "precision": 0.75,
                "recall": 0.7,
                "f1": 0.72,
                "roc_auc": 0.85,
            }
            baseline = {
                "models": {
                    name: {"test": scores}
                    for name in ("logistic_regression", "linear_svm", "random_forest")
                }
            }
            classifier = {"metrics": {"test": scores}}
            localization = {
                "thresholds": {
                    "0.20": {
                        "test_image_mean_best_iou": 0.1,
                        "pneumonia_patient_mean_best_iou": 0.2,
                    }
                }
            }
            inputs = {
                "baseline.json": baseline,
                "reference.json": classifier,
                "densenet.json": classifier,
                "reference-localization.json": localization,
                "densenet-localization.json": localization,
            }
            for filename, payload in inputs.items():
                (root / filename).write_text(json.dumps(payload), encoding="utf-8")

            outputs = generate_report(
                baseline_path=root / "baseline.json",
                reference_cnn_path=root / "reference.json",
                densenet_path=root / "densenet.json",
                reference_localization_path=root / "reference-localization.json",
                densenet_localization_path=root / "densenet-localization.json",
                output_dir=root / "report",
            )
            self.assertTrue(outputs["classification_plot"].is_file())
            self.assertTrue(outputs["localization_plot"].is_file())
            self.assertIn("DenseNet121", outputs["summary"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
