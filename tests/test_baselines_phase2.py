"""Focused tests for Phase 2 classification baseline helpers."""

from __future__ import annotations

import unittest

import numpy as np

from src.baselines import _score_model, build_models, classification_metrics


class BaselineMetricsTest(unittest.TestCase):
    def test_reports_binary_metrics_and_confusion_counts(self) -> None:
        metrics = classification_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 1]),
            np.array([-0.8, 0.2, 0.6, 0.9]),
        )
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_rejects_labels_without_both_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "both classes"):
            classification_metrics(
                np.array([0, 0]),
                np.array([0, 0]),
                np.array([-0.2, -0.1]),
            )

    def test_builds_reproducible_baseline_estimators(self) -> None:
        models = build_models(seed=13)
        self.assertEqual(set(models), {
            "logistic_regression",
            "linear_svm",
            "random_forest",
        })
        self.assertEqual(models["random_forest"].random_state, 13)
        self.assertEqual(models["linear_svm"].random_state, 13)

    def test_all_models_produce_scores_for_roc_auc(self) -> None:
        features = np.array([
            [0.0, 0.0],
            [0.1, 0.2],
            [0.8, 0.9],
            [1.0, 0.8],
        ])
        labels = np.array([0, 0, 1, 1])
        for model in build_models(seed=7).values():
            with self.subTest(model=type(model).__name__):
                model.fit(features, labels)
                metrics = _score_model(model, features, labels)
                self.assertIn("roc_auc", metrics)
                self.assertEqual(metrics["tp"] + metrics["fn"], 2)


if __name__ == "__main__":
    unittest.main()
