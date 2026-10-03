"""Train and evaluate flattened-pixel classification baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import LinearSVC

from .preprocessing import NormalizationStats, load_dicom_image, load_normalization_stats

MODEL_NAMES = ("logistic_regression", "linear_svm", "random_forest")
PARTITIONS = ("train", "validation", "test")


def _load_partition(
    manifest_path: str | Path,
    stats: NormalizationStats,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a manifest into flattened, standardized image features and labels."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required_columns = {"patient_id", "image_path", "label"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"Split manifest {path} must contain columns "
                f"{', '.join(sorted(required_columns))}."
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"Split manifest is empty: {path}")

    patient_ids: set[str] = set()
    labels = np.empty(len(rows), dtype=np.int64)
    features = np.empty(
        (len(rows), stats.mean.size),
        dtype=np.float32,
    )
    for index, row in enumerate(rows):
        patient_id = row["patient_id"].strip()
        if not patient_id:
            raise ValueError(f"Empty patient_id in {path}, row {index + 2}.")
        if patient_id in patient_ids:
            raise ValueError(f"Duplicate patient_id {patient_id!r} in {path}.")
        patient_ids.add(patient_id)

        try:
            label_value = float(row["label"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid label for patient {patient_id!r} in {path}: {row['label']!r}."
            ) from exc
        if not np.isfinite(label_value) or label_value not in (0, 1):
            raise ValueError(
                f"Label must be exactly 0 or 1 for patient {patient_id!r} in {path}."
            )
        labels[index] = int(label_value)

        image_path = row["image_path"].strip()
        if not image_path:
            raise ValueError(f"Empty image_path for patient {patient_id!r} in {path}.")
        image = load_dicom_image(image_path, stats)
        features[index] = image.reshape(-1)

    if np.unique(labels).size != 2:
        raise ValueError(f"Split manifest must contain both classes: {path}")
    return features, labels


def build_models(seed: int = 42) -> dict[str, Any]:
    """Create the three classical models used as project baselines."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=seed,
        ),
        "linear_svm": LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | int]:
    """Compute binary classification metrics with Pneumonia as the positive class."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    scores = np.asarray(scores)
    if labels.ndim != 1 or predictions.shape != labels.shape or scores.shape != labels.shape:
        raise ValueError("Labels, predictions, and scores must be matching one-dimensional arrays.")
    if labels.size == 0 or not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must be a non-empty binary array.")
    if not np.isin(predictions, (0, 1)).all():
        raise ValueError("Predictions must contain only 0 and 1.")
    if not np.isfinite(scores).all():
        raise ValueError("Prediction scores must contain only finite values.")
    if np.unique(labels).size != 2:
        raise ValueError("Metrics require both classes to be present in the labels.")

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=(0, 1)).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _score_model(model: Any, features: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    predictions = model.predict(features)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
    else:
        scores = model.predict_proba(features)[:, 1]
    return classification_metrics(labels, predictions, scores)


def run_baselines(
    splits_dir: str | Path = "data/splits",
    stats_path: str | Path | None = None,
    output_path: str | Path = "results/baseline_metrics.json",
    seed: int = 42,
    model_names: tuple[str, ...] = MODEL_NAMES,
) -> dict[str, Any]:
    """Fit baselines on training data and report untouched validation/test metrics."""
    splits_dir = Path(splits_dir)
    stats_file = Path(stats_path) if stats_path else splits_dir / "normalization.npz"
    unknown_models = set(model_names) - set(MODEL_NAMES)
    if unknown_models:
        raise ValueError(f"Unknown model name(s): {', '.join(sorted(unknown_models))}.")
    if not model_names:
        raise ValueError("Select at least one model.")

    stats = load_normalization_stats(stats_file)
    train_features, train_labels = _load_partition(splits_dir / "train.csv", stats)
    if set(np.unique(train_labels)) != {0, 1}:
        raise ValueError("Training manifest must contain both classes.")

    models = build_models(seed)
    model_results: dict[str, dict[str, dict[str, float | int]]] = {}
    for name in model_names:
        model = models[name]
        print(f"Training {name} on {len(train_labels)} training images...")
        model.fit(train_features, train_labels)
        model_results[name] = {"train": _score_model(model, train_features, train_labels)}

    del train_features
    for partition in PARTITIONS[1:]:
        features, labels = _load_partition(splits_dir / f"{partition}.csv", stats)
        for name in model_names:
            model_results[name][partition] = _score_model(models[name], features, labels)
        del features

    result: dict[str, Any] = {
        "protocol": {
            "feature": "flattened 128x128 per-pixel standardized image",
            "fit_partition": "train",
            "validation_used_for_tuning": False,
            "seed": seed,
            "positive_class": "Pneumonia",
            "model_parameters": {
                "logistic_regression": {"max_iter": 1000, "solver": "lbfgs"},
                "linear_svm": {"C": 1.0, "max_iter": 5000},
                "random_forest": {
                    "n_estimators": 100,
                    "max_features": "sqrt",
                    "n_jobs": -1,
                },
            },
        },
        "models": model_results,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Metrics written to {output}")
    for name, partitions in model_results.items():
        test = partitions["test"]
        print(
            f"{name}: test accuracy={test['accuracy']:.4f}, "
            f"precision={test['precision']:.4f}, recall={test['recall']:.4f}, "
            f"F1={test['f1']:.4f}, ROC-AUC={test['roc_auc']:.4f}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--stats", help="Normalization archive (defaults to <splits-dir>/normalization.npz)")
    parser.add_argument("--output", default="results/baseline_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=MODEL_NAMES,
        help="Subset of baseline model names to run.",
    )
    args = parser.parse_args()
    run_baselines(
        splits_dir=args.splits_dir,
        stats_path=args.stats,
        output_path=args.output,
        seed=args.seed,
        model_names=tuple(args.models),
    )


if __name__ == "__main__":
    main()
