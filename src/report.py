"""Create comparison plots and a concise experiment summary from result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Result file does not exist: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {source}.")
    return value


def generate_report(
    baseline_path: str | Path = "results/baseline_metrics.json",
    reference_cnn_path: str | Path = "results/reference_cnn_metrics.json",
    densenet_path: str | Path = "results/densenet121_metrics.json",
    reference_localization_path: str | Path = "results/reference_cnn_localization.json",
    densenet_localization_path: str | Path = "results/densenet121_localization.json",
    output_dir: str | Path = "results",
) -> dict[str, Path]:
    """Generate classification/CAM plots and a Markdown summary."""
    baseline = _read_json(baseline_path)
    reference_cnn = _read_json(reference_cnn_path)
    densenet = _read_json(densenet_path)
    reference_localization = _read_json(reference_localization_path)
    densenet_localization = _read_json(densenet_localization_path)
    for name, report in (
        ("baseline", baseline),
        ("reference CNN", reference_cnn),
        ("DenseNet121", densenet),
    ):
        if not isinstance(report.get("models", report.get("metrics")), dict):
            raise ValueError(f"{name} result JSON does not contain split metrics.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    model_metrics: list[tuple[str, dict[str, Any]]] = []
    display_names = {
        "logistic_regression": "Logistic Regression",
        "linear_svm": "Linear SVM",
        "random_forest": "Random Forest",
    }
    for model_name in ("logistic_regression", "linear_svm", "random_forest"):
        metrics = baseline["models"].get(model_name, {}).get("test")
        if metrics is None:
            raise ValueError(f"Missing test results for {model_name}.")
        model_metrics.append((display_names[model_name], metrics))
    model_metrics.extend((
        ("Reference CNN", reference_cnn["metrics"]["test"]),
        ("DenseNet121", densenet["metrics"]["test"]),
    ))

    classification_plot = output / "classification_comparison.png"
    labels = [name for name, _ in model_metrics]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "accuracy", "Test accuracy"),
        (axes[1], "roc_auc", "Test ROC-AUC"),
    ):
        values = [float(metrics[metric]) for _, metrics in model_metrics]
        bars = axis.bar(x, values, color=plt.get_cmap("tab10").colors[:len(labels)])
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.set_ylabel("Score")
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(classification_plot, dpi=160)
    plt.close(figure)

    localization_plot = output / "cam_localization_comparison.png"
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    localizations = (
        ("Reference CNN", reference_localization),
        ("DenseNet121", densenet_localization),
    )
    for axis, metric_key, title in (
        (axes[0], "test_image_mean_best_iou", "Mean IoU, all test images"),
        (axes[1], "pneumonia_patient_mean_best_iou", "Mean IoU, pneumonia-only"),
    ):
        for name, report in localizations:
            threshold_data = report.get("thresholds")
            if not isinstance(threshold_data, dict):
                raise ValueError(f"Localization result for {name} has no thresholds.")
            sorted_items = sorted(
                ((float(key), metrics) for key, metrics in threshold_data.items()),
                key=lambda item: item[0],
            )
            axis.plot(
                [threshold for threshold, _ in sorted_items],
                [float(metrics[metric_key]) for _, metrics in sorted_items],
                marker="o",
                label=name,
            )
        axis.set_title(title)
        axis.set_xlabel("Normalized CAM threshold")
        axis.set_ylabel("IoU")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(localization_plot, dpi=160)
    plt.close(figure)

    summary_path = output / "experiment_summary.md"
    summary_lines = [
        "# Experiment summary",
        "",
        "All metrics below were produced locally from the available RSNA split.",
        "The test split was not used to choose hyperparameters or thresholds.",
        "",
        "## Classification on the test split",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in model_metrics:
        summary_lines.append(
            f"| {name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['roc_auc']:.4f} |"
        )
    summary_lines.extend((
        "",
        "## CAM localization on the test split",
        "",
        "| Model | Threshold | All-image mean IoU | Pneumonia-only mean IoU |",
        "|---|---:|---:|---:|",
    ))
    for name, report in localizations:
        for threshold, metrics in sorted(
            report["thresholds"].items(), key=lambda item: float(item[0])
        ):
            summary_lines.append(
                f"| {name} | {float(threshold):.2f} | "
                f"{metrics['test_image_mean_best_iou']:.4f} | "
                f"{metrics['pneumonia_patient_mean_best_iou']:.4f} |"
            )
    summary_lines.extend((
        "",
        "## Important limitations",
        "",
        "- The available copy contains 14,863 included records, not the paper's reported 17,489.",
        "- The reference CNN uses original images only; no lung masks or U-Net weights were available.",
        "- DenseNet121 uses ImageNet-pretrained frozen features and a trained binary head.",
        "- CAM thresholds and the IoU aggregation are documented implementation choices, not claimed exact reproductions.",
        "- The paper's supervised R-CNN is not implemented because its full architecture and training recipe are underspecified.",
        "- Logistic Regression and linear SVM reported convergence warnings; interpret their scores as preliminary.",
        "",
        "Plots: `classification_comparison.png` and `cam_localization_comparison.png`.",
        "",
    ))
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    return {
        "classification_plot": classification_plot,
        "localization_plot": localization_plot,
        "summary": summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="results/baseline_metrics.json")
    parser.add_argument("--reference-cnn", default="results/reference_cnn_metrics.json")
    parser.add_argument("--densenet", default="results/densenet121_metrics.json")
    parser.add_argument(
        "--reference-localization",
        default="results/reference_cnn_localization.json",
    )
    parser.add_argument(
        "--densenet-localization",
        default="results/densenet121_localization.json",
    )
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    outputs = generate_report(
        baseline_path=args.baseline,
        reference_cnn_path=args.reference_cnn,
        densenet_path=args.densenet,
        reference_localization_path=args.reference_localization,
        densenet_localization_path=args.densenet_localization,
        output_dir=args.output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
