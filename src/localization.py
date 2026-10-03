"""Evaluate class activation maps against RSNA bounding-box annotations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pydicom
import torch
import torch.nn.functional as F

from .densenet121 import build_densenet121
from .preprocessing import load_dicom_image, load_normalization_stats
from .reference_cnn import ReferenceCNN


def compute_cam(
    model: torch.nn.Module,
    image: torch.Tensor,
    class_index: int = 1,
) -> tuple[np.ndarray, float]:
    """Return a normalized CAM resized to input shape and its class probability."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 1:
        raise ValueError("image must have shape (1, height, width) or (1, 1, height, width).")
    features = model.features(image)
    if hasattr(model, "pool") and isinstance(model, ReferenceCNN):
        activation = features
        logits = model.classifier(model.pool(features).flatten(start_dim=1))
    else:
        activation = F.relu(features, inplace=False)
        logits = model.classifier(F.adaptive_avg_pool2d(activation, (1, 1)).flatten(start_dim=1))
    if not 0 <= class_index < logits.shape[1]:
        raise ValueError(f"class_index must be between 0 and {logits.shape[1] - 1}.")

    weights = model.classifier.weight[class_index].view(1, -1, 1, 1)
    cam = F.relu((activation * weights).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    maximum = cam.max()
    if maximum.item() > 0:
        cam = cam / maximum
    else:
        cam = torch.zeros_like(cam)
    probability = torch.softmax(logits, dim=1)[0, class_index].item()
    return cam.detach().cpu().numpy().astype(np.float32), float(probability)


def connected_component_boxes(
    heatmap: np.ndarray,
    threshold: float,
) -> list[tuple[int, int, int, int]]:
    """Find 8-connected CAM regions and return (x, y, width, height) boxes."""
    heatmap = np.asarray(heatmap)
    if heatmap.ndim != 2:
        raise ValueError("heatmap must be a two-dimensional array.")
    if not np.isfinite(heatmap).all():
        raise ValueError("heatmap must contain only finite values.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    active = (heatmap >= threshold) & (heatmap > 0)
    visited = np.zeros(active.shape, dtype=bool)
    height, width = active.shape
    boxes: list[tuple[int, int, int, int]] = []
    for row, column in zip(*np.nonzero(active)):
        if visited[row, column]:
            continue
        queue = deque([(row, column)])
        visited[row, column] = True
        min_row = max_row = row
        min_column = max_column = column
        while queue:
            current_row, current_column = queue.popleft()
            min_row = min(min_row, current_row)
            max_row = max(max_row, current_row)
            min_column = min(min_column, current_column)
            max_column = max(max_column, current_column)
            for row_offset in (-1, 0, 1):
                for column_offset in (-1, 0, 1):
                    next_row = current_row + row_offset
                    next_column = current_column + column_offset
                    if (
                        0 <= next_row < height
                        and 0 <= next_column < width
                        and active[next_row, next_column]
                        and not visited[next_row, next_column]
                    ):
                        visited[next_row, next_column] = True
                        queue.append((next_row, next_column))
        boxes.append((
            int(min_column),
            int(min_row),
            int(max_column - min_column + 1),
            int(max_row - min_row + 1),
        ))
    return boxes


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Compute IoU for two (x, y, width, height) boxes."""
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[0] + first[2], second[0] + second[2])
    y2 = min(first[1] + first[3], second[1] + second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 0 else 0.0


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"patient_id", "image_path", "label", "boxes_json"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Split manifest {path} must contain columns "
                f"{', '.join(sorted(required))}."
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Split manifest is empty: {path}")
    return rows


def _load_checkpoint(model_name: str, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint_path}")
    if model_name == "reference_cnn":
        model: torch.nn.Module = ReferenceCNN()
    elif model_name == "densenet121":
        model = build_densenet121()
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no model_state_dict: {checkpoint_path}")
    model.load_state_dict(state)
    return model.to(device).eval()


def evaluate_localization(
    model_name: str,
    checkpoint_path: str | Path,
    splits_dir: str | Path = "data/splits",
    stats_path: str | Path | None = None,
    output_path: str | Path | None = None,
    thresholds: Iterable[float] = (0.2, 0.45, 0.7),
    prediction_cutoff: float = 0.5,
    device_name: str = "auto",
    visualization_dir: str | Path | None = None,
    visualization_count: int = 5,
    visualization_threshold: float = 0.7,
) -> dict[str, Any]:
    """Evaluate thresholded CAM boxes over the test manifest.

    Reports image-level max IoU for every test image (including healthy and
    missed pneumonia cases, scored as zero), plus a positive-patient-only mean.
    """
    threshold_values = tuple(float(value) for value in thresholds)
    if not threshold_values or any(not 0 <= value <= 1 for value in threshold_values):
        raise ValueError("thresholds must be a non-empty sequence in [0, 1].")
    if not 0 <= prediction_cutoff <= 1:
        raise ValueError("prediction_cutoff must be between 0 and 1.")
    if visualization_count < 0 or not 0 <= visualization_threshold <= 1:
        raise ValueError("visualization_count must be non-negative and its threshold in [0, 1].")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    if device.type == "cpu" and torch.get_num_threads() > 8:
        torch.set_num_threads(8)

    root = Path(splits_dir)
    stats_file = Path(stats_path) if stats_path else root / "normalization.npz"
    stats = load_normalization_stats(stats_file)
    model = _load_checkpoint(model_name, Path(checkpoint_path), device)
    rows = _read_manifest(root / "test.csv")
    results = {
        f"{threshold:.2f}": {
            "image_level_ious": [],
            "positive_patient_ious": [],
            "predicted_positive_count": 0,
            "detected_box_count": 0,
        }
        for threshold in threshold_values
    }
    visualization_path = Path(visualization_dir) if visualization_dir else None
    if visualization_path and visualization_count:
        visualization_path.mkdir(parents=True, exist_ok=True)
    saved_visualizations = 0

    for index, row in enumerate(rows, start=1):
        patient_id = row["patient_id"].strip()
        image_path = Path(row["image_path"])
        image = load_dicom_image(image_path, stats)
        image_tensor = torch.from_numpy(image[None, None]).to(device)
        try:
            label = int(row["label"])
            annotations = json.loads(row["boxes_json"])
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid label or boxes for {patient_id!r} in test manifest.") from exc
        if label not in (0, 1) or not isinstance(annotations, list):
            raise ValueError(f"Invalid label or boxes for {patient_id!r} in test manifest.")
        ds = pydicom.dcmread(image_path, stop_before_pixels=True)
        scale_x = image.shape[1] / float(ds.Columns)
        scale_y = image.shape[0] / float(ds.Rows)
        truth_boxes = [
            (
                float(box["x"]) * scale_x,
                float(box["y"]) * scale_y,
                float(box["width"]) * scale_x,
                float(box["height"]) * scale_y,
            )
            for box in annotations
        ]
        if label == 1 and not truth_boxes:
            raise ValueError(f"Pneumonia record {patient_id!r} has no annotation boxes.")

        with torch.inference_mode():
            heatmap, pneumonia_probability = compute_cam(model, image_tensor)
        predicted_positive = pneumonia_probability >= prediction_cutoff
        display_threshold_boxes = (
            connected_component_boxes(heatmap, visualization_threshold)
            if predicted_positive
            else []
        )
        for key, metric in results.items():
            threshold = float(key)
            predicted_boxes = (
                connected_component_boxes(heatmap, threshold)
                if predicted_positive
                else []
            )
            image_iou = max(
                (
                    box_iou(predicted, truth)
                    for predicted in predicted_boxes
                    for truth in truth_boxes
                ),
                default=0.0,
            )
            metric["image_level_ious"].append(image_iou)
            if label == 1:
                metric["positive_patient_ious"].append(image_iou)
            metric["predicted_positive_count"] += int(predicted_positive)
            metric["detected_box_count"] += len(predicted_boxes)
        if (
            visualization_path is not None
            and saved_visualizations < visualization_count
            and label == 1
            and predicted_positive
        ):
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle

            figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
            axes[0].imshow(image, cmap="gray")
            axes[0].set_title("Standardized image")
            axes[1].imshow(image, cmap="gray")
            axes[1].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            axes[1].set_title(f"CAM (p={pneumonia_probability:.2f})")
            axes[2].imshow(image, cmap="gray")
            for box in truth_boxes:
                axes[2].add_patch(Rectangle(
                    (box[0], box[1]),
                    box[2],
                    box[3],
                    fill=False,
                    edgecolor="lime",
                    linewidth=1.5,
                    label="Ground truth",
                ))
            for box in display_threshold_boxes:
                axes[2].add_patch(Rectangle(
                    (box[0], box[1]),
                    box[2],
                    box[3],
                    fill=False,
                    edgecolor="red",
                    linewidth=1.0,
                    label="CAM",
                ))
            axes[2].set_title(f"Boxes (threshold={visualization_threshold:.2f})")
            for axis in axes:
                axis.axis("off")
            figure.savefig(
                visualization_path / f"{model_name}_{patient_id}.png",
                dpi=140,
            )
            plt.close(figure)
            saved_visualizations += 1
        if index % 100 == 0 or index == len(rows):
            print(f"Localized {index}/{len(rows)} test images")

    threshold_results: dict[str, Any] = {}
    for key, metric in results.items():
        threshold_results[key] = {
            "test_image_mean_best_iou": float(np.mean(metric["image_level_ious"])),
            "pneumonia_patient_mean_best_iou": float(np.mean(metric["positive_patient_ious"])),
            "test_images": len(metric["image_level_ious"]),
            "pneumonia_patients": len(metric["positive_patient_ious"]),
            "predicted_positive_count": metric["predicted_positive_count"],
            "detected_box_count": metric["detected_box_count"],
        }
    result: dict[str, Any] = {
        "protocol": {
            "model": model_name,
            "split": "test",
            "prediction_cutoff": prediction_cutoff,
            "cam_thresholds": list(threshold_values),
            "connected_components": "8-connected",
            "iou_aggregation": "maximum overlap over predicted components and ground-truth boxes per image",
            "no_detection_iou": 0.0,
            "mean_metrics_include_healthy_cases": True,
            "mean_metrics_include_false_negative_classifications": True,
        },
        "thresholds": threshold_results,
        "visualizations_saved": saved_visualizations,
    }
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Localization metrics written to {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("reference_cnn", "densenet121"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--stats")
    parser.add_argument("--output")
    parser.add_argument("--thresholds", nargs="+", type=float, default=(0.2, 0.45, 0.7))
    parser.add_argument("--prediction-cutoff", type=float, default=0.5)
    parser.add_argument("--visualization-dir")
    parser.add_argument("--visualization-count", type=int, default=5)
    parser.add_argument("--visualization-threshold", type=float, default=0.7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evaluate_localization(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        splits_dir=args.splits_dir,
        stats_path=args.stats,
        output_path=args.output,
        thresholds=args.thresholds,
        prediction_cutoff=args.prediction_cutoff,
        device_name=args.device,
        visualization_dir=args.visualization_dir,
        visualization_count=args.visualization_count,
        visualization_threshold=args.visualization_threshold,
    )


if __name__ == "__main__":
    main()
