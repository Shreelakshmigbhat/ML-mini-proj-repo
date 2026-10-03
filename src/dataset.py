"""RSNA Pneumonia Detection metadata loading and reproducible data splitting."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.model_selection import train_test_split

from .preprocessing import (
    NormalizationStats,
    compute_normalization_stats,
    load_dicom_image,
    save_normalization_stats,
)


@dataclass(frozen=True)
class Box:
    """Ground-truth bounding box in the source DICOM pixel coordinate system."""

    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class XrayRecord:
    patient_id: str
    image_path: Path
    label: int  # 1 = Pneumonia, 0 = Healthy
    class_name: str
    boxes: tuple[Box, ...]


CLASS_TO_LABEL = {
    "lung opacity": ("Pneumonia", 1),
    "pneumonia": ("Pneumonia", 1),
    "normal": ("Healthy", 0),
}
EXCLUDED_CLASSES = {"no lung opacity / not normal", "diseased/no pneumonia"}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")
        return list(reader)


def _required(row: dict[str, str], key: str, source: str) -> str:
    value = row.get(key)
    if value is None or not value.strip():
        raise ValueError(f"Missing required column/value {key!r} in {source}.")
    return value.strip()


def _optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)


def _index_images(image_dir: str | Path) -> dict[str, Path]:
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"DICOM image directory does not exist: {root}")
    paths = sorted(root.rglob("*.dcm"))
    if not paths:
        raise FileNotFoundError(f"No .dcm images found under {root}")
    result: dict[str, Path] = {}
    for path in paths:
        if path.stem in result:
            raise ValueError(
                f"More than one DICOM has patient ID {path.stem!r}; "
                "place one copy per patient in the image directory."
            )
        result[path.stem] = path.resolve()
    return result


def load_rsna_records(
    labels_csv: str | Path,
    class_info_csv: str | Path,
    image_dir: str | Path,
) -> tuple[list[XrayRecord], int]:
    """Load labeled records, retaining pneumonia boxes and excluding other disease.

    Expects the RSNA Stage 2 ``train_labels.csv`` columns ``patientId, Target,
    x, y, width, height`` and ``stage_2_detailed_class_info.csv`` columns
    ``patientId, class``. The detailed class file is required to distinguish
    healthy negatives from ``No Lung Opacity / Not Normal`` negatives.

    Returns ``(records, excluded_count)``. The excluded count is the number of
    unique patients removed because their detailed class is an excluded class.
    """
    label_rows = _read_csv(labels_csv)
    class_rows = _read_csv(class_info_csv)
    labels_path, classes_path = str(labels_csv), str(class_info_csv)
    images = _index_images(image_dir)

    class_by_patient: dict[str, str] = {}
    for row in class_rows:
        patient_id = _required(row, "patientId", classes_path)
        class_name = _required(row, "class", classes_path)
        previous = class_by_patient.setdefault(patient_id, class_name)
        if previous != class_name:
            raise ValueError(f"Conflicting detailed classes for patient {patient_id}.")

    target_by_patient: dict[str, int] = {}
    boxes_by_patient: dict[str, list[Box]] = {}
    for row in label_rows:
        patient_id = _required(row, "patientId", labels_path)
        try:
            raw_target = float(_required(row, "Target", labels_path))
        except ValueError as exc:
            raise ValueError(f"Target must be 0 or 1 for patient {patient_id}.") from exc
        if not np.isfinite(raw_target) or raw_target not in (0, 1):
            raise ValueError(
                f"Target must be exactly 0 or 1 for patient {patient_id}; got {raw_target}."
            )
        target = int(raw_target)
        previous_target = target_by_patient.setdefault(patient_id, target)
        if previous_target != target:
            raise ValueError(f"Conflicting Target values for patient {patient_id}.")

        values = [_optional_float(row.get(name)) for name in ("x", "y", "width", "height")]
        if target == 1:
            if any(value is None for value in values):
                raise ValueError(f"Pneumonia patient {patient_id} has a row without a full box.")
            x, y, width, height = (float(value) for value in values)  # type: ignore[arg-type]
            if not np.isfinite((x, y, width, height)).all():
                raise ValueError(f"Pneumonia patient {patient_id} has a non-finite box value.")
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ValueError(
                    f"Pneumonia patient {patient_id} has a negative coordinate or "
                    "non-positive box size."
                )
            boxes_by_patient.setdefault(patient_id, []).append(Box(x, y, width, height))
        else:
            boxes_by_patient.setdefault(patient_id, [])

    records: list[XrayRecord] = []
    excluded: set[str] = set()
    for patient_id, target in target_by_patient.items():
        if patient_id not in class_by_patient:
            raise ValueError(f"No detailed class found for patient {patient_id}.")
        source_class = class_by_patient[patient_id]
        normalized_class = source_class.strip().casefold()
        if normalized_class in EXCLUDED_CLASSES:
            excluded.add(patient_id)
            continue
        if normalized_class not in CLASS_TO_LABEL:
            raise ValueError(
                f"Unrecognized detailed class {source_class!r} for {patient_id}. "
                "Update the explicit class mapping rather than inferring a label."
            )
        class_name, mapped_label = CLASS_TO_LABEL[normalized_class]
        if mapped_label != target:
            raise ValueError(
                f"Detailed class {source_class!r} conflicts with Target={target} "
                f"for patient {patient_id}."
            )
        if patient_id not in images:
            raise FileNotFoundError(f"No DICOM image found for patient {patient_id} under {image_dir}.")
        boxes = tuple(boxes_by_patient.get(patient_id, ()))
        if target == 1 and not boxes:
            raise ValueError(f"Pneumonia patient {patient_id} has no ground-truth box.")
        records.append(XrayRecord(patient_id, images[patient_id], target, class_name, boxes))

    if not records:
        raise ValueError("No Pneumonia or Healthy records remain after filtering.")
    return records, len(excluded)


def split_records(
    records: Iterable[XrayRecord], seed: int = 42
) -> tuple[list[XrayRecord], list[XrayRecord], list[XrayRecord]]:
    """Create stratified 70/20/10 train, validation, and test partitions."""
    records = list(records)
    if len(records) < 10:
        raise ValueError("At least 10 records are needed for a stratified 70/20/10 split.")
    indices = np.arange(len(records))
    labels = np.asarray([record.label for record in records])
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.10, random_state=seed, stratify=labels
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=(0.20 / 0.90),
        random_state=seed,
        stratify=labels[train_val_idx],
    )
    return (
        [records[index] for index in train_idx],
        [records[index] for index in val_idx],
        [records[index] for index in test_idx],
    )


def load_image(
    record: XrayRecord,
    stats: NormalizationStats,
    image_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Load and preprocess one record using training-split normalization stats."""
    return load_dicom_image(record.image_path, stats=stats, image_size=image_size)


def visualize_record(
    record: XrayRecord,
    stats: NormalizationStats,
    image_size: tuple[int, int] = (128, 128),
):
    """Display a resized image, its class label, and source-coordinate boxes."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    image = load_image(record, stats=stats, image_size=image_size)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray")
    ax.set_title(f"{record.patient_id}: {record.class_name}")
    ax.axis("off")

    # Scale the original annotation coordinates to the resized image using its
    # actual DICOM dimensions, which need not be exactly 1024 x 1024.
    import pydicom

    ds = pydicom.dcmread(record.image_path, stop_before_pixels=True)
    scale_x, scale_y = image_size[0] / float(ds.Columns), image_size[1] / float(ds.Rows)
    for box in record.boxes:
        ax.add_patch(Rectangle(
            (box.x * scale_x, box.y * scale_y),
            box.width * scale_x,
            box.height * scale_y,
            fill=False,
            edgecolor="red",
            linewidth=1.5,
        ))
    fig.tight_layout()
    return fig, ax


def _write_split(path: Path, records: list[XrayRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("patient_id", "image_path", "label", "class_name", "boxes_json")
        )
        writer.writeheader()
        for record in records:
            writer.writerow({
                "patient_id": record.patient_id,
                "image_path": str(record.image_path),
                "label": record.label,
                "class_name": record.class_name,
                "boxes_json": json.dumps([asdict(box) for box in record.boxes]),
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", required=True, help="Path to RSNA train_labels.csv")
    parser.add_argument("--class-info-csv", required=True, help="Path to detailed class info CSV")
    parser.add_argument("--image-dir", required=True, help="Directory containing patient DICOMs")
    parser.add_argument("--output-dir", default="data/splits", help="Directory for split CSV files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (implementation choice)")
    parser.add_argument(
        "--compute-normalization-stats",
        action="store_true",
        help="Compute per-pixel mean/std from training images and save normalization.npz",
    )
    args = parser.parse_args()

    print("Phase 1 data preparation configuration")
    print(f"  labels CSV: {Path(args.labels_csv).resolve()}")
    print(f"  class info CSV: {Path(args.class_info_csv).resolve()}")
    print(f"  DICOM directory: {Path(args.image_dir).resolve()}")
    print("  image size: 128 x 128")
    print("  normalization: per-pixel training-set mean/std standardization")
    print("  split: stratified 70/20/10")
    print(f"  random seed: {args.seed}")

    records, excluded = load_rsna_records(args.labels_csv, args.class_info_csv, args.image_dir)
    train, validation, test = split_records(records, seed=args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, partition in (("train", train), ("validation", validation), ("test", test)):
        _write_split(output_dir / f"{name}.csv", partition)
        print(f"  {name}: {len(partition)} records")
    if args.compute_normalization_stats:
        print("  computing per-pixel mean/std from training images only")
        stats = compute_normalization_stats(record.image_path for record in train)
        save_normalization_stats(stats, output_dir / "normalization.npz")
        print(f"  normalization statistics: {output_dir / 'normalization.npz'}")
    print(f"  excluded non-target disease: {excluded} records")
    print(f"  included records: {len(records)}")


if __name__ == "__main__":
    main()
