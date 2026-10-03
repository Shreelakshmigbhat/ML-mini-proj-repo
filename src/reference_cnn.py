"""Train the paper-inspired CNN on the original RSNA images."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .baselines import classification_metrics
from .preprocessing import (
    NormalizationStats,
    load_dicom_image,
    load_normalization_stats,
)

DEFAULT_CHANNELS = (8, 8, 16, 16, 32, 32, 64, 64, 64, 64)


class ReferenceCNN(nn.Module):
    """Ten 3x3 convolution layers, ReLU, global average pooling, and classifier."""

    def __init__(self, channels: tuple[int, ...] = DEFAULT_CHANNELS) -> None:
        super().__init__()
        if len(channels) != 10 or any(width < 1 for width in channels):
            raise ValueError("channels must contain exactly 10 positive widths.")

        layers: list[nn.Module] = []
        input_channels = 1
        for index, output_channels in enumerate(channels):
            layers.extend((
                nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ))
            if index in (0, 2, 4, 6):
                layers.append(nn.MaxPool2d(kernel_size=2))
            input_channels = output_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels[-1], 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("Expected input images with shape (batch, 1, height, width).")
        features = self.features(images)
        pooled = self.pool(features).flatten(start_dim=1)
        return self.classifier(pooled)


def _load_partition(
    manifest_path: str | Path,
    stats: NormalizationStats,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode standardized DICOM images from one split manifest into memory."""
    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {manifest}")
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"patient_id", "image_path", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Split manifest {manifest} must contain columns "
                f"{', '.join(sorted(required))}."
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Split manifest is empty: {manifest}")

    images = np.empty((len(rows), 1, *stats.mean.shape), dtype=np.float32)
    labels = np.empty(len(rows), dtype=np.int64)
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        patient_id = row["patient_id"].strip()
        if not patient_id or patient_id in seen_ids:
            raise ValueError(f"Missing or duplicate patient_id in {manifest}, row {index + 2}.")
        seen_ids.add(patient_id)
        try:
            label = float(row["label"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid label for patient {patient_id!r} in {manifest}: {row['label']!r}."
            ) from exc
        if not np.isfinite(label) or label not in (0, 1):
            raise ValueError(f"Label must be exactly 0 or 1 for patient {patient_id!r}.")
        image_path = row["image_path"].strip()
        if not image_path:
            raise ValueError(f"Empty image_path for patient {patient_id!r} in {manifest}.")
        image = load_dicom_image(image_path, stats=stats)
        images[index, 0] = image
        labels[index] = int(label)

    if np.unique(labels).size != 2:
        raise ValueError(f"Split manifest must contain both classes: {manifest}")
    return torch.from_numpy(images), torch.from_numpy(labels)


def _score(
    model: ReferenceCNN,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    loader = DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=False)
    model.eval()
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_images, _ in loader:
            logits = model(batch_images.to(device))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            probabilities.append(logits.softmax(dim=1)[:, 1].cpu().numpy())
    return classification_metrics(
        labels.numpy(),
        np.concatenate(predictions),
        np.concatenate(probabilities),
    )


def _train_epoch(
    model: ReferenceCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    example_count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(images), labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        example_count += batch_size
    if not example_count:
        raise ValueError("Cannot train on an empty training split.")
    return total_loss / example_count


def run_reference_cnn(
    splits_dir: str | Path = "data/splits",
    stats_path: str | Path | None = None,
    output_path: str | Path = "results/reference_cnn_metrics.json",
    checkpoint_path: str | Path = "results/reference_cnn.pt",
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train the reference CNN on original images and report all split metrics."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")
    splits_dir = Path(splits_dir)
    stats_file = Path(stats_path) if stats_path else splits_dir / "normalization.npz"
    stats = load_normalization_stats(stats_file)
    if not np.isfinite(stats.mean).all() or not np.isfinite(stats.std).all():
        raise ValueError("Normalization statistics contain non-finite values.")
    if np.any(stats.std < 0):
        raise ValueError("Normalization standard deviations must not be negative.")

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif torch.get_num_threads() > 8:
        torch.set_num_threads(8)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Loading normalized images into memory...")
    train_images, train_labels = _load_partition(splits_dir / "train.csv", stats)
    validation_images, validation_labels = _load_partition(
        splits_dir / "validation.csv", stats
    )
    test_images, test_labels = _load_partition(splits_dir / "test.csv", stats)
    print(
        f"Device: {device}; train={len(train_labels)}, "
        f"validation={len(validation_labels)}, test={len(test_labels)}"
    )

    model = ReferenceCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train_images, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        loss = _train_epoch(model, train_loader, optimizer, loss_function, device)
        validation = _score(
            model, validation_images, validation_labels, device, batch_size
        )
        history.append({"epoch": epoch, "train_loss": loss, "validation_accuracy": float(validation["accuracy"])})
        print(
            f"epoch {epoch:02d}/{epochs}: loss={loss:.4f}, "
            f"validation accuracy={validation['accuracy']:.4f}"
        )

    metrics = {
        "train": _score(model, train_images, train_labels, device, batch_size),
        "validation": _score(model, validation_images, validation_labels, device, batch_size),
        "test": _score(model, test_images, test_labels, device, batch_size),
    }
    result: dict[str, Any] = {
        "protocol": {
            "architecture": "10 zero-padded 3x3 convolutions with ReLU, spatial downsampling, global average pooling, and 2-class linear head",
            "input": "original image only; paper-required segmented image is omitted",
            "image_size": list(stats.mean.shape),
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "channel_widths": list(DEFAULT_CHANNELS),
            "seed": seed,
            "device": str(device),
            "validation_used_for_tuning": False,
        },
        "history": history,
        "metrics": metrics,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "protocol": result["protocol"],
            "metrics": metrics,
        },
        checkpoint,
    )
    print(f"Metrics written to {output}")
    print(f"Checkpoint written to {checkpoint}")
    print(f"CNN test accuracy={metrics['test']['accuracy']:.4f}, ROC-AUC={metrics['test']['roc_auc']:.4f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--stats", help="Normalization archive (defaults to <splits-dir>/normalization.npz)")
    parser.add_argument("--output", default="results/reference_cnn_metrics.json")
    parser.add_argument("--checkpoint", default="results/reference_cnn.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a PyTorch device such as cuda")
    args = parser.parse_args()
    run_reference_cnn(
        splits_dir=args.splits_dir,
        stats_path=args.stats,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
