"""Train a DenseNet121 classifier for the project comparison."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .baselines import classification_metrics
from .preprocessing import load_normalization_stats
from .reference_cnn import _load_partition


def build_densenet121(pretrained: bool = False) -> nn.Module:
    """Build DenseNet121 with a single-channel input and a binary output head."""
    from torchvision.models import densenet121

    from torchvision.models import DenseNet121_Weights

    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)
    original = model.features.conv0
    grayscale = nn.Conv2d(
        1,
        original.out_channels,
        kernel_size=original.kernel_size,
        stride=original.stride,
        padding=original.padding,
        bias=original.bias is not None,
    )
    if pretrained:
        with torch.no_grad():
            grayscale.weight.copy_(original.weight.mean(dim=1, keepdim=True))
            if original.bias is not None and grayscale.bias is not None:
                grayscale.bias.copy_(original.bias)
    model.features.conv0 = grayscale
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model


def _extract_features(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Extract and retain only pooled DenseNet features for efficient CPU training."""
    features: list[torch.Tensor] = []
    loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)
    model.eval()
    with torch.inference_mode():
        for (batch_images,) in loader:
            activations = F.relu(model.features(batch_images.to(device)), inplace=False)
            pooled = F.adaptive_avg_pool2d(activations, (1, 1)).flatten(start_dim=1)
            features.append(pooled.cpu())
    return torch.cat(features)


def _score_head(
    head: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    scores: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=False)
    head.eval()
    with torch.inference_mode():
        for batch_features, _ in loader:
            logits = head(batch_features.to(device))
            scores.append(logits.softmax(dim=1)[:, 1].cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return classification_metrics(labels.numpy(), np.concatenate(predictions), np.concatenate(scores))


def run_densenet121(
    splits_dir: str | Path = "data/splits",
    stats_path: str | Path | None = None,
    output_path: str | Path = "results/densenet121_metrics.json",
    checkpoint_path: str | Path = "results/densenet121.pt",
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train and evaluate DenseNet121 using original single-channel images."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")
    splits_dir = Path(splits_dir)
    stats = load_normalization_stats(
        Path(stats_path) if stats_path else splits_dir / "normalization.npz"
    )
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
    if device.type == "cpu" and torch.get_num_threads() > 8:
        torch.set_num_threads(8)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

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

    model = build_densenet121(pretrained=True).to(device)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    print("Extracting frozen pretrained DenseNet121 features...")
    train_features = _extract_features(model, train_images, device, batch_size)
    validation_features = _extract_features(model, validation_images, device, batch_size)
    test_features = _extract_features(model, test_images, device, batch_size)
    del train_images, validation_images, test_images

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.classifier.train()
        total_loss = 0.0
        examples = 0
        for features, labels in loader:
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model.classifier(features.to(device)), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            examples += labels.size(0)
        mean_loss = total_loss / examples
        validation = _score_head(
            model.classifier, validation_features, validation_labels, batch_size, device
        )
        history.append({
            "epoch": epoch,
            "train_loss": mean_loss,
            "validation_accuracy": float(validation["accuracy"]),
        })
        print(
            f"epoch {epoch:02d}/{epochs}: loss={mean_loss:.4f}, "
            f"validation accuracy={validation['accuracy']:.4f}"
        )

    metrics = {
        "train": _score_head(
            model.classifier, train_features, train_labels, batch_size, device
        ),
        "validation": _score_head(
            model.classifier, validation_features, validation_labels, batch_size, device
        ),
        "test": _score_head(model.classifier, test_features, test_labels, batch_size, device),
    }
    result: dict[str, Any] = {
        "protocol": {
            "architecture": "torchvision DenseNet121",
            "weights": "ImageNet pretrained; frozen feature extractor with trained binary head",
            "input": "original standardized grayscale image",
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
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
    torch.save({
        "model_state_dict": model.cpu().state_dict(),
        "protocol": result["protocol"],
        "metrics": metrics,
    }, checkpoint)
    print(f"Metrics written to {output}")
    print(f"Checkpoint written to {checkpoint}")
    print(
        f"DenseNet121 test accuracy={metrics['test']['accuracy']:.4f}, "
        f"ROC-AUC={metrics['test']['roc_auc']:.4f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--stats", help="Normalization archive (defaults to <splits-dir>/normalization.npz)")
    parser.add_argument("--output", default="results/densenet121_metrics.json")
    parser.add_argument("--checkpoint", default="results/densenet121.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_densenet121(
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
