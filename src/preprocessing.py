"""DICOM loading, resizing, and intensity normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class NormalizationStats:
    """Per-pixel training-set means and standard deviations."""

    mean: np.ndarray
    std: np.ndarray


def load_dicom_pixels(
    path: str | Path,
    image_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Read, orient, and resize one DICOM without normalizing its intensities."""
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ImportError(
            "Reading DICOM images requires pydicom. Install project dependencies "
            "with `python -m pip install -r requirements.txt`."
        ) from exc

    dicom_path = Path(path)
    if not dicom_path.is_file():
        raise FileNotFoundError(f"DICOM image does not exist: {dicom_path}")
    dataset = pydicom.dcmread(dicom_path)
    pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
    if pixels.ndim != 2:
        raise ValueError(
            f"Expected a single-channel 2D DICOM image at {dicom_path}; "
            f"found shape {pixels.shape}."
        )
    if not np.isfinite(pixels).all():
        raise ValueError(f"DICOM image contains non-finite pixel values: {dicom_path}")
    if str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
        pixels = pixels.max() - pixels

    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must contain positive dimensions, got {image_size}.")
    resized = Image.fromarray(pixels).resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def compute_normalization_stats(
    image_paths: Iterable[str | Path],
    image_size: tuple[int, int] = (128, 128),
) -> NormalizationStats:
    """Compute per-pixel population mean/std over training images only.

    Uses an online update so it does not retain the full image set in memory.
    The reference specifies pixel-location mean/std normalization but not the
    numerical convention for standard deviation; population std (ddof=0) is
    used here as an implementation choice.
    """
    count = 0
    mean = np.zeros((image_size[1], image_size[0]), dtype=np.float64)
    m2 = np.zeros_like(mean)
    for path in image_paths:
        pixels = load_dicom_pixels(path, image_size=image_size).astype(np.float64)
        count += 1
        delta = pixels - mean
        mean += delta / count
        m2 += delta * (pixels - mean)
    if count == 0:
        raise ValueError("Cannot compute normalization statistics from an empty image set.")
    std = np.sqrt(m2 / count)
    return NormalizationStats(mean.astype(np.float32), std.astype(np.float32))


def save_normalization_stats(stats: NormalizationStats, path: str | Path) -> None:
    """Save per-pixel statistics to a compressed NumPy archive."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, mean=stats.mean, std=stats.std)


def load_normalization_stats(path: str | Path) -> NormalizationStats:
    """Load statistics saved by :func:`save_normalization_stats`."""
    with np.load(path) as archive:
        return NormalizationStats(archive["mean"].astype(np.float32), archive["std"].astype(np.float32))


def load_dicom_image(
    path: str | Path,
    stats: NormalizationStats,
    image_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Return a resized DICOM standardized using per-pixel training statistics.

    The reference describes subtracting the mean and dividing by the standard
    deviation at each pixel location. Statistics must be computed from the
    training split and reused for validation, test, and inference images.
    """
    pixels = load_dicom_pixels(path, image_size=image_size)
    if stats.mean.shape != pixels.shape or stats.std.shape != pixels.shape:
        raise ValueError(
            f"Normalization statistics have shapes {stats.mean.shape} and {stats.std.shape}; "
            f"expected {pixels.shape}."
        )
    if not np.isfinite(stats.mean).all() or not np.isfinite(stats.std).all():
        raise ValueError("Normalization statistics contain non-finite values.")
    if np.any(stats.std < 0):
        raise ValueError("Normalization standard deviations must not be negative.")
    # A zero std means that location is constant in the training set. Keeping
    # its centered value avoids division by zero; using 1 is a numerical choice.
    safe_std = np.where(stats.std > 0, stats.std, 1.0)
    return ((pixels - stats.mean) / safe_std).astype(np.float32)
