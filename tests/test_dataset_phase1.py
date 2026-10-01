"""Smoke test the Phase 1 loader with generated local DICOMs (no RSNA download)."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from src.dataset import load_image, load_rsna_records, split_records
from src.preprocessing import compute_normalization_stats, load_normalization_stats, save_normalization_stats


class DatasetSmokeTest(unittest.TestCase):
    def test_metadata_boxes_filtering_split_and_image_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_dir = root / "images"
            image_dir.mkdir()
            label_rows = []
            class_rows = []
            cases = [(f"pos-{i:02d}", "Lung Opacity", 1) for i in range(10)]
            cases += [(f"normal-{i:02d}", "Normal", 0) for i in range(10)]
            cases += [("other-disease", "No Lung Opacity / Not Normal", 0)]

            for sample_index, (patient_id, class_name, target) in enumerate(cases):
                pixels = (
                    np.arange(32 * 24, dtype=np.uint16).reshape(32, 24)
                    + 10
                    + sample_index * 50
                )
                meta = FileMetaDataset()
                meta.TransferSyntaxUID = ExplicitVRLittleEndian
                meta.MediaStorageSOPClassUID = generate_uid()
                meta.MediaStorageSOPInstanceUID = generate_uid()
                meta.ImplementationClassUID = generate_uid()
                path = image_dir / f"{patient_id}.dcm"
                dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
                dataset.PatientName = patient_id
                dataset.PatientID = patient_id
                dataset.Rows, dataset.Columns = pixels.shape
                dataset.SamplesPerPixel = 1
                dataset.PhotometricInterpretation = "MONOCHROME2"
                dataset.BitsAllocated = 16
                dataset.BitsStored = 16
                dataset.HighBit = 15
                dataset.PixelRepresentation = 0
                dataset.PixelData = pixels.tobytes()
                dataset.save_as(path, enforce_file_format=True)

                class_rows.append({"patientId": patient_id, "class": class_name})
                label_rows.append({
                    "patientId": patient_id,
                    "Target": target,
                    "x": 2 if target else "",
                    "y": 3 if target else "",
                    "width": 8 if target else "",
                    "height": 9 if target else "",
                })

            labels_csv = root / "train_labels.csv"
            class_info_csv = root / "class_info.csv"
            with labels_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=("patientId", "Target", "x", "y", "width", "height")
                )
                writer.writeheader()
                writer.writerows(label_rows)
            with class_info_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("patientId", "class"))
                writer.writeheader()
                writer.writerows(class_rows)

            records, excluded = load_rsna_records(labels_csv, class_info_csv, image_dir)
            train, validation, test = split_records(records, seed=42)
            self.assertEqual((len(records), excluded), (20, 1))
            self.assertEqual((len(train), len(validation), len(test)), (14, 4, 2))
            self.assertEqual(sum(record.label == 1 for record in records), 10)
            pneumonia_record = next(record for record in records if record.label == 1)
            self.assertEqual((pneumonia_record.boxes[0].x, pneumonia_record.boxes[0].y), (2.0, 3.0))

            stats = compute_normalization_stats(record.image_path for record in train)
            stats_path = root / "normalization.npz"
            save_normalization_stats(stats, stats_path)
            loaded_stats = load_normalization_stats(stats_path)
            np.testing.assert_allclose(stats.mean, loaded_stats.mean)
            image = load_image(pneumonia_record, loaded_stats)
            self.assertEqual(image.shape, (128, 128))
            self.assertEqual(image.dtype, np.float32)
            self.assertTrue(np.isfinite(image).all())


if __name__ == "__main__":
    unittest.main()
