# Pneumonia localization mini-project

This repository is being built in phases to reproduce the paper *Weakly
Supervised Pneumonia Localization* and compare its reference CNN with a
DenseNet121 extension. Phase 1 implements data loading and preprocessing only.
No model metrics are reported yet.

## Reference paper

Shih-Cheng (Mars) Huang, Medi Monam, and Emanuel Cortes, *Weakly Supervised
Pneumonia Localization*. The provided paper and poster have now been inspected.
Paper specifications, poster differences, and implementation choices are
recorded separately in `results/reproduction_notes.md`.

## Dataset

Use the RSNA Pneumonia Detection Stage 2 DICOM data and metadata. Dataset files
are not included. See [data/README.md](data/README.md) for the expected files.
The loader requires both the box-label CSV and detailed-class CSV to distinguish
Healthy cases from Diseased/No Pneumonia cases.

## Phase 1 preprocessing

The loader keeps `Lung Opacity` as Pneumonia and `Normal` as Healthy, excludes
`No Lung Opacity / Not Normal`, and keeps pneumonia bounding boxes in source
pixel coordinates. It resizes images to 128 x 128, then computes per-pixel
location means and standard deviations over the training split and uses these
to standardize all partitions. This follows the paper's description; population
standard deviation and zero-variance handling are implementation choices. It
makes a stratified 70/20/10 split using seed 42 by default, without rebalancing
the classes. Stratification and the default seed are implementation choices.
See the reproduction notes for source details and ambiguities.

## Install and run Phase 1

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m src.dataset `
  --labels-csv data/raw/stage_2_train_labels.csv `
  --class-info-csv data/raw/stage_2_detailed_class_info.csv `
  --image-dir data/raw/stage_2_train_images `
  --output-dir data/splits `
  --seed 42 `
  --compute-normalization-stats
```

The command checks metadata and matching DICOMs, prints counts and configuration,
then writes `train.csv`, `validation.csv`, and `test.csv` manifests containing
labels, paths, and ground-truth boxes. With `--compute-normalization-stats`, it
also reads training images and writes `normalization.npz`. This can take time
for the full dataset. To change any path, pass a different command-line
argument; dataset paths are not hard-coded in the loader.

Normalization uses per-pixel training-set means and standard deviations. The
Stage 2 box-label CSV is required; the sample-submission CSV cannot replace it.
If your authorized dataset copy does not contain `stage_2_train_labels.csv`,
obtain that file from the same dataset source before running Phase 1. Dataset
CSVs and DICOM directories are excluded from Git.

The `notebooks/01_data_preparation.ipynb` notebook shows how to load records,
inspect a split, preprocess one DICOM, and visualize its label and box.

## Phase 2 modeling

Phase 2 implements flattened-pixel Logistic Regression, linear SVM, and Random
Forest baselines, plus a 10-convolution reference CNN. Run the baselines after
Phase 1 has generated its splits and normalization statistics:

```powershell
python -m src.baselines `
  --splits-dir data/splits `
  --output results/baseline_metrics.json `
  --seed 42
```

Models fit on the training partition only; validation and test metrics are
reported without tuning on those partitions. The JSON records accuracy,
precision, recall, F1, ROC-AUC, and confusion counts. Run one or more models
with `--models logistic_regression linear_svm random_forest`.

Train the paper-inspired reference CNN with the source's Adam optimizer,
learning rate, and 20 epochs:

```powershell
python -m src.reference_cnn `
  --splits-dir data/splits `
  --output results/reference_cnn_metrics.json `
  --checkpoint results/reference_cnn.pt `
  --seed 42
```

The CNN uses the configured Python environment's PyTorch dependency. It follows
the reported 10 padded 3 x 3 convolution / ReLU layers, global
average pooling, two-class linear head, Adam optimizer, learning rate 0.0001,
and 20 epochs. Widths and downsampling are implementation choices recorded in
the metrics JSON. The paper also requires segmented images; no lung masks or
trained U-Net were supplied, so this run uses original images only. This is an
explicit deviation, not an exact reproduction. CAM localization and DenseNet121
remain for later phases. Results are measured from the supplied data; reported
paper values are not substituted for local results.
