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

The `notebooks/01_data_preparation.ipynb` notebook shows how to load records,
inspect a split, preprocess one DICOM, and visualize its label and box.

## Later project phases

Baselines, the reference CNN, CAM localization, DenseNet121, result tables, and
their plots have not been implemented. They remain for later phases. Results
will be measured from the supplied data; reported paper values will not be
substituted for local results.
