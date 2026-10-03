# RSNA Pneumonia Classification and Localization — Project Report

## 1. Executive summary

This project implements and evaluates a data-preparation pipeline, three
classical classification baselines, a paper-inspired convolutional neural
network (CNN), a DenseNet121 comparison, and class activation map (CAM)
localization against RSNA lesion annotations.

The models and CAM evaluation were run on the locally available RSNA dataset
copy. The full test suite passed with **15 tests**. The best test classification
result was DenseNet121, with **90.38% accuracy** and **0.9589 ROC-AUC**.

This is a documented, partial reproduction—not an exact reproduction of every
result in the paper. The available cohort contains 14,863 included records,
compared with 17,489 reported by the paper/poster. Also, the paper's CNN input
includes lung-segmented images, but the available RSNA annotations are lesion
boxes, not lung masks, and no U-Net weights were provided. The implemented CNN
therefore uses original images only. The paper's supervised R-CNN is not
implemented because the source lacks a reproducible architecture and training
recipe.

## 2. Dataset and Phase 1 preparation

### Source files

The local preparation run used the RSNA Stage 2 training DICOMs and:

- `stage_2_train_labels.csv`
- `stage_2_detailed_class_info.csv`
- DICOM files named by patient ID

The sample-submission CSV is not used as training data because it has no
ground-truth class labels or lesion boxes. Dataset files and per-image outputs
are local and excluded from Git.

### Class inclusion and labels

- `Lung Opacity` is mapped to Pneumonia (`1`).
- `Normal` is mapped to Healthy (`0`).
- `No Lung Opacity / Not Normal` is excluded.
- Unknown classes, missing metadata, conflicting targets, malformed labels,
  and invalid lesion boxes are treated as errors.
- All boxes for each Pneumonia patient are preserved in source DICOM pixel
  coordinates.

The loader follows the RSNA CSV convention that `x` and `y` refer to the
upper-left box corner. This differs from the paper's lower-left wording.

### Image preparation and split

- Images are resized to 128 × 128.
- Per-pixel-location means and population standard deviations (`ddof=0`) are
  computed from training images only and reused for validation and test.
- A zero standard deviation is replaced by a divisor of 1 to avoid division by
  zero.
- A stratified 70/20/10 split is created with seed 42. Stratification and the
  seed are implementation choices; the paper only specifies the split ratio.

### Observed dataset counts

| Partition / count | Images |
|---|---:|
| Included Pneumonia and Healthy | 14,863 |
| Excluded `No Lung Opacity / Not Normal` | 11,821 |
| Train | 10,403 |
| Validation | 2,973 |
| Test | 1,487 |

The included count is 2,626 below the paper's reported 17,489. The reason has
not been established; the local copy may differ in completeness or source
version. Results should not be treated as a strict paper comparison until this
is reconciled.

## 3. Classification methods

### Classical baselines

Logistic Regression, linear SVM, and Random Forest use flattened,
per-pixel-standardized 128 × 128 images. Each model is fit on the training
partition. The validation and test partitions are reported separately and are
not used for tuning.

Implementation parameters:

- Logistic Regression: `lbfgs`, maximum 1,000 iterations.
- Linear SVM: `C=1.0`, maximum 5,000 iterations.
- Random Forest: 100 trees, square-root feature selection, fixed seed 42.

The paper does not specify all estimator parameters. Logistic Regression and
linear SVM emitted convergence warnings at their configured iteration limits;
their results are preliminary.

### Reference CNN

The CNN has ten zero-padded 3 × 3 convolution layers, ReLU activations,
spatial max-pooling after selected layers, global average pooling, and a
two-class linear head. It was trained for 20 epochs using Adam with learning
rate 0.0001, batch size 64, and seed 42.

The convolution widths and pooling positions are implementation choices because
the paper does not fully specify them. **The CNN uses original images only.**
The paper also calls for lung-segmented input, which could not be produced from
the available lesion-box annotations.

### DenseNet121

Torchvision DenseNet121 ImageNet weights were adapted for single-channel input
by averaging the RGB input weights into one channel. The pretrained feature
extractor was frozen and a two-class linear head was trained for 20 epochs using
Adam at learning rate 0.0001, batch size 64, and seed 42. This transfer-learning
configuration is an implementation choice.

## 4. Classification results

All results below are from the held-out test partition. Pneumonia is the
positive class.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7983 | 0.7393 | 0.7737 | 0.7561 | 0.8623 | 722 | 164 | 136 | 465 |
| Linear SVM | 0.7808 | 0.7172 | 0.7554 | 0.7358 | 0.8476 | 707 | 179 | 147 | 454 |
| Random Forest | 0.8783 | 0.8710 | 0.8203 | 0.8449 | 0.9377 | 813 | 73 | 108 | 493 |
| Reference CNN | 0.8642 | 0.8446 | 0.8136 | 0.8288 | 0.9298 | 796 | 90 | 112 | 489 |
| DenseNet121 | **0.9038** | 0.8779 | **0.8852** | **0.8815** | **0.9589** | 812 | 74 | 69 | 532 |

The paper reports CNN test accuracy 92.47% with TP 2,230, FN 210, FP 110, and
TN 1,823. Those counts correspond to a larger test set than the 1,487 images in
this run and are not directly comparable.

## 5. CAM localization

For each image, the Pneumonia-class CAM is computed from the final convolutional
feature maps and classifier weights, resized to the input resolution, and
normalized to `[0, 1]`. Thresholded, 8-connected regions are converted into
bounding boxes. The model's Pneumonia probability must be at least 0.5 for
predicted boxes to be emitted.

The paper illustrates thresholds 0.2, 0.45, and 0.7 without fully specifying
which threshold/channel convention produced its reported score. The project
evaluates all three. IoU for an image is the maximum overlap across predicted
CAM components and annotated boxes. Images with no predicted boxes score zero.
The all-image mean includes healthy images and missed Pneumonia classifications;
the Pneumonia-only mean includes all positive test patients, including
classification misses.

| Model | CAM threshold | All-test-image mean best IoU | Pneumonia-only mean best IoU |
|---|---:|---:|---:|
| Reference CNN | 0.20 | 0.0288 | 0.0712 |
| Reference CNN | 0.45 | 0.0357 | 0.0883 |
| Reference CNN | 0.70 | 0.0553 | **0.1368** |
| DenseNet121 | 0.20 | 0.0319 | 0.0789 |
| DenseNet121 | 0.45 | 0.0355 | 0.0878 |
| DenseNet121 | 0.70 | 0.0420 | 0.1040 |

The paper reports CNN + CAM test IoU 0.1508, but its exact threshold and
aggregation details are unclear. The values above use a separately documented
protocol and should not be interpreted as a like-for-like reproduction.
Localization is lesion-box localization, **not lung segmentation**.

## 6. Artifacts

Tracked experiment results and plots:

- [Experiment summary](../results/experiment_summary.md)
- [Classification comparison plot](../results/classification_comparison.png)
- [CAM localization comparison plot](../results/cam_localization_comparison.png)
- [Baseline metrics JSON](../results/baseline_metrics.json)
- [Reference CNN metrics JSON](../results/reference_cnn_metrics.json)
- [DenseNet121 metrics JSON](../results/densenet121_metrics.json)
- [Reference CNN localization JSON](../results/reference_cnn_localization.json)
- [DenseNet121 localization JSON](../results/densenet121_localization.json)
- [Reproduction notes](../results/reproduction_notes.md)

Five CAM overlays per model were generated under the local ignored
`results/cam_examples/` directory. The overlays contain radiographs and are
deliberately not committed. Model checkpoint files are also local and ignored.

## 7. Reproduction commands

Run from the repository root after obtaining the authorized dataset files.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Prepare splits and training-set normalization statistics:

```powershell
python -m src.dataset `
  --labels-csv data/stage_2_train_labels.csv `
  --class-info-csv data/stage_2_detailed_class_info.csv `
  --image-dir data/stage_2_train_images `
  --output-dir data/splits `
  --seed 42 `
  --compute-normalization-stats
```

Train the classical baselines:

```powershell
python -m src.baselines `
  --splits-dir data/splits `
  --output results/baseline_metrics.json `
  --seed 42
```

Train the reference CNN:

```powershell
python -m src.reference_cnn `
  --splits-dir data/splits `
  --output results/reference_cnn_metrics.json `
  --checkpoint results/reference_cnn.pt `
  --seed 42
```

Train DenseNet121 (downloads ImageNet weights on first run):

```powershell
python -m src.densenet121 `
  --splits-dir data/splits `
  --output results/densenet121_metrics.json `
  --checkpoint results/densenet121.pt `
  --seed 42
```

Evaluate CAM localization for either CNN:

```powershell
python -m src.localization `
  --model reference_cnn `
  --checkpoint results/reference_cnn.pt `
  --splits-dir data/splits `
  --output results/reference_cnn_localization.json `
  --thresholds 0.2 0.45 0.7 `
  --visualization-dir results/cam_examples/reference_cnn
```

Replace `reference_cnn` with `densenet121` and its checkpoint/output paths to
evaluate DenseNet121 CAMs.

Regenerate the comparison summary and plots:

```powershell
python -m src.report
```

Run tests:

```powershell
python -m pytest tests -q
```

## 8. Verification and repository state

- Final full test run: **15 passed**. Dependency deprecation warnings were
  reported by Matplotlib/Pyparsing and scikit-learn; no test failures occurred.
- Phase 1, both classification model families, both localization evaluations,
  and the reporting workflow were run against the available data.
- Latest feature commit:
  [`feat: complete DenseNet and CAM localization workflows`](https://github.com/Shreelakshmigbhat/ML-mini-proj-repo/commit/53e74d2e605650866003666fbce2ff915b60948a).

## 9. Scope still not exactly reproduced

The implemented workflows are complete for the available data and agreed
project scope. The following paper components remain unverified or
unimplemented:

1. **Dataset count discrepancy:** reconcile the available 14,863 included
   records with the paper's 17,489.
2. **Lung segmentation:** the repository has no lung masks or U-Net model.
   Obtain appropriate lung-mask labels or a specified/trained U-Net, then rerun
   the CNN with original and segmented inputs if exact source behavior is
   required.
3. **Supervised R-CNN:** the paper's architectural and training details are
   insufficient for a faithful implementation; additional source details are
   needed.
4. **Exact localization comparison:** the source does not fully specify its CAM
   threshold/channel handling and IoU aggregation, so the project reports its
   own explicit evaluation protocol rather than claiming its metric is directly
   equivalent.
