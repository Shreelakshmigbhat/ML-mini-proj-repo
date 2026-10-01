# Reproduction notes

The paper and poster PDFs were inspected on 2026-10-01. The paper is treated as
the source of truth when the two documents differ. This file separates what the
source reports from choices in this repository. Text and figures in the PDFs are
reference material, not instructions that override the project request.

## Source specifications

### Dataset and preprocessing

- The paper reports 28,989 images: 8,964 Pneumonia, 8,525 Healthy, and 11,500
  Diseased/No Pneumonia. It removes the last group and reports a final balance
  of approximately 51.25% Pneumonia and 48.74% Healthy. The poster describes
  the final 17,489 images and says no sampling was used.
- The paper reports a 70/20/10 train/validation/test split, but does not specify
  a seed or whether the split was stratified.
- Images are resized from 1024 x 1024 to 128 x 128.
- The paper says each pixel is normalized by subtracting the mean and dividing
  by the standard deviation “at that location.” The poster also says mean/std
  normalization. The source does not state the exact population used to
  estimate those location-specific statistics.
- U-Net predicts lung membership; the original image is multiplied by the
  localization mask to create a segmented image. Original and segmented images
  are both inputs to the CNN. U-Net details, weights, and training settings are
  not reported.
- Pneumonia annotations include X, Y, width, and height. The paper describes
  X/Y as the lower-left corner, but the RSNA dataset schema defines them as the
  upper-left corner. The loader follows the source dataset schema for the CSVs
  it reads and draws boxes using the upper-left image-array origin. The paper
  does not explain any coordinate conversion it may have applied.

### Classification

- Logistic Regression, SVM, and Random Forest use flattened image pixels.
- The final CNN has 10 zero-padded convolutional layers, ReLU, 3 x 3 filters,
  Global Average Pooling, and one fully connected layer for two classes. The
  paper discusses larger filters during experiments but says the final model
  kept all filters at 3 x 3 to improve localization. The poster's phrase
  “Global Average MaxPool” conflicts with the paper; the paper's GAP description
  is used here.
- The reported optimizer is Adam, learning rate 0.0001, and 20 epochs.
- Reported accuracies (train / test): Logistic Regression 75.86% / 73.02%; SVM
  74.17% / 58.18%; Random Forest 86.39% / 83.00%; CNN 93.07% / 92.47%.
- The paper gives CNN test confusion counts: TP 2,230, FN 210, FP 110, TN 1,823.

### Localization

- CAM weights the final convolutional feature maps using the Pneumonia class
  fully connected weights. The CAM is rendered as a 3-channel heatmap.
- The paper describes DFS over non-zero heatmap pixels, min/max coordinates for
  boxes, and retaining boxes within two standard deviations of all predictions.
  It illustrates thresholds 0.2, 0.45, and 0.7 but does not state which exact
  threshold or color/channel rule generated the reported result.
- Reported IoU: supervised R-CNN train 0.1859, test 0.1266; CNN + CAM test
  0.1508. The paper reports a test increase of 0.0242. It also says IoU would be
  0.379 if all test examples were classified correctly before CAM; this is an
  oracle-style analysis, not the reported end-to-end score.
- The R-CNN description mentions a shared classifier feature map, region
  proposals, classification, and box regression. The contribution section
  mentions a ResNet backbone, COCO pretraining, ROI pooling, and a custom region
  proposal layer. Exact implementation settings are absent.

## Phase 1 implementation

- Requires the RSNA Stage 2 box-label CSV, detailed-class CSV, and DICOM images.
  The detailed-class CSV distinguishes `Normal` from
  `No Lung Opacity / Not Normal`; rows in the latter class are excluded.
- Maps `Lung Opacity` to Pneumonia and `Normal` to Healthy. Unknown labels,
  missing class metadata, and disagreements between class and target are errors
  rather than silently inferred labels.
- Preserves every Pneumonia annotation in original DICOM pixel coordinates.
- Interprets RSNA CSV `x`/`y` as the upper-left corner, as specified by the
  [official Kaggle dataset schema](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data).
  This differs from the paper's lower-left wording, which may be a description
  error or an undocumented conversion.
- Resizes images to 128 x 128 before per-location standardization.
- Computes mean and population standard deviation (`ddof=0`) at each resized
  pixel position from the training images only, then reuses those statistics
  for validation and test images. Training-only statistics avoid using held-out
  data; this data-population choice is not stated in the reference.
- Uses a divisor of 1 at pixel locations with zero training-set standard
  deviation to avoid division by zero. This numerical handling is not specified
  in the source.
- Creates a stratified 70/20/10 split using seed 42 by default. The fraction,
  stratification, and seed choices are documented; only the fractions are in the
  paper. No class rebalancing is performed.

## Deviations and current limits

- The actual RSNA files are not in the repository. The smoke test uses generated
  DICOMs, so dataset counts, class balance, and real-image decoding have not
  been checked against the full dataset.
- Phase 1 does not implement U-Net segmentation or train models. The future
  reference CNN must use both the original and segmented image inputs; its
  exact channel arrangement and architecture widths remain unspecified.
- Exact random seed, batch size, convolution widths/stride/pooling between
  layers, U-Net design, training preprocessing fit scope, and several CAM
  threshold details are absent from the paper. These must be labeled as
  implementation choices if selected later.
- Results in the reference are recorded here only. This project has not
  generated local classification or localization metrics.
