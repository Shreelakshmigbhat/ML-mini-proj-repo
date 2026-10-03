# Data directory

The RSNA Pneumonia Detection dataset is not included in this repository. Obtain
the Stage 2 DICOM images and metadata through the dataset's authorized source.
Do not commit the dataset here.

For Phase 1, the loader expects the following Stage 2 training files (place
them under `data/raw/`, or pass their actual paths to the CLI):

- `train_labels.csv` with `patientId`, `Target`, `x`, `y`, `width`, and `height`
- `stage_2_detailed_class_info.csv` with `patientId` and `class`
- a directory of `.dcm` files named by patient ID

`stage_2_sample_submission.csv` is for the competition test set and does not
contain training labels or ground-truth boxes. It cannot replace
`train_labels.csv`.

The detailed class file is necessary to separate `Normal` from
`No Lung Opacity / Not Normal`; both can have `Target=0` in the box labels.
The loader includes `Lung Opacity` and `Normal`, preserves all pneumonia boxes,
and excludes `No Lung Opacity / Not Normal` rows. It resizes images to 128 x
128, then can compute per-pixel means and standard deviations from the training
split and reuse them for the other splits. It writes split manifests to
`data/splits/` when run; DICOM pixel files and generated manifests are ignored
by Git.
