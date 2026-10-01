# Data directory

The RSNA Pneumonia Detection dataset is not included in this repository. Obtain
the Stage 2 DICOM images and metadata through the dataset's authorized source.
Do not commit the dataset here.

For Phase 1, the loader expects:

- `train_labels.csv` with `patientId`, `Target`, `x`, `y`, `width`, and `height`
- `stage_2_detailed_class_info.csv` with `patientId` and `class`
- a directory of `.dcm` files named by patient ID

The detailed class file is necessary to separate `Normal` from
`No Lung Opacity / Not Normal`; both can have `Target=0` in the box labels.
The loader includes `Lung Opacity` and `Normal`, preserves all pneumonia boxes,
and excludes `No Lung Opacity / Not Normal` rows. It writes split manifests to
`data/splits/` when run; DICOM pixel files and generated manifests are ignored
by Git.
