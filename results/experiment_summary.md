# Experiment summary

All metrics below were produced locally from the available RSNA split.
The test split was not used to choose hyperparameters or thresholds.

## Classification on the test split

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7983 | 0.7393 | 0.7737 | 0.7561 | 0.8623 |
| Linear SVM | 0.7808 | 0.7172 | 0.7554 | 0.7358 | 0.8476 |
| Random Forest | 0.8783 | 0.8710 | 0.8203 | 0.8449 | 0.9377 |
| Reference CNN | 0.8642 | 0.8446 | 0.8136 | 0.8288 | 0.9298 |
| DenseNet121 | 0.9038 | 0.8779 | 0.8852 | 0.8815 | 0.9589 |

## CAM localization on the test split

| Model | Threshold | All-image mean IoU | Pneumonia-only mean IoU |
|---|---:|---:|---:|
| Reference CNN | 0.20 | 0.0288 | 0.0712 |
| Reference CNN | 0.45 | 0.0357 | 0.0883 |
| Reference CNN | 0.70 | 0.0553 | 0.1368 |
| DenseNet121 | 0.20 | 0.0319 | 0.0789 |
| DenseNet121 | 0.45 | 0.0355 | 0.0878 |
| DenseNet121 | 0.70 | 0.0420 | 0.1040 |

## Important limitations

- The available copy contains 14,863 included records, not the paper's reported 17,489.
- The reference CNN uses original images only; no lung masks or U-Net weights were available.
- DenseNet121 uses ImageNet-pretrained frozen features and a trained binary head.
- CAM thresholds and the IoU aggregation are documented implementation choices, not claimed exact reproductions.
- The paper's supervised R-CNN is not implemented because its full architecture and training recipe are underspecified.
- Logistic Regression and linear SVM reported convergence warnings; interpret their scores as preliminary.

Plots: `classification_comparison.png` and `cam_localization_comparison.png`.
