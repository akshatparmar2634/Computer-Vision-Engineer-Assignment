# Task 1 Report — Custom Object Detection from Scratch

## Problem & Dataset

- Goal: train object detectors from scratch (no pretrained weights) and compare accuracy, speed, and size.
- Dataset: PASCAL VOC subset (train/val sampled), 3–5+ classes plus background; images resized to 320×320 for speed.
- Augmentation: horizontal flip, color jitter, Gaussian blur, resize; normalized boxes; custom collate for variable objects.

## Models Implemented

- Simple detector: single-scale CNN head; 1.29M params (4.9 MB).
- Medium detector: lightweight backbone + FPN with 3 scales; 3.35M params (12.8 MB).
- Complex detector: Faster R-CNN–style backbone + RPN + ROI head; 40.3M params (153.9 MB).

## Training Setup

- Optimizer: Adam; grad clipping 10.0.
- LR scheduling: ReduceLROnPlateau (Simple), OneCycleLR (Medium), fixed for Complex prototype.
- Early stopping: patience 5 epochs; checkpoints saved under checkpoints/.
- Loss: classification + bbox + objectness with scale-aware targets; Medium averages per-FPN losses; Complex uses placeholder RPN/ROI targets (needs upgrade).

## Results (validation, from notebook)

| Model | mAP@0.5 | Precision | Recall | FPS | Latency (ms) | Size (MB) |
| --- | --- | --- | --- | --- | --- | --- |
| Simple | 0.0033 | 0.040 | 0.069 | 195.2 | 5.12 | 4.9 |
| Medium | 0.0050 | 0.062 | 0.049 | 411.4 | 2.43 | 12.8 |
| Complex | 0.0000 | 0.000 | 0.000 | 63.9 | 15.6 | 153.9 |

- Source table: [task1_object_detection_from_scratch/checkpoints/evaluation_results.csv](checkpoints/evaluation_results.csv)
- Interpretation: speed targets met, but accuracy is very low; Complex failed to learn due to simplistic ROI/target assignment.

## Architecture Rationale

- Simple: minimal depth to benchmark speed and footprint; single scale limits small-object recall.
- Medium: FPN adds multi-scale context with modest cost; best speed and slightly higher accuracy.
- Complex: intended for higher accuracy via proposals and refinement; current implementation lacks proper proposal sampling and target matching.

## Next Steps to Improve Accuracy

- Implement proper RPN/ROI target assignment for Complex; add NMS and balanced sampling.
- Tune anchors and strides; increase input size (e.g., 416–512) and train longer.
- Improve loss: CIOU/DIoU for boxes; focal loss for class/objectness to handle imbalance.
- Expand augmentation (mosaic, mixup) and rebalance rare classes.
- Export and test ONNX/TensorRT once mAP improves; record inference GIF/MP4 under runs/detect/ and link here.

## Deliverables

- Notebook: task1.ipynb with full pipeline and evaluation.
- Metrics CSV: [task1_object_detection_from_scratch/checkpoints/evaluation_results.csv](checkpoints/evaluation_results.csv).
- Checkpoints: best weights per model under checkpoints/ (LFS-managed for large files).
- Report: this file (task1_report.md) summarizing design, training, results, and improvements.
