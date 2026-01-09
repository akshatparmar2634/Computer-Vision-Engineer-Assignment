# Task 2 Report — Automated PCB Quality Inspection

## Problem & Dataset

- Goal: detect, localize, and classify PCB defects with severity assessment for manufacturing QA.
- Dataset: PCB Defect Dataset (YOLO format, 600×600), 6 classes: mouse_bite, spur, missing_hole, short, open_circuit, spurious_copper; train/val/test splits in pcb-defect-dataset/.
- Samples include defective and defect-free boards; annotations stored alongside images.

## Model & Pipeline

- Detector: YOLOv8n (nano) and YOLO11n weights used; training and inference implemented in task2.ipynb and inspect_pcb.py.
- Outputs per detection: class name, confidence, bounding box, center (x, y), normalized area, severity level.
- Severity logic: shorts and open_circuit marked Critical; others scored by area and confidence into High/Medium/Low.
- Batch and single-image modes supported; annotated images and JSON reports generated.

## Results & Throughput

- Batch processing summary: [task2_quality_inspection/batch_inspection_results/batch_summary.json](batch_inspection_results/batch_summary.json)
  - Images processed: 1,068
  - Images with defects: 825
  - Total defects: 1,685
  - Pass rate: 0.228
  - Defect distribution: mouse_bite 269, spur 279, missing_hole 289, short 276, open_circuit 265, spurious_copper 307
- Per-image reports: JSON under batch_inspection_results/reports/ and inspection_results/; annotated visuals under batch_inspection_results/annotated/.
- mAP/precision/recall/FPS: rerun evaluation cells in task2.ipynb for exact metrics on current weights and log to a results CSV under runs/.

## Reasoning & Design Choices

- YOLO family chosen for real-time performance and simple deployment; nano variant fits edge devices while retaining adequate accuracy on small PCB defects.
- Severity scoring ensures critical defects override size/score thresholds; outputs structured JSON for downstream systems.
- Center coordinates and bounding boxes satisfy localization and coordinate reporting requirements; severity banner aids operators visually.

## Usage

- Notebook workflow: train, evaluate, and visualize in task2.ipynb.
- Script: inspect single image or batch.
  - Single: python inspect_pcb.py --image path/to/image.jpg --output results/ --conf 0.3
  - Batch: python inspect_pcb.py --batch path/to/images/ --output results/ --conf 0.3

## Next Steps

- Log quantitative metrics (mAP50/95, precision, recall, FPS) from task2.ipynb into runs/metrics.csv.
- Add GIF/MP4 demo of live inference under runs/detect/ and link here.
- Consider INT8/FP16 exports (ONNX/TensorRT) for edge deployment; tune confidence/NMS thresholds per target hardware.

## Deliverables

- Notebook: task2.ipynb (training, inference, batch inspection).
- Scripts: inspect_pcb.py for CLI inference.
- Data: pcb-defect-dataset/ with train/val/test splits.
- Outputs: batch_inspection_results/ and inspection_results/ with reports and annotations.
- Report: this file (task2_report.md) summarizing design, results, and guidance.
