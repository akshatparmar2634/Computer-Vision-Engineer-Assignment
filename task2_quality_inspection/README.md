# Task 2: Automated Quality Inspection System for PCB Manufacturing

## Overview
This solution implements an automated visual inspection system for detecting defects on Printed Circuit Boards (PCBs) using deep learning with YOLOv8.

## Dataset
- **Source**: PCB Defect Dataset
- **Format**: YOLO format with normalized bounding boxes
- **Classes**: 6 defect types
  - `0`: mouse_bite
  - `1`: spur
  - `2`: missing_hole
  - `3`: short (Critical)
  - `4`: open_circuit (Critical)
  - `5`: spurious_copper
- **Image Size**: 600×600 pixels
- **Splits**: Train, Validation, Test

## Project Structure
```
task2_quality_inspection/
├── data_processing.ipynb      # Exploratory Data Analysis
├── task2.ipynb                 # Training & Inference
├── inspect_pcb.py             # Standalone inspection script
├── pcb-defect-dataset/        # Dataset
│   ├── data.yaml
│   ├── train/
│   ├── val/
│   └── test/
└── README.md
```

## Features

### 1. **Defect Detection & Localization**
- Detects multiple defect types in a single image
- Provides bounding boxes around each defect
- Outputs (x, y) pixel coordinates of defect centers

### 2. **Classification with Confidence Scores**
- Each detection includes:
  - Defect class name
  - Confidence score (0-1)
  - Bounding box coordinates
  - Defect center coordinates

### 3. **Severity Assessment**
Defects are automatically classified into severity levels:
- **Critical**: Open circuits and shorts (always high priority)
- **High**: Large defects with high confidence (score ≥ 4)
- **Medium**: Medium-sized defects or moderate confidence (score ≥ 2)
- **Low**: Small defects with lower confidence

Severity factors:
- Defect type (critical vs non-critical)
- Normalized area (size relative to image)
- Detection confidence

### 4. **Comprehensive Output**
- JSON reports with detailed information
- Annotated images with color-coded severity levels
- Batch processing summaries
- Quality status (PASS/FAIL)

## Installation

### Requirements
```bash
pip install ultralytics opencv-python pillow pyyaml matplotlib seaborn pandas numpy
```

### Setup
1. Clone/download the project
2. Ensure the `pcb-defect-dataset` folder is in the correct location
3. Install dependencies

## Usage

### Option 1: Jupyter Notebooks

#### A. Exploratory Data Analysis
```bash
jupyter notebook data_processing.ipynb
```
This notebook provides:
- Dataset statistics
- Class distribution analysis
- Defect size analysis
- Visual samples with annotations

#### B. Training & Inference
```bash
jupyter notebook task2.ipynb
```
This notebook includes:
- Model training with YOLOv8
- Model evaluation
- Defect analysis system
- Batch processing
- Results visualization

### Option 2: Standalone Script

#### Single Image Inspection
```bash
python inspect_pcb.py --image path/to/image.jpg --output results/
```

#### Batch Processing
```bash
python inspect_pcb.py --batch path/to/images/ --output results/ --conf 0.3
```

#### Arguments
- `--image`: Path to a single image
- `--batch`: Path to directory containing multiple images
- `--output`: Output directory for results (default: `inspection_results`)
- `--model`: Path to trained model weights (default: `runs/detect/pcb_defect_detector/weights/best.pt`)
- `--conf`: Confidence threshold (default: 0.3)

## Output Format

### JSON Report Structure
```json
{
  "image_path": "path/to/image.jpg",
  "image_dimensions": {"width": 600, "height": 600},
  "timestamp": "2026-01-09T...",
  "total_defects": 2,
  "quality_status": "FAIL",
  "severity_breakdown": {
    "Critical": 1,
    "High": 0,
    "Medium": 1,
    "Low": 0
  },
  "defects": [
    {
      "class_id": 3,
      "class_name": "short",
      "confidence": 0.892,
      "bbox": {"x1": 120, "y1": 150, "x2": 180, "y2": 190},
      "center": {"x": 150, "y": 170},
      "dimensions": {
        "width_px": 60,
        "height_px": 40,
        "area_normalized": 0.0067
      },
      "severity": "Critical"
    }
  ]
}
```

### Visual Output
Annotated images include:
- Color-coded bounding boxes:
  - 🔴 Red: Critical
  - 🟠 Orange: High
  - 🟡 Yellow: Medium
  - 🟢 Green: Low
- Center point markers
- Labels with defect type, confidence, and severity
- Overall quality status banner

## Key Components

### DefectAnalyzer Class
The core inspection system with methods:
- `analyze_image()`: Analyzes a single image
- `assess_severity()`: Determines defect severity
- `visualize_results()`: Creates annotated images
- `generate_report()`: Produces detailed text reports

### Severity Assessment Logic
```python
severity_score = 0

# Area scoring
if area > 0.1:    severity_score += 3  # Large
elif area > 0.05: severity_score += 2  # Medium
elif area > 0.01: severity_score += 1  # Small

# Confidence scoring
if confidence > 0.9:  severity_score += 2  # Very confident
elif confidence > 0.7: severity_score += 1  # Confident

# Critical defects override
if defect_type in ['open_circuit', 'short']:
    return 'Critical'

# Map score to severity
if severity_score >= 4: return 'High'
elif severity_score >= 2: return 'Medium'
else: return 'Low'
```

## Model Training

### Architecture
- Base Model: YOLOv8n (nano)
- Input Size: 640×640
- Training Epochs: 50 (with early stopping)
- Batch Size: 16

### Training Configuration
```python
model.train(
    data='pcb-defect-dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    device='cpu'  # or 'cuda' for GPU
)
```

## Results & Performance

The system provides:
- **Detection Metrics**: mAP50, mAP50-95, Precision, Recall
- **Per-Class Performance**: Individual metrics for each defect type
- **Inference Speed**: Real-time processing capability
- **Accuracy**: High confidence detections with threshold control

## Sample Output

### Console Output
```
==================================================================
PCB DEFECT INSPECTION REPORT
==================================================================

Image: pcb-defect-dataset/test/images/sample_01.jpg
Timestamp: 2026-01-09T10:30:45.123456
Quality Status: FAIL

Total Defects Detected: 2

Severity Breakdown:
  Critical: 1
  Medium: 1

Detailed Defect Information:
------------------------------------------------------------------

Defect #1:
  Type: short
  Confidence: 0.892
  Center (x, y): (150, 170)
  Size: 60×40 px
  Normalized Area: 0.0067
  Severity: Critical

Defect #2:
  Type: missing_hole
  Confidence: 0.765
  Center (x, y): (320, 280)
  Size: 45×38 px
  Normalized Area: 0.0048
  Severity: Medium

==================================================================
```

## Advantages

1. **Automated**: No manual inspection required
2. **Fast**: Real-time processing capability
3. **Accurate**: Deep learning-based detection
4. **Comprehensive**: Multi-class defect detection
5. **Severity-Aware**: Prioritizes critical defects
6. **Scalable**: Batch processing for production lines
7. **Documented**: Detailed JSON reports for traceability

## Future Enhancements

- [ ] Integration with production line systems
- [ ] Real-time video stream processing
- [ ] Database integration for historical tracking
- [ ] Advanced analytics dashboard
- [ ] Model optimization for edge devices
- [ ] Additional defect types
- [ ] Multi-scale detection improvements

## License

This project is part of a Computer Vision Engineer Assignment.

## Author

Akshat Parmar
