# Vietnamese License Plate Recognition (LPR) with YOLOv11 + PaddleOCR

## Metadata

```yaml
name: vietnamese-lpr-yolo-paddleocr
description: >
  End-to-end pipeline for detecting and recognizing Vietnamese license plates using 
  YOLOv11 for plate detection and PaddleOCR for character recognition. 
  Use when building ANPR (Automatic Number Plate Recognition) systems for Vietnam.
```

---

## Overview

This skill guides the development of a production-ready Vietnamese License Plate Recognition system using a two-stage pipeline:

1. **Detection Stage**: YOLOv11 identifies and localizes license plates in images/video frames
2. **Recognition Stage**: PaddleOCR extracts characters from detected plate regions

The pipeline follows computer vision best practices with proper data management, model optimization, and evaluation protocols.

---

## When to Use

- Building a Vietnamese license plate recognition system from scratch
- Adding plate detection/recognition to an existing CV application
- Optimizing inference speed for real-time applications
- Training custom models on Vietnamese plate datasets
- Evaluating and benchmarking LPR performance

**When NOT to use:** General object detection without plate recognition, non-Vietnamese plate formats.

---

## Vietnamese License Plate Formats

Understand the plate types before implementation:

| Type | Format | Example | Characters |
|------|--------|---------|-------------|
| **Private (80x20)** | XX-YYYY.NN | 30A-1234.56 | 7-8 chars |
| **Motorcycle (60x20)** | YY-NNNNN | 43-12345 | 6-7 chars |
| **Police (White)** | XX-YYYY-NN | 60-1234-56 | 7-8 chars |
| **Army (Red)** | YYYYYY-NN | 123456-78 | 8-9 chars |

**Key characteristics:**
- Font: VnExpress, Arial-like Vietnamese font
- Colors: White (private), Yellow (commercial), Red (military/gov), Green (electric)
- Reflective properties vary by era
- Some plates have provinces embedded (30A = Ho Chi Minh City)

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VIETNAMESE LPR PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  INPUT   │───▶│PREPROCESS│───▶│ DETECT   │───▶│ CROP     │              │
│  │  IMAGE   │    │          │    │ (YOLOv11)│    │          │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│                                                             │                │
│                                                             ▼                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  OUTPUT  │◀───│  FORMAT  │◀───│ RECOGNIZE│◀───│  ALIGN   │              │
│  │  RESULT  │    │          │    │(PaddleOCR│    │          │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Stage Details:

1. INPUT
   - Image: JPG, PNG, BMP (max 4096x4096)
   - Video: MP4, AVI, RTSP stream
   - Batch processing supported

2. PREPROCESS
   - Resize: maintain aspect ratio, max 1920x1080
   - Normalize: /255.0 or ImageNet stats
   - Augment (training): rotation ±5°, brightness ±10%, noise

3. DETECT (YOLOv11)
   - Model: yolov11n.pt → yolov11s.pt → yolov11m.pt (scale up for accuracy)
   - Confidence threshold: 0.25 (configurable)
   - NMS threshold: 0.45
   - Output: [x1, y1, x2, y2, conf, class_id]

4. CROP & ALIGN
   - Perspective transform for skewed plates
   - Padding: 10% border around detected region
   - Resize to standard plate size (320x64 for car, 200x80 for motorcycle)

5. RECOGNIZE (PaddleOCR)
   - Detection: DB (Differentiable Binarization)
   - Recognition: CRNN with CTC loss
   - Language: Vietnamese + English + Numbers
   - Output: [text, confidence, bbox]

6. FORMAT
   - Normalize Vietnamese characters
   - Format validation (plate format rules)
   - Post-processing: spell-check against known patterns
```

---

## Project Structure

```
vietnamese-lpr/
├── data/
│   ├── raw/                    # Original collected data
│   │   ├── images/
│   │   └── annotations/
│   ├── processed/              # Preprocessed data
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── datasets/               # Formatted datasets
│   │   ├── yolo_detection/
│   │   └── ocr_recognition/
│   └── external/               # Public datasets
│
├── src/
│   ├── detection/              # YOLOv11 detection module
│   │   ├── models/
│   │   ├── datasets/
│   │   ├── utils/
│   │   └── train.py
│   │
│   ├── recognition/            # PaddleOCR module
│   │   ├── models/
│   │   ├── preprocessing/
│   │   └── train.py
│   │
│   ├── pipeline/               # End-to-end pipeline
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   ├── recognizer.py
│   │   ├── postprocessor.py
│   │   └── inference.py
│   │
│   └── utils/
│       ├── visualization.py
│       ├── metrics.py
│       └── logger.py
│
├── configs/
│   ├── detection.yaml
│   ├── recognition.yaml
│   └── pipeline.yaml
│
├── models/                     # Trained models
│   ├── yolov11/               # Detection weights
│   └── paddleocr/            # Recognition weights
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_pipeline_evaluation.ipynb
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmark/
│
├── outputs/
│   ├── logs/
│   ├── visualizations/
│   └── results/
│
└── docs/
    ├── architecture.md
    ├── api_reference.md
    └── training_guide.md
```

---

## Development Workflow

### Phase 1: Data Preparation

**Step 1.1: Data Collection**
```bash
# Download Vietnamese plate datasets
python scripts/download_datasets.py --sources AI-Challenger,OpenLPR-VN

# Or collect your own data
python scripts/collect_data.py --source camera --output data/raw/
```

**Step 1.2: Annotation**
```bash
# Label with LabelImg for detection (YOLO format)
# Label with LabelStudio for OCR recognition
python scripts/convert_annotations.py --format yolo --output data/datasets/
```

**Step 1.3: Dataset Statistics**
```bash
# Analyze dataset composition
python scripts/analyze_dataset.py --data data/processed/
# Output: class distribution, image quality, plate type breakdown
```

**Verification:**
- [ ] Minimum 1000 images per plate type
- [ ] Balanced dataset across provinces
- [ ] Annotations verified for accuracy (sample 5%)
- [ ] Dataset logged with metadata

---

### Phase 2: Detection Model (YOLOv11)

**Step 2.1: Configuration**
```yaml
# configs/detection.yaml
model:
  name: yolov11s
  pretrained: true
  classes: 1  # plate only

data:
  train: data/datasets/yolo_detection/train/
  val: data/datasets/yolo_detection/val/
  nc: 1
  names: ['license_plate']

training:
  epochs: 100
  batch_size: 16
  imgsz: 640
  device: 0  # or 'cpu'
  optimizer: AdamW
  lr0: 0.001
  warmup_epochs: 3
```

**Step 2.2: Training**
```bash
# Train detection model
python src/detection/train.py \
    --config configs/detection.yaml \
    --epochs 100 \
    --batch 16 \
    --name yolov11s_plates
```

**Step 2.3: Evaluation**
```bash
# Evaluate on test set
python src/detection/eval.py \
    --weights runs/train/yolov11s_plates/weights/best.pt \
    --data data/datasets/yolo_detection/test/

# Expected metrics:
# mAP@0.5: > 0.95
# mAP@0.5:0.95: > 0.85
```

**Verification:**
- [ ] mAP@0.5 > 0.95 on validation set
- [ ] Inference speed < 10ms per image (GPU)
- [ ] Visual inspection of detection results

---

### Phase 3: Recognition Model (PaddleOCR)

**Step 3.1: Prepare OCR Data**
```bash
# Extract cropped plates and create OCR annotations
python scripts/prepare_ocr_data.py \
    --detection-model models/yolov11/best.pt \
    --output data/datasets/ocr_recognition/
```

**Step 3.2: Configuration**
```yaml
# configs/recognition.yaml
Global:
  algorithm: CRNN
  character_dict_path: ppocr/utils/vietnam_dict.txt
  max_text_length: 25
  infer_img_shape: "3, 48, 320"

Architecture:
  model_type: det,rec
  Backbone:
    name: ResNet34_vd
  Neck:
    name: RNNEncoder
  Head:
    name: CTCHead

Train:
  dataset:
    name: SimpleDataset
    data_dir: data/datasets/ocr_recognition/train/
  loader:
    batch_size: 64
    num_workers: 4
```

**Step 3.3: Training**
```bash
# Train OCR model
python src/recognition/train.py \
    --config configs/recognition.yaml \
    --epochs 200

# Fine-tune with PaddleOCR
python src/recognition/train_paddleocr.py \
    --pretrained_model ppocr/v4/en_PP-OCRv4_rec_train \
    --train_data data/datasets/ocr_recognition/train/
```

**Verification:**
- [ ] Character accuracy > 95%
- [ ] Word accuracy > 90%
- [ ] Inference speed < 50ms per plate (GPU)

---

### Phase 4: Pipeline Integration

**Step 4.1: Build Pipeline**
```python
# src/pipeline/inference.py
from pipeline import VietnameseLPRPipeline

pipeline = VietnameseLPRPipeline(
    detector_weights="models/yolov11/best.pt",
    recognizer_config="configs/recognition.yaml",
    recognizer_weights="models/paddleocr/best",
    device="cuda"
)

# Single image inference
result = pipeline.process_image("test.jpg")
print(result)
# {'plate': '30A-1234.56', 'confidence': 0.95, 'bbox': [x1, y1, x2, y2]}

# Video inference
results = pipeline.process_video("traffic.mp4")
# Yields: [{'plate': '...', 'timestamp': 1.5}, ...]
```

**Step 4.2: Post-processing**
```python
# Validate and normalize results
def normalize_vietnamese_plate(text: str) -> str:
    """Normalize plate text to standard format."""
    # Remove spaces and special chars
    # Convert to uppercase
    # Validate against known patterns
    # Return formatted string
```

**Verification:**
- [ ] End-to-end inference works correctly
- [ ] Batch processing supported
- [ ] Video stream processing works
- [ ] Error handling for failed detections

---

### Phase 5: Evaluation & Benchmarking

**Step 5.1: Metrics**
```bash
# Run comprehensive evaluation
python scripts/evaluate_pipeline.py \
    --pipeline models/pipeline \
    --test-data data/processed/test \
    --output outputs/results/evaluation.json
```

**Metrics to report:**
| Metric | Target | Description |
|--------|--------|-------------|
| Detection mAP@0.5 | > 0.95 | Plate detection accuracy |
| Recognition Accuracy | > 90% | Character-level accuracy |
| End-to-End Accuracy | > 85% | Full plate match rate |
| Inference Speed (GPU) | < 50ms | Per image latency |
| Inference Speed (CPU) | < 200ms | Per image latency |

**Step 5.2: Benchmark Tests**
```bash
# Performance benchmarking
python scripts/benchmark.py \
    --model models/pipeline \
    --test-images data/processed/test \
    --iterations 1000

# Output: latency percentiles, throughput, memory usage
```

---

## Common Rationalizations

| Rationalization | Reality |
|----------------|---------|
| "Generic OCR will work for Vietnamese" | Vietnamese characters (ạ, ă, đ, ê, ô, ơ, ư) require dedicated training |
| "YOLOv8 is fine, no need for v11" | YOLOv11 has better speed-accuracy tradeoffs; upgrade if available |
| "Training on 500 images is enough" | Vietnamese plates have high variation; need 5000+ for production |
| "Skip validation, training loss is low" | Validation mAP is required to detect overfitting |
| "Pre-trained models are sufficient" | Public models often trained on non-Vietnamese plates |

---

## Red Flags

- Using generic character recognition without Vietnamese fine-tuning
- No data augmentation for lighting/angle variations
- Ignoring plate type classification (car vs motorcycle)
- Single-model approach for different plate colors
- No handling for blurry/occluded plates
- Missing confidence thresholds in output

---

## Verification Checklist

Before considering the pipeline production-ready:

### Detection Module
- [ ] mAP@0.5 > 0.95 on held-out test set
- [ ] Works on images with multiple plates
- [ ] Handles various plate sizes (near/far)
- [ ] Robust to partial occlusions
- [ ] Speed meets latency requirements

### Recognition Module  
- [ ] Character accuracy > 95%
- [ ] Handles worn/damaged characters
- [ ] Works across different lighting conditions
- [ ] Vietnamese character support verified
- [ ] Confidence calibration appropriate

### Pipeline Integration
- [ ] End-to-end accuracy > 85%
- [ ] Video processing functional
- [ ] Batch inference optimized
- [ ] Error cases handled gracefully
- [ ] Results logged and traceable

### Documentation
- [ ] API documentation complete
- [ ] Training procedures documented
- [ ] Deployment guide written
- [ ] Known limitations documented

---

## Resources

### Datasets
- [AI Challenger VehicleID](https://github.com/ai-forever/AIAG)
- [OpenLPR Vietnam](https://github.com/openlpr-vn/dataset)
- [Vietnamese License Plates (custom collection)](#)

### Models
- YOLOv11: [Ultralytics](https://github.com/ultralytics/ultralytics)
- PaddleOCR: [PaddlePaddle](https://github.com/PaddlePaddle/PaddleOCR)

### Tools
- Annotation: [LabelImg](https://github.com/tzutalin/labelImg), [LabelStudio](https://labelstud.io/)
- Visualization: [CVAT](https://cvat.org/), [Supervisely](https://supervisely.com/)

---

## Appendix: Vietnamese Plate Regex Patterns

```python
# Private car: 30A-1234.56 or 51B-1234.56
PRIVATE_CAR_PATTERN = r'^\d{2}[A-Z]-\d{4}\.\d{2}$'

# Motorcycle: 43-12345 or 43-12345.12
MOTORCYCLE_PATTERN = r'^\d{2}-\d{5}(\.\d{2})?$'

# Police: 60-1234-56
POLICE_PATTERN = r'^\d{2}-\d{4}-\d{2}$'

# Army: 123456-78
ARMY_PATTERN = r'^\d{6}-\d{2}$'

# All patterns combined
ALL_PLATES_PATTERN = r'^(\d{2}[A-Z]-\d{4}\.\d{2}|\d{2}-\d{5}(\.\d{2})?|\d{2}-\d{4}-\d{2}|\d{6}-\d{2})$'
```
