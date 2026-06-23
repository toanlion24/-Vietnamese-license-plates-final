# Implementation Plan: Vietnamese License Plate Recognition

## Project Overview

**Mục tiêu:** Xây dựng hệ thống Nhận dạng Biển số xe Việt Nam (Vietnamese LPR) sử dụng YOLOv11 + PaddleOCR với kết quả rõ ràng và production-ready.

---

## Pipeline Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT (Image/Video)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: PREPROCESSING                                                     │
│  • CLAHE Enhancement (Contrast Limited Adaptive Histogram Equalization)     │
│  • Resize: Max 1920x1080, maintain aspect ratio                           │
│  • Normalize: ImageNet stats                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: YOLOv11 DETECTION                                                 │
│  • Model: yolov11s.pt (balanced speed/accuracy)                          │
│  • Output: [x1, y1, x2, y2, confidence, class_id]                         │
│  • Threshold: conf=0.25, iou=0.45                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: PLATE PROCESSING                                                  │
│  • Crop with 10% padding                                                   │
│  • Perspective transform for skewed plates                                 │
│  • Type classification (car, motorcycle, police, army)                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: PaddleOCR RECOGNITION                                             │
│  • Detection: DB (Differentiable Binarization)                              │
│  • Recognition: CRNN with CTC loss                                         │
│  • Character set: Vietnamese (full alphabet) + 0-9 + special chars        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: POST-PROCESSING                                                   │
│  • Text normalization (remove noise, correct OCR errors)                    │
│  • Format validation (regex patterns for each plate type)                  │
│  • Confidence scoring: combined = detection × recognition × format         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT                                                                      │
│  { plate: "30A-1234.56", confidence: 0.95, type: "private_car" }          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Vietnamese License Plate Formats

| Type | Pattern | Example | Characters |
|------|---------|---------|------------|
| **Private Car** | XX-YYYY.NN | 30A-1234.56 | 7-8 |
| **Motorcycle** | YY-NNNNN | 43-12345 | 6-7 |
| **Police** | XX-YYYY-NN | 60-1234-56 | 7-8 |
| **Army** | YYYYYY-NN | 123456-78 | 8-9 |

---

## Implementation Phases

### Phase 1: Environment Setup ✓

- [x] Create project structure
- [x] Create `requirements.txt`
- [x] Create configuration files
- [x] Create SKILL.md

**Files created:**
```
vietnamese-lpr/
├── SKILL.md                    # Main skill documentation
├── ARCHITECTURE.md             # Pipeline architecture
├── requirements.txt            # Dependencies
├── README.md                  # Project overview
├── configs/
│   ├── pipeline.yaml          # Main pipeline config
│   ├── detection.yaml         # YOLOv11 config
│   ├── recognition.yaml        # PaddleOCR config
│   └── vietnamese_dict.txt    # Vietnamese characters
├── src/
│   ├── detection/             # YOLOv11 module
│   ├── recognition/          # PaddleOCR module
│   ├── pipeline/              # End-to-end pipeline
│   ├── utils/                # Utilities
│   └── demo/                 # Demo interface
├── scripts/
│   └── download_models.py    # Model downloader
├── notebooks/
│   ├── 01_data_exploration.py
│   ├── 02_model_training.py
│   └── 03_pipeline_evaluation.py
└── tests/
    └── test_modules.py
```

---

### Phase 2: Core Modules ✓

| Task | Status | Description |
|------|--------|-------------|
| 2.1 | ✓ | `PlateDetector` - YOLOv11 wrapper |
| 2.2 | ✓ | `PlateRecognizer` - PaddleOCR wrapper |
| 2.3 | ✓ | `VietnameseLPRPipeline` - End-to-end pipeline |
| 2.4 | ✓ | Preprocessing utilities (CLAHE, perspective transform) |
| 2.5 | ✓ | Metrics and evaluation utilities |

---

### Phase 3: Data Preparation

**Checkpoint:** Before training, ensure you have:

```bash
# Required data structure
data/datasets/yolo_detection/
├── train/
│   ├── images/  (1000+ images)
│   └── labels/  (YOLO format)
├── val/
│   ├── images/  (200+ images)
│   └── labels/
└── test/
    ├── images/  (200+ images)
    └── labels/
```

**Tasks:**
- [ ] Collect Vietnamese license plate images
- [ ] Annotate with LabelImg (YOLO format)
- [ ] Split into train/val/test
- [ ] Verify annotation quality

---

### Phase 4: Model Training

#### 4.1 Detection Training (YOLOv11)

```bash
python -m src.detection.train \
    --data data/datasets/yolo_detection/data.yaml \
    --model yolov11s \
    --epochs 100 \
    --batch 16 \
    --device cuda
```

**Target metrics:**
- mAP@0.5: > 0.95
- mAP@0.5:0.95: > 0.85
- Inference time: < 10ms (GPU)

#### 4.2 Recognition Training (PaddleOCR)

```bash
# 1. Extract cropped plates
python scripts/prepare_ocr_data.py \
    --detection-model models/yolov11/best.pt \
    --output data/datasets/ocr_recognition/

# 2. Train OCR
python -m src.recognition.train \
    --config configs/recognition.yaml
```

**Target metrics:**
- Character accuracy: > 95%
- Word accuracy: > 90%
- Inference time: < 50ms (GPU)

---

### Phase 5: Pipeline Integration ✓

```python
from src.pipeline import VietnameseLPRPipeline

# Initialize
pipeline = VietnameseLPRPipeline(
    detector_weights="models/yolov11/best.pt",
    device="cuda"
)

# Single image
results = pipeline.process_image("test.jpg")

# Video
frame_results = pipeline.process_video("traffic.mp4")

# Camera
pipeline.process_camera(camera_id=0)
```

**Target end-to-end metrics:**
- Accuracy: > 85%
- Speed: < 50ms/image (GPU)
- FPS: > 20 (video)

---

### Phase 6: Evaluation & Benchmarking

```bash
python scripts/evaluate_pipeline.py \
    --test-data data/test/ \
    --output outputs/results/evaluation.json
```

**Required metrics report:**
| Metric | Target | Priority |
|--------|--------|----------|
| Detection mAP@0.5 | > 0.95 | HIGH |
| Recognition Accuracy | > 90% | HIGH |
| End-to-End Accuracy | > 85% | HIGH |
| Inference Speed (GPU) | < 50ms | MEDIUM |
| Video FPS | > 20 | MEDIUM |

---

### Phase 7: Deployment

Options:
1. **Gradio Demo:** `python -m src.demo.gradio_demo`
2. **CLI Tool:** `python -m src.pipeline.inference --image test.jpg`
3. **API Server:** FastAPI/Flask wrapper
4. **ONNX Export:** `torch.onnx.export()` for cross-platform

---

## Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download models
python scripts/download_models.py

# 3. Train detection
python -m src.detection.train --epochs 100 --device cuda

# 4. Run inference
python -m src.pipeline.inference --image test.jpg

# 5. Start demo
python -m src.demo.gradio_demo
```

---

## File Summary

| File | Purpose |
|------|---------|
| `SKILL.md` | Main skill documentation for AI agents |
| `ARCHITECTURE.md` | Detailed pipeline architecture |
| `README.md` | Project overview |
| `requirements.txt` | Python dependencies |
| `configs/pipeline.yaml` | Main configuration |
| `src/detection/detector.py` | YOLOv11 detection module |
| `src/recognition/recognizer.py` | PaddleOCR recognition module |
| `src/pipeline/inference.py` | End-to-end pipeline |
| `src/utils/image_utils.py` | Image processing utilities |
| `src/utils/metrics.py` | Evaluation metrics |
| `scripts/download_models.py` | Model download script |
| `src/demo/gradio_demo.py` | Web demo interface |

---

## Next Steps (Priority Order)

1. **Collect Dataset** - Most critical, need 2000+ images
2. **Annotate Data** - Use LabelImg for YOLO format
3. **Train Detection Model** - YOLOv11 fine-tuning
4. **Train Recognition Model** - PaddleOCR fine-tuning
5. **Evaluate & Optimize** - Benchmark and tune
6. **Deploy** - Choose deployment method

---

*Generated: June 2026 | Vietnamese LPR Project*
