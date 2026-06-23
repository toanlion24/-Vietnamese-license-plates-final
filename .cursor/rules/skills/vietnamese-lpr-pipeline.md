# Vietnamese LPR - Complete Pipeline Skill

---
name: vietnamese-lpr-pipeline
description: >
  Complete pipeline implementation for Vietnamese License Plate Recognition.
  Use when building, modifying, or debugging the LPR pipeline.
---

## Overview

This skill guides the implementation and modification of the Vietnamese LPR pipeline using YOLOv11 for detection and PaddleOCR for recognition.

## When to Use

- Building the LPR pipeline from scratch
- Modifying detection or recognition stages
- Debugging pipeline issues
- Adding new plate types
- Optimizing inference speed
- Converting to different deployment format

**When NOT to use:** General Python development, unrelated computer vision tasks.

## Pipeline Structure

```
src/pipeline/
├── inference.py      # Main pipeline class
├── cli.py           # Command-line interface
└── __init__.py

src/detection/
├── detector.py      # PlateDetector class
├── train.py         # YOLOv11 training
└── __init__.py

src/recognition/
├── recognizer.py    # PlateRecognizer class
├── train.py         # OCR training
└── __init__.py
```

## Process

### Step 1: Initialize Pipeline

```python
from src.pipeline import VietnameseLPRPipeline

pipeline = VietnameseLPRPipeline(
    detector_weights="models/yolov11/best.pt",
    device="cuda",
    min_confidence=0.7,
)
```

### Step 2: Process Input

**Single Image:**
```python
results = pipeline.process_image("test.jpg")
# Returns: List[LPRResult]

for result in results:
    print(f"Plate: {result.plate}, Confidence: {result.confidence}")
```

**Video:**
```python
frame_results = pipeline.process_video("traffic.mp4", show_progress=True)
```

**Camera:**
```python
pipeline.process_camera(camera_id=0)
```

### Step 3: Handle Results

```python
@dataclass
class LPRResult:
    plate: str                    # "30A-1234.56"
    confidence: float             # 0.95
    detection_confidence: float   # 0.98
    recognition_confidence: float # 0.97
    bbox: List[float]             # [x1, y1, x2, y2]
    plate_type: str               # "private_car"
    processing_time_ms: float     # 42.5
```

## Adding New Plate Types

### Step 1: Add Pattern

Edit `configs/pipeline.yaml`:

```yaml
postprocessing:
  patterns:
    new_type: '^XX-YYYY.NN$'
```

### Step 2: Update Classification

Edit `src/pipeline/inference.py`:

```python
def _classify_plate_type(self, plate_text: str) -> Optional[str]:
    patterns = {
        'private_car': r'^\d{2}[A-Z]-\d{4}\.\d{2}$',
        'motorcycle': r'^\d{2}-\d{5}(\.\d{2})?$',
        'new_type': r'^XX-YYYY.NN$',  # Add here
    }
    # ... validation logic
```

### Step 3: Update Dict

Add characters to `configs/vietnamese_dict.txt` if needed.

## Optimization

### Speed Optimization

```python
# Use smaller model for speed
pipeline = VietnameseLPRPipeline(
    detector_weights="models/yolov11/yolov11n.pt",  # nano
    device="cuda",
)

# Use batch processing
results = pipeline.process_batch(images)
```

### Accuracy Optimization

```python
# Use larger model
pipeline = VietnameseLPRPipeline(
    detector_weights="models/yolov11/yolov11l.pt",  # large
    device="cuda",
    min_confidence=0.5,  # Lower threshold
)
```

### Memory Optimization

```python
# Process in batches
for batch in chunks(images, batch_size=8):
    results = pipeline.process_batch(batch)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No plates detected | Check confidence threshold, image quality |
| Wrong text | Check OCR model, preprocess image |
| Slow inference | Use smaller model, batch processing |
| GPU OOM | Reduce batch size, use CPU fallback |

## Verification

Verify pipeline with test images:

```bash
python -m src.pipeline.inference \
    --image tests/test_images/car1.jpg \
    --output outputs/results.jpg
```

Expected output:
- Bounding box on plate
- Correct text (e.g., "30A-1234.56")
- Confidence score > 0.7

## Rationalizations

| Rationalization | Reality |
|----------------|---------|
| "I'll skip validation" | Always validate pipeline outputs |
| "Default threshold is fine" | Tune threshold for your use case |
| "Batch doesn't matter" | Batch size affects speed/accuracy trade-off |

## Red Flags

- Hardcoded confidence values
- Missing error handling
- No logging
- Direct OpenCV without preprocessing
- Ignoring plate type differences
