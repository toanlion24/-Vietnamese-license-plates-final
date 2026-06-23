# Vietnamese LPR - Detection Module Skill

---
name: vietnamese-lpr-detection
description: >
  YOLOv11 plate detection module. Use when modifying detection,
  training, or debugging detection issues.
---

## Overview

The detection module uses YOLOv11 to localize license plates in images. It wraps Ultralytics YOLO for easy integration with the pipeline.

## Module Structure

```
src/detection/
├── detector.py    # PlateDetector class
├── train.py      # Training script
└── __init__.py
```

## PlateDetector Class

```python
from src.detection import PlateDetector

detector = PlateDetector(
    model_path="models/yolov11/best.pt",
    confidence_threshold=0.25,
    nms_threshold=0.45,
    input_size=640,
    device="cuda"
)
```

## Methods

### detect()

```python
detections = detector.detect(
    image,                    # np.ndarray or path
    return_cropped=False,      # Return cropped plates
    padding_percent=0.1        # Padding around box
)
```

Returns: `List[DetectionResult]`

### detect_batch()

```python
detections = detector.detect_batch(images)
```

Returns: `List[List[DetectionResult]]`

### detect_from_video_frame()

Optimized for video processing:
```python
detections = detector.detect_from_video_frame(frame, max_detections=10)
```

## DetectionResult Dataclass

```python
@dataclass
class DetectionResult:
    bbox: List[float]      # [x1, y1, x2, y2]
    confidence: float       # Detection confidence
    class_id: int          # Usually 0 for plate
    class_name: str        # "license_plate"
    
    @property
    def xywh(self) -> List[float]:
        """Convert to [x_center, y_center, width, height]"""
        
    @property
    def area(self) -> float:
        """Bounding box area"""
```

## Training

### Quick Training

```bash
python -m src.detection.train \
    --data data/datasets/yolo_detection/data.yaml \
    --model yolov11s \
    --epochs 100 \
    --device cuda
```

### Configuration

Edit `configs/detection.yaml`:

```yaml
model:
  name: yolov11s
  pretrained: true

training:
  epochs: 100
  batch_size: 16
  imgsz: 640
  optimizer: AdamW
  lr0: 0.001
  
augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 5.0
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0
```

## Data Format

YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```

Example (one plate in image):
```
0 0.512 0.483 0.234 0.156
```

## Evaluation

### Metrics

| Metric | Target | Acceptable |
|--------|--------|------------|
| mAP@0.5 | > 0.95 | > 0.90 |
| mAP@0.5:0.95 | > 0.85 | > 0.80 |
| Precision | > 0.95 | > 0.90 |
| Recall | > 0.93 | > 0.88 |

### Manual Evaluation

```python
from src.detection import visualize_detections
import cv2

detections = detector.detect("test.jpg")
visualized = visualize_detections(image, detections)
cv2.imwrite("output.jpg", visualized)
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| No detections | Low confidence | Reduce threshold to 0.15 |
| Many false positives | Low threshold | Increase to 0.35 |
| Slow inference | Large model | Use yolov11n or yolov11s |
| Wrong boxes | Training issue | Check annotations |
| Small boxes | Distance | Use higher resolution |

## Model Selection

| Model | Speed | Accuracy | GPU Memory |
|-------|-------|----------|------------|
| yolov11n | 2.5x | 0.85 mAP | 2GB |
| yolov11s | 1.5x | 0.92 mAP | 4GB |
| yolov11m | 1.0x | 0.95 mAP | 8GB |
| yolov11l | 0.7x | 0.97 mAP | 12GB |

**Recommendation:** Start with yolov11s, upgrade if needed.

## Red Flags

- Training without validation set
- Using mAP < 0.90
- Ignoring inference speed
- No data augmentation
- Single epoch training
