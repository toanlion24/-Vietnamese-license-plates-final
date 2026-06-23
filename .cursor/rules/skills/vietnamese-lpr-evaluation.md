# Vietnamese LPR - Evaluation & Benchmarking Skill

---
name: vietnamese-lpr-evaluation
description: >
  Evaluation and benchmarking for LPR pipeline.
  Use when measuring accuracy, speed, or comparing models.
---

## Overview

Comprehensive evaluation framework for Vietnamese LPR including detection, recognition, and end-to-end metrics.

## Evaluation Module

```
src/utils/metrics.py
├── calculate_detection_metrics()
├── calculate_recognition_metrics()
├── calculate_lpr_metrics()
└── generate_evaluation_report()
```

## Quick Evaluation

```python
from src.pipeline import VietnameseLPRPipeline
from src.utils.metrics import calculate_lpr_metrics, generate_evaluation_report

pipeline = VietnameseLPRPipeline()

# Process test set
detection_results = []
ground_truth = []
processing_times = []

for image, gt in test_data:
    start = time.time()
    results = pipeline.process_image(image)
    processing_times.append(time.time() - start)
    
    detection_results.append(results[0] if results else {})
    ground_truth.append(gt)

# Calculate metrics
metrics = calculate_lpr_metrics(
    detection_results,
    ground_truth,
    processing_times
)

# Generate report
report = generate_evaluation_report(metrics)
print(report)
```

## Metrics

### Detection Metrics

```python
@dataclass
class DetectionMetrics:
    precision: float      # TP / (TP + FP)
    recall: float         # TP / (TP + FN)
    f1_score: float       # 2 * P * R / (P + R)
    map50: float          # mAP at IoU=0.5
    map50_95: float      # mAP at IoU=0.5:0.95
```

### Recognition Metrics

```python
@dataclass
class RecognitionMetrics:
    accuracy: float           # Exact match rate
    character_accuracy: float # Character-level accuracy
    total_samples: int        # Number of test samples
    correct_samples: int      # Correct predictions
    mean_confidence: float    # Average confidence
    confidence_std: float     # Confidence std dev
```

### LPR Metrics

```python
@dataclass
class LPRMetrics:
    detection: DetectionMetrics
    recognition: RecognitionMetrics
    end_to_end_accuracy: float     # Full plate match
    avg_processing_time_ms: float  # Latency
    fps: float                     # Frames per second
```

## Performance Profiling

```python
from src.utils.metrics import PerformanceProfiler

profiler = PerformanceProfiler()

for image in images:
    # Profile each stage
    t0 = time.time()
    detections = detector.detect(image)
    profiler.record('detection', (time.time() - t0) * 1000)
    
    t0 = time.time()
    for det in detections:
        cropped = crop_plate(image, det)
        result = recognizer.recognize(cropped)
    profiler.record('recognition', (time.time() - t0) * 1000)

# Get summary
summary = profiler.get_summary()
print(summary)
```

## Benchmarking

### Inference Speed

```python
import time

# Warmup
for _ in range(10):
    pipeline.process_image("test.jpg")

# Benchmark
times = []
for _ in range(100):
    start = time.time()
    pipeline.process_image("test.jpg")
    times.append(time.time() - start)

print(f"Mean: {np.mean(times)*1000:.2f} ms")
print(f"Std:  {np.std(times)*1000:.2f} ms")
print(f"FPS:  {1/np.mean(times):.1f}")
```

### Video Processing

```python
import cv2

cap = cv2.VideoCapture("traffic.mp4")
fps_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    start = time.time()
    pipeline.process_image(frame)
    fps = 1 / (time.time() - start)
    fps_list.append(fps)

print(f"Average FPS: {np.mean(fps_list):.1f}")
```

## Test Data Requirements

| Metric | Minimum Samples | Recommended |
|--------|-----------------|-------------|
| Detection | 500 images | 2000 images |
| Recognition | 500 plates | 2000 plates |
| End-to-End | 300 images | 1000 images |

## Ground Truth Format

```json
[
  {
    "image": "test001.jpg",
    "plates": [
      {
        "bbox": [100, 50, 300, 100],
        "text": "30A-1234.56",
        "type": "private_car"
      }
    ]
  }
]
```

## Evaluation Checklist

### Detection
- [ ] mAP@0.5 > 0.95
- [ ] Precision > 0.90
- [ ] Recall > 0.88
- [ ] No systematic false positives

### Recognition
- [ ] Character accuracy > 95%
- [ ] Word accuracy > 90%
- [ ] Mean confidence > 0.85
- [ ] Handles worn/damaged plates

### End-to-End
- [ ] Full plate match > 85%
- [ ] Processing time < 50ms (GPU)
- [ ] Video FPS > 20

## Report Template

```markdown
# Vietnamese LPR Evaluation Report

## Test Set
- Total images: 500
- Total plates: 523
- Date: 2024-01-15

## Detection Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| mAP@0.5 | 0.96 | >0.95 | ✓ |
| Precision | 0.95 | >0.90 | ✓ |
| Recall | 0.94 | >0.88 | ✓ |

## Recognition Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Char Accuracy | 96.2% | >95% | ✓ |
| Word Accuracy | 91.5% | >90% | ✓ |
| Mean Confidence | 0.92 | >0.85 | ✓ |

## Performance
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| End-to-End | 88.3% | >85% | ✓ |
| Latency | 42ms | <50ms | ✓ |
| FPS | 23.8 | >20 | ✓ |

## Issues
- None significant
```

## Red Flags

- Reporting only accuracy without detection metrics
- Ignoring confidence calibration
- No per-plate-type breakdown
- Missing latency measurements
- No failure case analysis
