# Vietnamese LPR Debugging Guide

## Common Issues and Solutions

### Detection Issues

#### No plates detected
```
Symptom: detector.detect() returns empty list
```

**Causes:**
1. Confidence threshold too high
2. Image quality too low
3. Model not trained on similar data

**Solutions:**
```python
# Lower threshold
detector = PlateDetector(confidence_threshold=0.15)

# Enhance image
from src.utils import apply_clahe
enhanced = apply_clahe(image)
detections = detector.detect(enhanced)
```

#### Too many false positives
```
Symptom: Detects non-plate objects
```

**Solutions:**
1. Increase confidence threshold
2. Filter by box size
```python
detections = [d for d in detections if d.area > 1000]
```

#### Wrong bounding boxes
```
Symptom: Boxes don't align with plates
```

**Check:**
1. Training data annotations
2. Image preprocessing
3. Model checkpoint quality

### Recognition Issues

#### Empty recognition result
```
Symptom: recognizer.recognize() returns []
```

**Solutions:**
```python
# Preprocess first
from src.recognition import preprocess_for_recognition
processed = preprocess_for_recognition(cropped_plate, enhance=True)
results = recognizer.recognize(processed)
```

#### Wrong characters
```
Symptom: "30A" recognized as "3OA"
```

**Solutions:**
```python
# Normalize output
from src.recognition import normalize_vietnamese_plate
text = normalize_vietnamese_plate(raw_text)
```

#### Low confidence
```
Symptom: confidence < 0.7
```

**Check:**
1. Image quality
2. Contrast
3. Character clarity

### Pipeline Issues

#### GPU Out of Memory
```
Symptom: CUDA out of memory error
```

**Solutions:**
```python
# Reduce batch size
pipeline = VietnameseLPRPipeline(device='cpu')

# Or use smaller model
pipeline = VietnameseLPRPipeline(
    detector_weights="models/yolov11/yolov11n.pt"
)
```

#### Slow inference
```
Symptom: > 100ms per image
```

**Solutions:**
1. Use GPU instead of CPU
2. Use smaller model (yolov11n)
3. Reduce input size
4. Enable batch processing

### Training Issues

#### Loss is NaN
```
Symptom: training loss becomes nan
```

**Solutions:**
1. Check data for corruption
2. Reduce learning rate (lr0=0.0001)
3. Check annotation format
4. Verify class IDs are 0-indexed

#### Model not converging
```
Symptom: loss plateaus or increases
```

**Solutions:**
1. Increase learning rate
2. Use different optimizer
3. Check data quality
4. Verify train/val split

#### Overfitting
```
Symptom: train loss decreases but val loss increases
```

**Solutions:**
1. Add regularization
2. Use data augmentation
3. Reduce model size
4. Early stopping

## Debugging Workflow

### Step 1: Isolate the Issue

```python
# Test detection separately
detections = detector.detect(image)
print(f"Detections: {len(detections)}")

# Test recognition separately
if detections:
    cropped = crop_from_detection(image, detections[0])
    results = recognizer.recognize(cropped)
    print(f"Recognition: {results}")
```

### Step 2: Check Intermediate Results

```python
# Enable debug output
import cv2
cv2.imwrite("debug_original.jpg", image)
cv2.imwrite("debug_cropped.jpg", cropped)
cv2.imwrite("debug_processed.jpg", processed)
```

### Step 3: Verify Configuration

```python
# Print config
import yaml
with open("configs/pipeline.yaml") as f:
    config = yaml.safe_load(f)
    print(config)
```

## Error Messages

| Error | Meaning | Fix |
|-------|---------|-----|
| `CUDA out of memory` | GPU RAM exceeded | Reduce batch size |
| `Model not found` | Wrong path | Check model path |
| `Invalid annotation` | Wrong YOLO format | Fix label file |
| `Image read failed` | Corrupt file | Remove bad images |
| `Dimension mismatch` | Size incompatibility | Check preprocessing |

## Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in pipeline
pipeline = VietnameseLPRPipeline()
# Check console for debug output
```

## Performance Debugging

```python
import time

# Profile pipeline stages
start = time.time()
detections = detector.detect(image)
t1 = time.time()
cropped = crop_plates(image, detections)
t2 = time.time()
results = recognizer.recognize_batch(cropped)
t3 = time.time()

print(f"Detection: {(t1-start)*1000:.1f}ms")
print(f"Cropping: {(t2-t1)*1000:.1f}ms")
print(f"Recognition: {(t3-t2)*1000:.1f}ms")
```
