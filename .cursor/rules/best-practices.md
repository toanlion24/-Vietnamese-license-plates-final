# Vietnamese LPR Best Practices

## Code Organization

### Module Structure

```
src/
├── detection/           # Detection module
│   ├── __init__.py
│   ├── detector.py      # Main class
│   ├── train.py        # Training script
│   └── utils.py        # Helper functions
├── recognition/         # Recognition module
│   ├── __init__.py
│   ├── recognizer.py   # Main class
│   └── train.py        # Training script
├── pipeline/            # Integration
│   ├── __init__.py
│   ├── inference.py    # Main pipeline
│   └── cli.py         # CLI interface
└── utils/              # Shared utilities
    ├── image_utils.py
    └── metrics.py
```

### Import Order

```python
# 1. Standard library
import time
import json
from pathlib import Path
from typing import List, Dict, Optional

# 2. Third-party
import cv2
import numpy as np
import torch

# 3. Local imports
from src.detection import PlateDetector
from src.recognition import PlateRecognizer
```

## Function Design

### Single Responsibility

```python
# Good
def detect_plates(image):
    detector = PlateDetector()
    return detector.detect(image)

def recognize_text(cropped_plate):
    recognizer = PlateRecognizer()
    return recognizer.recognize(cropped_plate)

# Bad
def process_everything(image):
    # Does detection, recognition, formatting, saving...
```

### Return Types

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PlateResult:
    plate: str
    confidence: float
    bbox: List[float]

def detect_and_recognize(image) -> List[PlateResult]:
    """Detect and recognize plates in image.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        List of PlateResult objects
    """
    pass
```

## Error Handling

### Graceful Degradation

```python
def process_image(image_path: str) -> List[PlateResult]:
    try:
        detector = PlateDetector()
        return detector.detect(image_path)
    except FileNotFoundError:
        logger.error(f"Image not found: {image_path}")
        return []
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return []
```

### Validation

```python
def validate_plate_text(text: str) -> bool:
    """Validate plate format."""
    if not text:
        return False
    if len(text) < 5 or len(text) > 15:
        return False
    # Add format validation
    return True
```

## Performance

### Batch Processing

```python
# Good
def process_batch(images: List[np.ndarray], batch_size: int = 8):
    results = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        results.extend(pipeline.process_batch(batch))
    return results

# Avoid
for image in images:
    results.append(pipeline.process_image(image))  # No batching
```

### Memory Management

```python
# Release GPU memory
import torch
torch.cuda.empty_cache()

# Use generators for large datasets
def image_generator(paths):
    for path in paths:
        yield cv2.imread(path)
```

## Testing

### Unit Tests

```python
def test_normalize_plate():
    assert normalize_vietnamese_plate("30A-1234.56") == "30A-1234.56"
    assert normalize_vietnamese_plate("30a-1234.56") == "30A-1234.56"

def test_validate_format():
    assert validate_plate_format("30A-1234.56", "car") == True
    assert validate_plate_format("invalid", "car") == False
```

### Integration Tests

```python
def test_pipeline():
    pipeline = VietnameseLPRPipeline()
    results = pipeline.process_image("tests/test_car.jpg")
    
    assert len(results) > 0
    assert results[0].plate == "30A-1234.56"
    assert results[0].confidence > 0.7
```

## Documentation

### Docstrings

```python
def detect_plates(
    image: np.ndarray,
    confidence: float = 0.25
) -> List[DetectionResult]:
    """Detect license plates in an image.
    
    Uses YOLOv11 for object detection to locate
    license plate regions in the input image.
    
    Args:
        image: Input image as numpy array (BGR format)
        confidence: Minimum confidence threshold (0-1)
        
    Returns:
        List of DetectionResult objects sorted by confidence
        
    Raises:
        ValueError: If image is None or invalid
        
    Example:
        >>> image = cv2.imread("car.jpg")
        >>> results = detect_plates(image)
        >>> print(f"Found {len(results)} plates")
    """
    pass
```

## Logging

### Structured Logging

```python
import logging

logger = logging.getLogger(__name__)

def process_plate(image_path: str):
    logger.info(
        "Processing plate",
        extra={
            "image_path": image_path,
            "timestamp": time.time()
        }
    )
    try:
        result = pipeline.process_image(image_path)
        logger.info(
            "Plate processed",
            extra={"plate": result.plate, "conf": result.confidence}
        )
        return result
    except Exception as e:
        logger.error(f"Failed to process: {e}")
        raise
```

## Configuration

### YAML Configuration

```yaml
# configs/pipeline.yaml
detection:
  model_path: "models/yolov11/best.pt"
  confidence_threshold: 0.25
  nms_threshold: 0.45

recognition:
  confidence_threshold: 0.5
  dictionary_path: "configs/vietnamese_dict.txt"
```

### Load Config

```python
import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

config = load_config("configs/pipeline.yaml")
```

## Git Workflow

### Commits

```
feat: add motorcycle plate detection
fix: correct OCR confusion between 0 and O
docs: update README with new API
perf: optimize detection for batch processing
test: add integration tests for pipeline
```

### Branch Naming

```
feature/motorcycle-recognition
fix/ocr-character-errors
refactor/pipeline-cleanup
experiment/transformer-recognition
```

## Security

### Input Validation

```python
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
MAX_SIZE = 10 * 1024 * 1024  # 10MB

def validate_image(file_path: str) -> bool:
    path = Path(file_path)
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False
    if path.stat().st_size > MAX_SIZE:
        return False
    return True
```

### No Hardcoded Secrets

```python
# Bad
API_KEY = "sk-1234567890"

# Good
API_KEY = os.environ.get("API_KEY")
```

## Performance Checklist

- [ ] Use batch processing for multiple images
- [ ] Pre-load models at startup
- [ ] Release GPU memory after use
- [ ] Use appropriate model size for use case
- [ ] Cache frequent operations
- [ ] Profile before optimizing
