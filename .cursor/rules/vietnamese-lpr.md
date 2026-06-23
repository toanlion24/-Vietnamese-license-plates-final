# Vietnamese LPR - Cursor Agent Rules

## Project Context

You are working on a Vietnamese License Plate Recognition (LPR) project using YOLOv11 + PaddleOCR. This project follows a specific pipeline architecture and coding conventions.

## Pipeline Architecture

```
Input Image → Preprocessing (CLAHE) → YOLOv11 Detection → Plate Processing → PaddleOCR Recognition → Post-processing (Validation) → Output
```

### Stage Responsibilities

1. **Detection** (`src/detection/`): YOLOv11 for plate localization
2. **Recognition** (`src/recognition/`): PaddleOCR for character recognition
3. **Pipeline** (`src/pipeline/`): End-to-end integration
4. **Utils** (`src/utils/`): Shared utilities

## Vietnamese Plate Formats

```
Private Car:  XX-YYYY.NN  (e.g., 30A-1234.56)
Motorcycle:   YY-NNNNN    (e.g., 43-12345)
Police:       XX-YYYY-NN  (e.g., 60-1234-56)
Army:         YYYYYY-NN   (e.g., 123456-78)
```

## Coding Conventions

### File Naming
- Modules: `snake_case.py`
- Classes: `PascalCase`
- Functions/methods: `snake_case()`
- Constants: `UPPER_SNAKE_CASE`

### Module Structure
```python
"""
Module docstring
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Dataclasses first
@dataclass
class Result:
    """Result structure"""
    text: str
    confidence: float

# Then functions
def process_image(image: np.ndarray) -> List[Result]:
    """Process single image"""
    pass
```

### Type Hints
- Always use type hints for function parameters and return values
- Use `Optional[X]` instead of `X | None`
- Use `List[X]`, `Dict[K, V]` from typing

### Error Handling
```python
try:
    result = pipeline.process_image(image)
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    return []
except RuntimeError as e:
    logger.error(f"Pipeline error: {e}")
    raise
```

### Documentation
- All public classes/functions must have docstrings
- Document parameters, return values, and exceptions
- Use Google-style docstrings

## Quality Gates

Before considering a PR ready:
- [ ] All type hints present
- [ ] Docstrings on public APIs
- [ ] Unit tests pass
- [ ] No hardcoded paths (use configs)
- [ ] Performance < 50ms per image on GPU

## Imports Order

1. Standard library
2. Third-party (cv2, np, torch, etc.)
3. Local project imports

```python
# 1. stdlib
import time
from typing import List

# 2. third-party
import cv2
import numpy as np
from ultralytics import YOLO

# 3. local
from src.detection import PlateDetector
```

## Configuration

All settings must come from YAML configs, not hardcoded:
- `configs/pipeline.yaml` - Main pipeline
- `configs/detection.yaml` - YOLOv11 settings
- `configs/recognition.yaml` - PaddleOCR settings

## Performance Targets

| Metric | Target |
|--------|--------|
| Detection mAP@0.5 | > 0.95 |
| Recognition Accuracy | > 90% |
| End-to-End Accuracy | > 85% |
| Inference Time (GPU) | < 50ms |
