# 📦 Vietnamese LPR - Modules

## Cấu trúc

```
src/modules/
├── __init__.py              # Package init + exports
├── index.py                 # Module index (this file)
│
├── input_stream.py          📦 Module 1: Input Stream
├── yolo_interactive.py      📦 Module 2: YOLOv11 Detection ⚡
├── yolo_detection.py        📦 Module 2: YOLOv11 (Legacy)
│
├── vehicle_plate_association.py  📦 Module 3: Association
├── rectify_perspective.py         📦 Module 4: Rectify
├── image_enhancement.py           📦 Module 5: Enhancement
├── paddleocr_extraction.py        📦 Module 6: OCR
├── rule_engine.py                  📦 Module 7: Validation
├── bytetrack_voting.py            📦 Module 8: Tracking
└── database_manager.py             📦 Module 9: Database
```

## Pipeline Flow

```
Image/Video/Webcam
       │
       ▼
┌─────────────┐
│ 1. Input    │ ─── Frame (image + metadata)
└─────────────┘
       │
       ▼
┌─────────────┐
│ 2. Detect   │ ─── BoundingBox (vehicles, plates)
└─────────────┘
       │
       ▼
┌─────────────┐
│ 3. Associate │ ─── Vehicle-Plate pairs
└─────────────┘
       │
       ▼
┌─────────────┐
│ 4. Rectify  │ ─── Flattened plate image
└─────────────┘
       │
       ▼
┌─────────────┐
│ 5. Enhance  │ ─── Enhanced plate image
└─────────────┘
       │
       ▼
┌─────────────┐
│ 6. OCR      │ ─── Raw text
└─────────────┘
       │
       ▼
┌─────────────┐
│ 7. Validate │ ─── Clean plate number
└─────────────┘
       │
       ▼
┌─────────────┐
│ 8. Track    │ ─── Optimized result (voting)
└─────────────┘
       │
       ▼
┌─────────────┐
│ 9. Store    │ ─── SQLite database
└─────────────┘
```

## Quick Start

### Import

```python
from src.modules import *
```

### Module 1: Input Stream

```python
from src.modules import InputStream, WebcamStream

# Image
stream = InputStream("test.jpg")
frame = stream.read()

# Video
for frame in InputStream("video.mp4"):
    process(frame.image)

# Webcam
with WebcamStream(0) as cam:
    frame = cam.read()
```

### Module 2: Detection (Recommended)

```python
from src.modules import quick_detect, YOLOInteractiveDetector

# Quick - 1 dòng
result, cropped = quick_detect("image.jpg")

# Full control
detector = YOLOInteractiveDetector(
    model_path="weights/best.pt",
    confidence=0.25,
    device="cuda"
)

# Từ image/video/webcam/folder
result = detector.detect_from_image("test.jpg")
results = detector.detect_from_video("video.mp4", max_frames=100)
results = detector.detect_from_webcam(0, max_frames=10)
results = detector.detect_from_folder("images/")
```

### Module 3-9: Các module khác

```python
from src.modules import (
    VehiclePlateAssociator,  # Module 3
    PlateRectifier,           # Module 4
    ImageEnhancer,            # Module 5
    PaddleOCRRecognizer,      # Module 6
    PlateValidator,           # Module 7
    ByteTracker,              # Module 8
    DatabaseManager,          # Module 9
)
```

## Module Status

| # | Module | File | Status |
|---|--------|------|--------|
| 1 | Input Stream | `input_stream.py` | ✅ |
| 2 | YOLOv11 Detection | `yolo_interactive.py` | ✅ |
| 3 | Vehicle-Plate Association | `vehicle_plate_association.py` | ✅ |
| 4 | Rectify & Perspective | `rectify_perspective.py` | ✅ |
| 5 | Image Enhancement | `image_enhancement.py` | ✅ |
| 6 | PaddleOCR | `paddleocr_extraction.py` | ✅ |
| 7 | Rule Engine | `rule_engine.py` | ✅ |
| 8 | ByteTrack & Voting | `bytetrack_voting.py` | ✅ |
| 9 | Database | `database_manager.py` | ✅ |
