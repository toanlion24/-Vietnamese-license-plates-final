# Vietnamese LPR - Modules Index

## Pipeline Modules

| Module | File | Description | Status |
|--------|------|-------------|--------|
| 1. Input Stream | `modules/input_stream.py` | Image, Video, Webcam handling | ✅ **Done** |
| 2. YOLOv11 Detection | `modules/yolo_detection.py` | Vehicle & plate detection | ✅ |
| 3. Vehicle-Plate Association | `modules/vehicle_plate_association.py` | Link plates to vehicles | ✅ |
| 4. Rectify & Perspective | `modules/rectify_perspective.py` | Perspective correction | ✅ |
| 5. Image Enhancement | `modules/image_enhancement.py` | CLAHE, denoise, sharpen | ✅ |
| 6. PaddleOCR Extraction | `modules/paddleocr_extraction.py` | Character recognition | ✅ |
| 7. Rule Engine & Regex | `modules/rule_engine.py` | Plate validation | ✅ |
| 8. ByteTrack & Voting | `modules/bytetrack_voting.py` | Multi-frame tracking | ✅ |
| 9. Database & Streamlit UI | `modules/database_manager.py` + `ui/` | Storage & web interface | ✅ |

---

## Module Details

### Module 1: Input Stream
**File:** `src/modules/input_stream.py`

```python
from src.modules import InputStream, WebcamStream, Frame, load_image, load_video_info, list_cameras

# From image
stream = InputStream("test.jpg")
frame = stream.read()
print(f"Shape: {frame.image.shape}")

# From video
stream = InputStream("video.mp4")
for frame in stream:
    process(frame)

# From webcam
with WebcamStream(camera_id=0) as cam:
    frame = cam.read()

# Utility functions
img = load_image("test.jpg")
info = load_video_info("video.mp4")
cameras = list_cameras(max_cameras=5)
```

**Features:**
- ✅ Load image files (jpg, png, etc.)
- ✅ Video streaming with FPS control
- ✅ Webcam with resolution settings
- ✅ Frame metadata (frame_id, timestamp)
- ✅ Iterator support for easy looping
- ✅ Context manager support

**Input:** Image file, Video file, Webcam/RTSP stream
**Output:** Frame with metadata (frame_id, timestamp, source)

---

### Module 2: YOLOv11 Detection
**File:** `src/modules/yolo_interactive.py` + `src/detection/detector.py`

```python
from src.modules.yolo_interactive import YOLOInteractiveDetector, quick_detect, DetectionMode

# Cách 1: Quick detect - đơn giản nhất
result, cropped = quick_detect("image.jpg", confidence=0.25)

# Cách 2: Tùy chỉnh nhiều tham số
detector = YOLOInteractiveDetector(
    model_path="weights/best.pt",
    confidence=0.25,
    iou_threshold=0.45,
    device="cuda",
    detection_mode=DetectionMode.PLATE_ONLY
)

# Từ image
result = detector.detect_from_image("test.jpg", return_cropped=True)

# Từ video (100 frames)
results = detector.detect_from_video("video.mp4", max_frames=100)

# Từ webcam (10 frames)
results = detector.detect_from_webcam(0, max_frames=10, callback=lambda r: print(r.plates))

# Từ folder
results = detector.detect_from_folder("test_images/")

# Save results
detector.save_results(results, "output.json")

# Access results
for plate in result.plates:
    print(f"Bbox: {plate.xyxy}")
    print(f"Conf: {plate.confidence}")
```

**Input:** RGB Frame (image/video/webcam/folder)
**Output:** BoundingBox (xe, biển số) + cropped images

---

### Module 3: Vehicle-Plate Association
**File:** `src/modules/vehicle_plate_association.py`

```python
from src.modules import VehiclePlateAssociator

associator = VehiclePlateAssociator(method="spatial")
result = associator.associate(detections)

for pair in result.pairs:
    print(f"Vehicle: {pair.vehicle.class_name}")
    print(f"Plate: {pair.plate.bbox}")
```

**Input:** Danh sách tọa độ các Box
**Output:** Cặp liên kết (Phương tiện - Biển số)

---

### Module 4: Rectify & Perspective
**File:** `src/modules/rectify_perspective.py`

```python
from src.modules import PlateRectifier

rectifier = PlateRectifier(target_width=480, target_height=140)
result = rectifier.rectify(image, bbox)
rectified = result.image
```

**Input:** Ảnh vùng biển số bị cắt
**Output:** Ảnh biển số đã căn phẳng

---

### Module 5: Image Enhancement
**File:** `src/modules/image_enhancement.py`

```python
from src.modules import ImageEnhancer

enhancer = ImageEnhancer()
enhanced = enhancer.enhance_for_ocr(plate_image)

# Or specific presets
enhanced = enhancer.enhance_low_light(plate_image)
enhanced = enhancer.enhance_night(plate_image)
```

**Input:** Ảnh biển số chuẩn hóa
**Output:** Ảnh độ tương phản cao, khử nhiễu

---

### Module 6: PaddleOCR Extraction
**File:** `src/modules/paddleocr_extraction.py`

```python
from src.modules import PaddleOCRRecognizer

ocr = PaddleOCRRecognizer(lang="vi", use_gpu=False)
result = ocr.recognize(enhanced_plate)

print(f"Text: {result.full_text}")
print(f"Confidence: {result.avg_confidence}")
```

**Features:**
- PaddleOCR 2.10 support
- Vietnamese language model (lang='vi')
- Character-level confidence tracking
- Batch processing support

**Input:** Ảnh biển số đã tăng cường
**Output:** Chuỗi văn bản thô (Raw Text) + confidence score

**Demo Scripts:**
```bash
# OCR Demo
python scripts/ocr_demo.py

# Full LPR Demo (Detection + OCR)
python scripts/lpr_demo.py --image path/to/image.jpg
```

---

### Module 7: Rule Engine & Regex
**File:** `src/modules/rule_engine.py`

```python
from src.modules import PlateValidator, validate_vietnamese_plate

validator = PlateValidator()
result = validator.validate("30A-1234.56")

print(f"Valid: {result.is_valid}")
print(f"Type: {result.plate_type}")
print(f"Province: {result.province}")
```

**Supported Vietnamese Plate Formats:**
- **Private Car:** `30A-1234.56` (2 digits + letter + hyphen + 4 digits + dot + 2 digits)
- **Motorcycle (Old):** `43-12345` (2 digits + hyphen + 5 digits)
- **Motorcycle (New):** `80-NG-63` (2 digits + hyphen + 2-3 letters + hyphen + 2 digits)
- **Police:** `60-1234-56` (2 digits + hyphen + 4 digits + hyphen + 2 digits)
- **Army:** `123456-78` (6 digits + hyphen + 2 digits)

**Input:** Chuỗi văn bản thô
**Output:** Chuỗi ký tự biển số chuẩn hóa

---

### Module 8: ByteTrack & Voting
**File:** `src/modules/bytetrack_voting.py`

```python
from src.modules import ByteTracker, VotingSystem

tracker = ByteTracker()
voting = VotingSystem(method="majority")

# Process each frame
tracks = tracker.update(detections)

for track in tracks:
    result = voting.vote(track)
    print(f"Final: {result.final_plate}")
```

**Input:** Clean Text + Vehicle ID qua từng frame
**Output:** Kết quả nhận diện tối ưu nhất

---

### Module 9: Database & Streamlit UI
**File:** `src/modules/database_manager.py` + `src/ui/streamlit_app.py`

```python
from src.modules import DatabaseManager, create_record_from_result

db = DatabaseManager()

# Save result
record = create_record_from_result(
    plate_text="30A-1234.56",
    confidence=0.95,
    ...
)
db.insert_recognition(record)

# Query
recent = db.get_recent_recognitions(limit=100)
stats = db.get_statistics()
```

**Input:** Kết quả tối ưu + Ảnh phương tiện
**Output:** Lịch sử lưu SQLite, cập nhật Streamlit UI

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT STREAM                                     │
│                         Image/Video/Webcam                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        YOLOv11 DETECTION                                    │
│                    BoundingBox (xe, biển số)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VEHICLE-PLATE ASSOCIATION                                │
│                     Cặp liên kết (Phương tiện - Biển số)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   RECTIFY & PERSPECTIVE CORRECTION                           │
│                   Ảnh biển số đã căn phẳng                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        IMAGE ENHANCEMENT                                     │
│                     Ảnh độ tương phản cao, khử nhiễu                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PaddleOCR EXTRACTION                                 │
│                        Chuỗi văn bản thô                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RULE ENGINE & REGEX                                   │
│                    Chuỗi ký tự biển số chuẩn hóa                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BYTETRACK & VOTING                                   │
│                   Kết quả nhận diện tối ưu nhất                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATABASE & UI STORAGE                                 │
│                      Lịch sử lưu SQLite + Streamlit                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Install PaddleOCR (required for Module 6)
python -m pip install paddlepaddle==2.6.2 "paddleocr>=2.7,<3.0" "albumentations<1.5"

# Run OCR Demo (standalone)
python scripts/ocr_demo.py

# Run Full LPR Demo (Detection + OCR)
python scripts/lpr_demo.py --image path/to/image.jpg

# Run Streamlit UI
streamlit run src/ui/streamlit_app.py

# Run individual modules
python -c "
from src.modules import *
print('All modules imported successfully!')
"
```
