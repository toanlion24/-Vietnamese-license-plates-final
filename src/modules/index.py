"""
═══════════════════════════════════════════════════════════════════════════════════
                        VIETNAMESE LPR - MODULES INDEX
═══════════════════════════════════════════════════════════════════════════════════

Cấu trúc thư mục:
  src/
  └── modules/
      ├── __init__.py          # Package init + exports
      ├── index.py             # This file - module overview
      │
      ├── input_stream.py      📦 Module 1: Input Stream
      ├── yolo_interactive.py  📦 Module 2: YOLOv11 Detection (NEW)
      ├── yolo_detection.py    📦 Module 2: YOLOv11 Detection (Legacy)
      │
      ├── vehicle_plate_association.py  📦 Module 3: Association
      ├── rectify_perspective.py        📦 Module 4: Rectify
      ├── image_enhancement.py          📦 Module 5: Enhancement
      ├── paddleocr_extraction.py       📦 Module 6: OCR
      ├── rule_engine.py                📦 Module 7: Validation
      ├── bytetrack_voting.py           📦 Module 8: Tracking
      └── database_manager.py            📦 Module 9: Database

═══════════════════════════════════════════════════════════════════════════════════
                                    PIPELINE
═══════════════════════════════════════════════════════════════════════════════════

┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │
│  │   M1    │───▶│   M2    │───▶│   M3    │───▶│   M4    │───▶│   M5    │       │
│  │  Input  │    │ Detect  │    │Associate│    │Rectify  │    │Enhance  │       │
│  │  Stream │    │  (YOLO) │    │  Link   │    │Correct  │    │  Image  │       │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘       │
│       │              │              │              │              │           │
│       │              │              │              │              ▼           │
│       │              │              │              │        ┌─────────┐       │
│       │              │              │              └───────▶│   M6    │       │
│       │              │              │                       │   OCR   │       │
│       │              │              │                       └─────────┘       │
│       │              │              │                             │           │
│       │              │              ▼                             ▼           │
│       │              │        ┌─────────┐                   ┌─────────┐       │
│       │              └───────▶│   M7    │                   │   M8    │       │
│       │                       │  Rule   │                   │  Track  │       │
│       │                       │ Engine │                   │ Voting  │       │
│       │                       └─────────┘                   └─────────┘       │
│       │                                                         │           │
│       │                                                         ▼           │
│       │                                                 ┌─────────┐         │
│       └────────────────────────────────────────────────▶│   M9    │         │
│                                                         │Database │         │
│                                                         └─────────┘         │
└───────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════
                                  MODULE DETAILS
═══════════════════════════════════════════════════════════════════════════════════

┌───────────────────────────────────────────────────────────────────────────────┐
│ MODULE 1: INPUT STREAM                                                      │
│ File: src/modules/input_stream.py                                           │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   from src.modules import InputStream, WebcamStream, Frame                   │
│                                                                              │
│   # Image                                                                  │
│   stream = InputStream("test.jpg")                                          │
│   frame = stream.read()                                                     │
│                                                                              │
│   # Video                                                                  │
│   stream = InputStream("video.mp4")                                         │
│   for frame in stream:                                                      │
│       process(frame.image)                                                  │
│                                                                              │
│   # Webcam                                                                  │
│   with WebcamStream(camera_id=0) as cam:                                    │
│       frame = cam.read()                                                    │
│                                                                              │
│ INPUT:  Image file, Video file, Webcam/RTSP stream                          │
│ OUTPUT: Frame with metadata (frame_id, timestamp, image)                    │
│                                                                              │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│ MODULE 2: YOLOv11 DETECTION                                                 │
│ File: src/modules/yolo_interactive.py  ⚡ RECOMMENDED                        │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   from src.modules import YOLOInteractiveDetector, quick_detect               │
│                                                                              │
│   # Quick detect                                                            │
│   result, cropped = quick_detect("image.jpg", confidence=0.25)              │
│                                                                              │
│   # Full control                                                            │
│   detector = YOLOInteractiveDetector(                                      │
│       model_path="weights/best.pt",                                        │
│       confidence=0.25,                                                      │
│       iou_threshold=0.45,                                                  │
│       device="cuda"                                                         │
│   )                                                                        │
│                                                                              │
│   # Từ image/video/webcam/folder                                            │
│   result = detector.detect_from_image("test.jpg")                          │
│   results = detector.detect_from_video("video.mp4", max_frames=100)        │
│   results = detector.detect_from_webcam(0, max_frames=10)                  │
│                                                                              │
│   # Access results                                                          │
│   for plate in result.plates:                                               │
│       print(f"Bbox: {plate.xyxy}, Conf: {plate.confidence}")               │
│                                                                              │
│ INPUT:  RGB Frame (image/video/webcam)                                      │
│ OUTPUT: BoundingBox (vehicles, plates) + cropped images                     │
│                                                                              │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│ MODULE 3: VEHICLE-PLATE ASSOCIATION                                         │
│ File: src/modules/vehicle_plate_association.py                              │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   from src.modules import VehiclePlateAssociator                            │
│                                                                              │
│   associator = VehiclePlateAssociator(method="spatial")                      │
│   pairs = associator.associate(vehicle_boxes, plate_boxes)                   │
│                                                                              │
│   for pair in pairs:                                                         │
│       print(f"Vehicle: {pair.vehicle.class_name}")                          │
│       print(f"Plate: {pair.plate.bbox}")                                   │
│                                                                              │
│ INPUT:  Danh sách tọa độ vehicle boxes + plate boxes                        │
│ OUTPUT: Cặp liên kết (vehicle ↔ plate)                                     │
│                                                                              │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│ MODULE 4: RECTIFY & PERSPECTIVE                                             │
│ File: src/modules/rectify_perspective.py                                    │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   from src.modules import PlateRectifier                                    │
│                                                                              │
│   rectifier = PlateRectifier(target_width=480, target_height=140)            │
│   result = rectifier.rectify(image, bbox)                                   │
│   rectified = result.image                                                  │
│                                                                              │
│ INPUT:  Ảnh vùng biển số bị cắt                                            │
│ OUTPUT: Ảnh biển số đã căn phẳng (perspective correction)                   │
│                                                                              │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│ MODULE 5: IMAGE ENHANCEMENT                                                 │
│ File: src/modules/image_enhancement.py                                      │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   from src.modules import ImageEnhancer                                      │
│                                                                              │
│   enhancer = ImageEnhancer()                                                │
│   enhanced = enhancer.enhance_for_ocr(plate_image)                          │
│                                                                              │
│   # Các preset                                                              │
│   enhanced = enhancer.enhance_low_light(plate_image)                        │
│   enhanced = enhancer.enhance_night(plate_image)                           │
│                                                                              │
│ INPUT:  Ảnh biển số chuẩn hóa                                               │
│ OUTPUT: Ảnh độ tương phản cao, khử nhiễu                                   │
│                                                                              │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│ MODULE 6: PADDLEOCR EXTRACTION                                              │
│ File: src/modules/paddleocr_extraction.py                                   │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   from src.modules import PaddleOCRRecognizer                               │
│                                                                              │
│   ocr = PaddleOCRRecognizer(lang="vi", use_gpu=True)                        │
│   result = ocr.recognize(enhanced_plate)                                    │
│                                                                              │
│   print(f"Text: {result.full_text}")                                       │
│   print(f"Conf: {result.avg_confidence}")                                   │
│                                                                              │
│ INPUT:  Ảnh biển số đã tăng cường                                          │
│ OUTPUT: Chuỗi văn bản thô (raw OCR text)                                   │
│                                                                              │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│ MODULE 7: RULE ENGINE & REGEX                                                │
│ File: src/modules/rule_engine.py                                           │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   from src.modules import PlateValidator, validate_vietnamese_plate          │
│                                                                              │
│   validator = PlateValidator()                                              │
│   result = validator.validate("30A-1234.56")                               │
│                                                                              │
│   print(f"Valid: {result.is_valid}")                                       │
│   print(f"Type: {result.plate_type}")                                      │
│   print(f"Province: {result.province}")                                    │
│                                                                              │
│ INPUT:  Chuỗi văn bản thô từ OCR                                           │
│ OUTPUT: Chuỗi ký tự biển số chuẩn hóa                                      │
│                                                                              │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│ MODULE 8: BYTETRACK & VOTING                                                │
│ File: src/modules/bytetrack_voting.py                                      │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   from src.modules import ByteTracker, VotingSystem                          │
│                                                                              │
│   tracker = ByteTracker()                                                   │
│   voting = VotingSystem(method="majority")                                  │
│                                                                              │
│   for frame_detections in all_frames:                                       │
│       tracks = tracker.update(frame_detections)                             │
│       for track in tracks:                                                 │
│           result = voting.vote(track)                                       │
│           print(f"Final: {result.final_plate}")                            │
│                                                                              │
│ INPUT:  Clean text + vehicle ID qua từng frame                              │
│ OUTPUT: Kết quả nhận diện tối ưu nhất (voting)                             │
│                                                                              │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│ MODULE 9: DATABASE & STORAGE                                                 │
│ File: src/modules/database_manager.py                                      │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   from src.modules import DatabaseManager, create_record_from_result        │
│                                                                              │
│   db = DatabaseManager()                                                    │
│                                                                              │
│   # Save result                                                              │
│   record = create_record_from_result(                                       │
│       plate_text="30A-1234.56",                                             │
│       confidence=0.95                                                       │
│   )                                                                        │
│   db.insert_recognition(record)                                             │
│                                                                              │
│   # Query                                                                   │
│   recent = db.get_recent_recognitions(limit=100)                           │
│   stats = db.get_statistics()                                               │
│                                                                              │
│ INPUT:  Kết quả tối ưu + ảnh phương tiện                                   │
│ OUTPUT: Lịch sử lưu SQLite + query/analytics                              │
│                                                                              │
└───────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════
                                  QUICK START
═══════════════════════════════════════════════════════════════════════════════════

```bash
# Import tất cả modules
from src.modules import *

# Xem danh sách modules
list_modules()

# Quick demo Module 2
python src/modules/yolo_interactive.py
```

═══════════════════════════════════════════════════════════════════════════════════
"""

# Import list_modules from __init__
from . import list_modules

__doc__ = __doc__  # Use this file as module docstring
