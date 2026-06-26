# 📋 COMMAND REFERENCE & API CHEAT SHEET

## Vietnamese LPR System - Terminal & Code Reference

---

## 1. TERMINAL COMMANDS

### 1.1 Demo Applications

```bash
# === GRADIO DEMOS ===

# Demo cơ bản (root)
python gradio_demo.py
# → Mở: http://localhost:7860

# Demo nâng cao (26 OCR + History)
python -m src.demo.gradio_demo
# → Mở: http://localhost:7860

# Demo đơn giản (nếu có)
python -m src.demo.simple_demo
```

### 1.2 Pipeline CLI

```bash
# === XỬ LÝ ẢNH ===

# Ảnh đơn
python -m src.pipeline.inference --image test.jpg

# Ảnh với output tùy chỉnh
python -m src.pipeline.inference --image test.jpg --output result.jpg

# Ảnh với threshold cao
python -m src.pipeline.inference --image test.jpg --conf 0.5

# === XỬ LÝ VIDEO ===

# Video cơ bản
python -m src.pipeline.inference --video traffic.mp4

# Video với output
python -m src.pipeline.inference --video traffic.mp4 --output output.mp4

# Video bỏ qua frame (nhanh hơn)
python -m src.pipeline.inference --video traffic.mp4 --frame-skip 2

# === CAMERA REALTIME ===

# Webcam mặc định
python -m src.pipeline.inference --camera 0

# Camera cụ thể
python -m src.pipeline.inference --camera 1

# === HELP ===
python -m src.pipeline.inference --help
```

### 1.3 Streamlit Dashboard

```bash
# Khởi động Streamlit
streamlit run src/ui/streamlit_app.py
# → Mở: http://localhost:8501

# Với cổng tùy chỉnh
streamlit run src/ui/streamlit_app.py --server.port 8502
```

### 1.4 Model Training (Colab)

```bash
# Training script
python scripts/train.py --data data.yaml --epochs 100 --model yolo11s

# Validation
python scripts/validate.py --weights weights/best.pt
```

### 1.5 Utility Commands

```bash
# === DATABASE ===

# Xem database
sqlite3 outputs/lpr_database.db
# SELECT * FROM recognitions LIMIT 10;

# Export CSV
python -c "import pandas as pd; pd.read_sql('SELECT * FROM recognitions', 'sqlite:///outputs/lpr_database.db').to_csv('export.csv')"

# === CLEAR CACHE ===
python -c "import shutil; shutil.rmtree('__pycache__', ignore_errors=True)"
```

### 1.6 Process Management

```bash
# === KILL PROCESS ===
# Windows
taskkill /F /PID <PID>

# Kill by port
Get-NetTCPConnection -LocalPort 7860 | Stop-Process -Force

# === CHECK PORT ===
netstat -ano | findstr :7860

# === PYTHON ===
# Check Python version
python --version

# Check installed packages
pip list | findstr paddle
pip list | findstr ultralytics
```

---

## 2. API REFERENCE - Các Class & Function Quan trọng

### 2.1 VietnameseLPRPipeline (src/pipeline/inference.py)

```python
from src.pipeline.inference import VietnameseLPRPipeline

# === KHỞI TẠO ===
pipeline = VietnameseLPRPipeline(
    detector_weights="weights/best.pt",      # Đường dẫn model YOLO
    recognizer_config="configs/recognition.yaml",
    dictionary_path="configs/vietnamese_dict.txt",
    device="cuda",                          # "cuda" hoặc "cpu"
    detection_conf=0.25,                    # Confidence threshold detection
    recognition_conf=0.5,                    # Confidence threshold OCR
    min_confidence=0.7                      # Confidence tối thiểu trả về
)

# === PROCESS IMAGE ===
results = pipeline.process_image(
    image,                          # np.ndarray hoặc PIL Image hoặc str(path)
    return_debug_image=False,       # True để trả về ảnh có vẽ bbox
    padding_percent=0.1             # Padding thêm quanh bbox
) -> List[LPRResult]

# Kết quả:
for r in results:
    print(r.plate)                 # "30A-1234.56"
    print(r.confidence)            # 0.95
    print(r.bbox)                  # [x1, y1, x2, y2]
    print(r.plate_type)            # "PRIVATE_CAR"

# === PROCESS VIDEO ===
results = pipeline.process_video(
    video_path="traffic.mp4",
    output_path="output.mp4",      # Lưu video kết quả
    show_progress=True,
    frame_skip=0                   # Bỏ qua frame (0 = không bỏ)
) -> List[LPRFrameResult]

# === PROCESS CAMERA ===
for frame_id, timestamp, results in pipeline.process_camera(
    camera_id=0,
    window_name="Camera",
    buffer_size=30
):
    print(f"Frame {frame_id}: {[r.plate for r in results]}")
```

### 2.2 PlateDetector (src/detection/detector.py)

```python
from src.detection import PlateDetector

# === KHỞI TẠO ===
detector = PlateDetector(
    model_path="weights/best.pt",
    confidence_threshold=0.25,      # Threshold cho detection
    nms_threshold=0.45,             # NMS threshold
    input_size=640,                # Kích thước input YOLO
    device="cuda"                  # "cuda" hoặc "cpu"
)

# === DETECT SINGLE ===
detections = detector.detect(image) -> List[DetectionResult]

for det in detections:
    print(det.bbox)           # [x1, y1, x2, y2]
    print(det.confidence)    # 0.95
    print(det.class_id)       # 0
    print(det.class_name)    # "license_plate"

# === DETECT WITH CROP ===
detections = detector.detect(
    image,
    return_cropped=True,      # Trả về ảnh đã crop
    padding_percent=0.1       # Padding thêm
)

# === DETECT BATCH ===
detections = detector.detect_batch(images) -> List[List[DetectionResult]]

# === DETECT FROM VIDEO FRAME ===
detections = detector.detect_from_video_frame(
    frame,
    max_detections=10
)
```

### 2.3 PlateRecognizer (src/recognition/recognizer.py)

```python
from src.recognition import PlateRecognizer

# === KHỞI TẠO ===
recognizer = PlateRecognizer(
    config_path="configs/recognition.yaml",
    model_path=None,                   # None = dùng pretrained
    dictionary_path="configs/vietnamese_dict.txt",
    use_angle_cls=True,                # Phân loại góc xoay
    use_gpu=True,
    lang="vi"
)

# === RECOGNIZE ===
results = recognizer.recognize(image) -> List[RecognitionResult]

# === RECOGNIZE PLATE ===
plate_text, confidence = recognizer.recognize_plate(
    plate_image,
    plate_type=None                    # None, "car", "motorcycle"
)
```

### 2.4 AdvancedLPROCRProcessor (src/modules/advanced_ocr.py)

```python
from src.modules.advanced_ocr import (
    AdvancedLPROCRProcessor,
    process_plate_fast,
    process_plate_full_ensemble,
    process_plate_standard
)

# === KHỞI TẠO ===
processor = AdvancedLPROCRProcessor(use_gpu=True)

# === PROCESS FAST (3 methods - NHANH) ===
result = processor.process_fast(plate_img) -> Dict

# === PROCESS ENSEMBLE (13 methods - CHÍNH XÁC CAO) ===
result = processor.process_ensemble(plate_img) -> Dict

# === PROCESS STANDARD (1 method) ===
result = processor.process_standard(plate_img, scale=4) -> Dict

# === CONVENIENCE FUNCTIONS ===
result = process_plate_fast(plate_img, use_gpu=True)
result = process_plate_full_ensemble(plate_img, use_gpu=True)
result = process_plate_standard(plate_img, scale=4, use_gpu=True)

# Kết quả:
result = {
    'raw_text': '30A-1234.56',           # Text gốc từ OCR
    'best_text': '30A-1234.56',          # Text tốt nhất
    'confidence': 0.95,                   # Độ chính xác
    'normalized_text': '30A-1234.56',    # Text đã chuẩn hóa
    'is_valid': True,                     # Có hợp lệ không
    'plate_type': 'PRIVATE_CAR',         # Loại biển số
    'province': 'Hà Nội',                # Tỉnh/thành (nếu có)
    'method': 'p_colab_3.0x',            # Phương pháp tốt nhất
    'all_candidates': [                  # Tất cả candidates
        {'method': 'p_colab_3.0x', 'text': '30A-1234.56', 'conf': 0.95},
        {'method': 'p_adaptive', 'text': '30A-123456', 'conf': 0.90},
        ...
    ],
    'total_inferences': 26,              # Tổng số lần suy luận
    'errors': []                          # Danh sách lỗi
}
```

### 2.5 ImageEnhancer (src/modules/image_enhancement.py)

```python
from src.modules.image_enhancement import ImageEnhancer

# === KHỞI TẠO ===
enhancer = ImageEnhancer()

# === ENHANCE FOR OCR ===
enhanced = enhancer.enhance_for_ocr(
    image,
    target_height=48,
    target_width=320,
    apply_clahe=True,
    apply_denoise=True,
    apply_sharpen=True
)

# === ENHANCE LOW LIGHT ===
enhanced = enhancer.enhance_low_light(image)

# === ENHANCE NIGHT ===
enhanced = enhancer.enhance_night(image)

# === ANALYZE QUALITY ===
quality = enhancer.analyze_image_quality(image)
# quality = {
#     'brightness': 0.65,
#     'contrast': 0.45,
#     'sharpness': 0.55,
#     'is_blurry': False,
#     'is_dark': False
# }
```

### 2.6 PlateRectifier (src/modules/rectify_perspective.py)

```python
from src.modules.rectify_perspective import PlateRectifier

# === KHỞI TẠO ===
rectifier = PlateRectifier()

# === CORRECT SKEW ===
corrected = rectifier.correct_skew(
    plate_image,
    angle=None          # None = tự động detect
)

# === NORMALIZE SIZE ===
normalized = rectifier.normalize_size(
    plate_image,
    plate_type="car"    # "car" (480x140) hoặc "motorcycle" (280x110)
)

# === CORRECT WITH TEMPLATE ===
corrected = rectifier.correct_with_template(
    plate_image,
    template="car"
)
```

### 2.7 PlateValidator (src/modules/rule_engine.py)

```python
from src.modules.rule_engine import PlateValidator

# === KHỞI TẠO ===
validator = PlateValidator()

# === VALIDATE ===
validation = validator.validate("30A-1234.56")

print(validation.is_valid)           # True
print(validation.normalized_text)    # "30A-1234.56"
print(validation.plate_type)         # PlateType.PRIVATE_CAR
print(validation.province)           # "Hà Nội"
print(validation.errors)             # []

# === GET PROVINCE ===
province = validator.get_province("30A-1234.56")
# "Hà Nội"

# === IS VALID FORMAT ===
is_valid = validator.is_valid_format("30A-1234.56", "PRIVATE_CAR")
# True
```

### 2.8 ByteTracker (src/modules/bytetrack_voting.py)

```python
from src.modules.bytetrack_voting import ByteTracker, VotingSystem

# === BYTE TRACKER ===
tracker = ByteTracker(
    max_time_lost=30,          # Frame tối đa không thấy
    min_box_area=100,           # Diện tích bbox tối thiểu
    min_confidence=0.5
)

# Track objects
tracks = tracker.update(
    detections,    # List[DetectionResult]
    frame          # Current frame
)

for track in tracks:
    print(track.track_id)      # ID duy nhất
    print(track.bbox)          # Vị trí
    print(track.confidence)    # Confidence
    print(track.hits)          # Số lần được track

# === VOTING SYSTEM ===
voter = VotingSystem(
    method="confidence"         # "majority", "confidence", "decay"
)

# Vote for best plate
best_plate, final_conf = voter.vote(
    observations   # List of (plate, confidence, timestamp)
)
```

### 2.9 DatabaseManager (src/modules/database_manager.py)

```python
from src.modules.database_manager import DatabaseManager

# === KHỞI TẠO ===
db = DatabaseManager("outputs/lpr_database.db")

# === INSERT ===
record = RecognitionRecord(
    plate_text="30A-1234.56",
    confidence=0.95,
    plate_type="PRIVATE_CAR",
    province="Hà Nội",
    image_path="test.jpg"
)
db_id = db.insert_recognition(record)

# === QUERY ===
records = db.get_recent_recognitions(limit=10)
plates = db.search_by_plate("30A")
stats = db.get_statistics()

# === DELETE ===
db.delete_old_records(days=30)
```

---

## 3. DATACLASSES & ENUMS

### 3.1 LPRResult (Output chính)

```python
@dataclass
class LPRResult:
    plate: str                         # "30A-1234.56"
    confidence: float                  # 0.95
    detection_confidence: float         # 0.98
    recognition_confidence: float       # 0.92
    bbox: List[float]                  # [x1, y1, x2, y2]
    plate_type: Optional[str]          # "PRIVATE_CAR"
    processing_time_ms: float           # 45.2
```

### 3.2 DetectionResult

```python
@dataclass
class DetectionResult:
    bbox: List[float]                  # [x1, y1, x2, y2]
    confidence: float                  # 0.95
    class_id: int                      # 0
    class_name: str = "license_plate"  # Tên class
```

### 3.3 PlateType Enum

```python
from src.modules.rule_engine import PlateType

PlateType.PRIVATE_CAR    # Ô tô cá nhân
PlateType.COMMERCIAL     # Xe thương mại
PlateType.MOTORCYCLE     # Xe máy
PlateType.POLICE         # Công an
PlateType.ARMY           # Quân đội
PlateType.FOREIGN        # Nước ngoài
PlateType.TEMPORARY      # Tạm trú
```

---

## 4. CONFIG FILES

### 4.1 pipeline.yaml

```yaml
# Detection settings
detection:
  model_path: weights/best.pt
  confidence_threshold: 0.25
  nms_threshold: 0.45
  input_size: 640
  device: cuda

# Recognition settings
recognition:
  config_path: configs/recognition.yaml
  use_gpu: true
  lang: vi

# Preprocessing
preprocessing:
  apply_clahe: true
  apply_denoise: true
  padding_percent: 0.1

# Postprocessing
postprocessing:
  min_confidence: 0.7
  enable_validation: true
  enable_correction: true
```

### 4.2 recognition.yaml

```yaml
# PaddleOCR config
ocr:
  use_angle_cls: true
  lang: en
  use_gpu: true
  
# Recognition algorithm
rec_algorithm: SVTR_LCNet

# Thresholds
det_db_thresh: 0.1
det_db_box_thresh: 0.3
```

---

## 5. QUICK REFERENCE

### 5.1 Import Pattern

```python
# Standard import
from src.pipeline.inference import VietnameseLPRPipeline
from src.detection import PlateDetector
from src.recognition import PlateRecognizer
from src.modules.advanced_ocr import AdvancedLPROCRProcessor
from src.modules.image_enhancement import ImageEnhancer
from src.modules.rule_engine import PlateValidator, PlateType
from src.modules.bytetrack_voting import ByteTracker, VotingSystem
from src.modules.database_manager import DatabaseManager, RecognitionRecord
```

### 5.2 Minimal Example

```python
from src.pipeline.inference import VietnameseLPRPipeline

# Khởi tạo
pipeline = VietnameseLPRPipeline(device="cuda")

# Xử lý ảnh
results = pipeline.process_image("car.jpg")

# In kết quả
for r in results:
    print(f"Plate: {r.plate}, Conf: {r.confidence:.2%}")
```

### 5.3 Performance Tips

```python
# GPU nhanh hơn CPU ~10x
pipeline = VietnameseLPRPipeline(device="cuda")  # Khuyến nghị

# Bỏ qua frame để tăng tốc video
pipeline.process_video("video.mp4", frame_skip=2)

# Batch processing cho nhiều ảnh
for img_path in image_paths:
    results = pipeline.process_image(img_path)
```

---

## 6. ERROR HANDLING

```python
try:
    pipeline = VietnameseLPRPipeline()
    results = pipeline.process_image("test.jpg")
    
    if not results:
        print("Không phát hiện biển số")
    else:
        for r in results:
            if r.confidence < 0.7:
                print(f"Cảnh báo: {r.plate} có confidence thấp")
                
except Exception as e:
    print(f"Lỗi: {e}")
```

---

## 7. FILE PATHS

```
D:\ComputerVisionLasted\
├── weights/
│   └── best.pt              # YOLOv11 model
├── configs/
│   ├── pipeline.yaml        # Pipeline config
│   ├── recognition.yaml     # OCR config
│   └── vietnamese_dict.txt  # Dictionary
├── outputs/
│   ├── lpr_database.db      # SQLite database
│   ├── lpr_history.csv      # CSV export
│   └── video_result.mp4     # Video output
├── src/
│   ├── detection/           # YOLO detector
│   ├── recognition/         # OCR recognizer
│   ├── pipeline/            # Main pipeline
│   ├── modules/             # 9 modules
│   ├── demo/                # Gradio demos
│   └── ui/                  # Streamlit UI
└── gradio_demo.py           # Main demo entry
```

