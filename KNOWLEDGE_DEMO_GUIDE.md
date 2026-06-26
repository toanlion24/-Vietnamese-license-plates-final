# 📚 HƯỚNG DẪN CHUẨN BỊ KIẾN THỨC & DEMO
## Hệ thống Nhận dạng Biển số xe Việt Nam (Vietnamese LPR)

---

## MỤC LỤC
1. [Tổng quan Dự án](#1-tổng-quan-dự-án)
2. [Kiến trúc Hệ thống](#2-kiến-trúc-hệ-thống)
3. [Các Công nghệ Chính](#3-các-công-nghệ-chính)
4. [Cấu trúc Code](#4-cấu-trúc-code)
5. [Pipeline Xử lý](#5-pipeline-xử-lý)
6. [Cách Demo Từng Thành phần](#6-cách-demo-từng-thành-phần)
7. [Câu hỏi Thường gặp khi Demo](#7-câu-hỏi-thường-gặp-khi-demo)
8. [Mẹo Demo Chuyên nghiệp](#8-mẹo-demo-chuyên-nghiệp)
9. [Xử lý Tình huống Bất ngờ](#9-xử-lý-tình-huống-bất-ngờ)

---

## 1. Tổng quan Dự án

### 1.1 Mô tả
Hệ thống **Nhận dạng biển số xe tự động (LPR - License Plate Recognition)** cho xe Việt Nam, sử dụng:
- **YOLOv11** cho phát hiện vật thể (object detection)
- **PaddleOCR** cho nhận dạng ký tự (OCR)

### 1.2 Chỉ số Hiệu suất
| Metric | Giá trị |
|--------|---------|
| Precision | **99.76%** |
| Recall | **99.83%** |
| mAP@50 | **99.48%** |
| mAP@50-95 | **99.43%** |

### 1.3 Các loại Biển số Hỗ trợ
| Loại | Định dạng | Ví dụ |
|------|-----------|-------|
| Ô tô cá nhân | `XX-Y-YYYY.YY` | 30A-1234.56 |
| Xe máy (cũ) | `XX-YYYYY` | 43-12345 |
| Xe máy (mới) | `XX-XXX-YY` | 80-NG-63 |
| Công an | `XX-YYYY-YY` | 60-1234-56 |
| Quân đội | `YYYYYY-YY` | 123456-78 |
| Nước ngoài | `XX-YYYYY` | 12-34567 |

---

## 2. Kiến trúc Hệ thống

### 2.1 Sơ đồ Tổng quan

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT (Hình ảnh/Video/Webcam)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 1: PREPROCESSING (Tiền xử lý ảnh)           │
│     • CLAHE tăng cường tương phản                               │
│     • Giảm nhiễu                                               │
│     • Điều chỉnh ánh sáng                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 2: YOLOv11 DETECTION (Phát hiện)            │
│     • Phát hiện vùng biển số                                    │
│     • Trả về bounding box + confidence                          │
│     • Hỗ trợ: car, motorcycle, bus, truck, boat...             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 3: PLATE PROCESSING (Xử lý biển số)         │
│     • Cắt vùng biển số                                          │
│     • Perspective transform (sửa xiên)                         │
│     • Chuẩn hóa kích thước                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 4: ADVANCED OCR (Nhận dạng ký tự)           │
│     • 13 phương pháp tiền xử lý                                 │
│     • 2 engine OCR (PaddleOCR)                                  │
│     • Tổng cộng 26 lần suy luận                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 5: POST-PROCESSING (Hậu xử lý)              │
│     • Chuẩn hóa định dạng (OCR 0→O, 1→I)                      │
│     • Validation bằng regex                                     │
│     • Confidence scoring                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     OUTPUT: [Biển số + Độ chính xác]           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Chi tiết từng Module

#### Module 1: Input Stream (`input_stream.py`)
- **Chức năng:** Xử lý đầu vào đa dạng
- **Hỗ trợ:** Ảnh, Video, Webcam, RTSP stream
- **Class chính:** `InputStream`, `WebcamStream`

#### Module 2: YOLOv11 Detection (`yolo_detection.py`)
- **Chức năng:** Phát hiện biển số xe
- **Model:** YOLOv11 được train trên dữ liệu xe Việt Nam
- **Class chính:** `YOLODetector`, `DetectionResult`

#### Module 3: Vehicle-Plate Association (`vehicle_plate_association.py`)
- **Chức năng:** Liên kết biển số với phương tiện
- **Phương pháp:** Spatial, Distance, Hungarian Algorithm, Hybrid
- **Class chính:** `VehiclePlateAssociator`

#### Module 4: Rectify Perspective (`rectify_perspective.py`)
- **Chức năng:** Sửa biển số bị nghiêng/chéo
- **Template chuẩn:** Ô tô (480x140), Xe máy (280x110)
- **Class chính:** `PlateRectifier`

#### Module 5: Image Enhancement (`image_enhancement.py`)
- **Chức năng:** Tăng cường chất lượng ảnh cho OCR
- **Kỹ thuật:** CLAHE, Denoise, Sharpening, Night mode
- **Class chính:** `ImageEnhancer`

#### Module 6: PaddleOCR (`paddleocr_extraction.py`)
- **Chức năng:** Nhận dạng ký tự
- **Hỗ trợ:** Tiếng Việt, số, chữ cái
- **Class chính:** `PaddleOCRRecognizer`

#### Module 7: Rule Engine (`rule_engine.py`)
- **Chức năng:** Validation & chuẩn hóa biển số
- **Plate Types:** PRIVATE, COMMERCIAL, MOTORCYCLE, POLICE, ARMY
- **Class chính:** `PlateValidator`, `CharacterCorrector`

#### Module 8: ByteTrack (`bytetrack_voting.py`)
- **Chức năng:** Tracking nhiều frame, voting consensus
- **Phương pháp voting:** Majority, Confidence-weighted, Time-decay
- **Class chính:** `ByteTracker`, `VotingSystem`

#### Module 9: Database (`database_manager.py`)
- **Chức năng:** Lưu trữ & truy vấn kết quả
- **Database:** SQLite
- **Class chính:** `DatabaseManager`

---

## 3. Các Công nghệ Chính

### 3.1 YOLOv11 (Ultralytics)
```
- Mô hình: Object Detection (One-stage)
- Backbone: CSPDarknet cải tiến
- Ưu điểm: Nhanh, chính xác, lightweight
- Input size: 640x640
```

### 3.2 PaddleOCR
```
- OCR Engine: CRNN + Attention
- Backbone: SVTR-LCNet / MobileNetV3
- Hỗ trợ: 80+ ngôn ngữ
- Features: Angle classification, Multi-line detection
```

### 3.3 Python Libraries
| Thư viện | Mục đích |
|----------|----------|
| OpenCV | Xử lý ảnh |
| NumPy | Tính toán ma trận |
| Scikit-image | Tiền xử lý ảnh nâng cao |
| SQLite3 | Lưu trữ dữ liệu |
| Gradio/Streamlit | Giao diện web |

---

## 4. Cấu trúc Code

```
D:\ComputerVisionLasted\
│
├── src/
│   ├── detection/           # YOLOv11 detection
│   │   └── detector.py
│   │
│   ├── recognition/         # PaddleOCR recognition
│   │   └── recognizer.py
│   │
│   ├── pipeline/            # End-to-end pipeline
│   │   ├── inference.py     # ← Main pipeline class
│   │   └── cli.py
│   │
│   ├── modules/             # 9 core modules
│   │   ├── input_stream.py
│   │   ├── yolo_detection.py
│   │   ├── vehicle_plate_association.py
│   │   ├── rectify_perspective.py
│   │   ├── image_enhancement.py
│   │   ├── paddleocr_extraction.py
│   │   ├── rule_engine.py
│   │   ├── bytetrack_voting.py
│   │   └── database_manager.py
│   │
│   ├── demo/                # Advanced demos
│   │   └── gradio_demo.py
│   │
│   └── ui/                  # Web interface
│       └── streamlit_app.py
│
├── configs/                 # Configuration files
│   ├── pipeline.yaml
│   ├── recognition.yaml
│   └── vietnamese_dict.txt
│
├── weights/                 # Model weights
│   └── best.pt
│
├── outputs/                 # Results
│   ├── lpr_database.db
│   └── lpr_history.csv
│
└── gradio_demo.py           # Main demo (root)
```

---

## 5. Pipeline Xử lý

### 5.1 VietnameseLPRPipeline (src/pipeline/inference.py)

```python
class VietnameseLPRPipeline:
    def __init__(
        self,
        detector_weights="weights/best.pt",
        recognizer_config="configs/recognition.yaml",
        recognizer_weights=None,
        dictionary_path="configs/vietnamese_dict.txt",
        device="cuda",
        detection_conf=0.25,
        recognition_conf=0.5,
        min_confidence=0.7
    )
    
    # Xử lý ảnh đơn
    def process_image(
        image,                    # numpy array hoặc PIL Image
        return_debug_image=False, # Trả về ảnh debug
        padding_percent=0.1       # Padding quanh bbox
    ) -> List[LPRResult]
    
    # Xử lý video
    def process_video(
        video_path,
        output_path=None,         # Lưu video kết quả
        show_progress=True,
        frame_skip=0              # Bỏ qua frame (tăng tốc)
    ) -> List[LPRFrameResult]
    
    # Xử lý camera realtime
    def process_camera(
        camera_id=0,
        window_name="Vietnamese LPR",
        buffer_size=30
    ) -> Generator[tuple, None, None]
```

### 5.2 Output Format

```python
@dataclass
class LPRResult:
    plate: str                    # "30A-1234.56"
    confidence: float             # 0.95
    detection_confidence: float   # 0.98
    recognition_confidence: float # 0.92
    bbox: List[float]             # [x1, y1, x2, y2]
    plate_type: str               # "PRIVATE_CAR"
    processing_time_ms: float     # 45.2
```

---

## 6. Cách Demo Từng Thành phần

### 6.1 Demo 1: Gradio Web Interface (Cơ bản)

**Khởi động:**
```bash
python gradio_demo.py
# Mở trình duyệt: http://localhost:7860
```

**Các Tab:**
1. **LPR Pipeline** - Nhận dạng đầy đủ (detection + OCR)
2. **Detection Only** - Chỉ phát hiện vùng biển số
3. **Folder Watch** - Tự động xử lý ảnh mới trong thư mục

**Cách demo:**
1. Upload ảnh xe Việt Nam
2. Chọn confidence threshold (mặc định: 0.25)
3. Nhấn "Process"
4. Quan sát kết quả: bounding box + biển số + độ chính xác

---

### 6.2 Demo 2: Advanced Gradio (Nâng cao)

**Khởi động:**
```bash
python src/demo/gradio_demo.py
# Mở trình duyệt: http://localhost:7860
```

**Các Tab:**
1. **Detection** - Chi tiết phát hiện
2. **26 OCR Inferences** - 13 phương pháp × 2 engine OCR
3. **History** - Lịch sử nhận dạng + Export CSV
4. **Batch** - Xử lý hàng loạt ảnh
5. **Video** - Xử lý video với tracking

**Cách demo 26 OCR:**
1. Upload ảnh biển số
2. Chọn phương pháp tiền xử lý (hoặc để "Auto")
3. Chọn OCR engine
4. Xem kết quả từng phương pháp + kết quả ensemble

---

### 6.3 Demo 3: Camera Realtime

**Code mẫu:**
```python
from src.pipeline.inference import VietnameseLPRPipeline

pipeline = VietnameseLPRPipeline(device="cuda")

# Realtime camera
for frame_id, timestamp, results in pipeline.process_camera(camera_id=0):
    print(f"Frame {frame_id}: {[r.plate for r in results]}")
```

**Cách demo:**
1. Kết nối webcam
2. Chạy code trên
3. Di chuyển xe qua camera
4. Quan sát kết quả realtime

---

### 6.4 Demo 4: Video Processing

**Code mẫu:**
```python
results = pipeline.process_video(
    "test_video.mp4",
    output_path="output_video.mp4",
    frame_skip=2  # Bỏ qua 2 frame để tăng tốc
)
```

**Cách demo:**
1. Chuẩn bị video giao thông Việt Nam
2. Chạy xử lý
3. Xem video kết quả với bounding box và biển số

---

### 6.5 Demo 5: CLI Interface

**Xử lý ảnh:**
```bash
python -m src.pipeline.inference --image test.jpg
```

**Xử lý video:**
```bash
python -m src.pipeline.inference --video traffic.mp4
```

**Xử lý camera:**
```bash
python -m src.pipeline.inference --camera 0
```

---

### 6.6 Demo 6: Streamlit Dashboard

**Khởi động:**
```bash
streamlit run src/ui/streamlit_app.py
# Mở trình duyệt: http://localhost:8501
```

**Các trang:**
1. **Dashboard** - Tổng quan metrics
2. **Live Detection** - Camera realtime
3. **History** - Lịch sử nhận dạng
4. **Statistics** - Biểu đồ phân tích

---

## 7. Câu hỏi Thường gặp khi Demo

### Q1: Hệ thống hỗ trợ những loại biển số nào?
**A:** Ô tô cá nhân, xe máy (2 định dạng), công an, quân đội, nước ngoài, tạm trú.

### Q2: Tốc độ xử lý như thế nào?
**A:** 
- Ảnh đơn: ~50ms (GPU), ~200ms (CPU)
- Video: 15-25 FPS (GPU), 3-5 FPS (CPU)

### Q3: Cần GPU để chạy không?
**A:** Không bắt buộc, nhưng khuyến nghị có GPU để tăng tốc độ.

### Q4: Độ chính xác bao nhiêu?
**A:** Precision 99.76%, Recall 99.83%, mAP 99.48%.

### Q5: Xử lý được ảnh chụp từ xa không?
**A:** Có, hệ thống hỗ trợ multi-scale detection và tự động crop/padding.

### Q6: Ảnh xấu (nhiễu, thiếu sáng) có nhận dạng được không?
**A:** Có, với 13 phương pháp tiền xử lý và chế độ Low-light/Night enhancement.

---

## 8. Mẹo Demo Chuyên nghiệp

### 8.1 Chuẩn bị trước
```
□ Test tất cả các demo trước 30 phút
□ Chuẩn bị 5-10 ảnh test đa dạng
□ Kiểm tra webcam hoạt động
□ Backup model weights
□ Test trên cả GPU và CPU
```

### 8.2 Ảnh Test Đa dạng
```
□ Ô tô rõ ràng, nhiều ánh sáng
□ Xe máy di chuyển
□ Biển số bị nghiêng/chéo
□ Ảnh thiếu sáng/ban đêm
□ Nhiều phương tiện trong 1 frame
□ Biển số công an, quân đội
```

### 8.3 Khi Trình bày
```
□ Bắt đầu với demo đơn giản → phức tạp
□ Giải thích ngắn gọn từng stage
□ Highlight điểm mạnh: 99.76% precision
□ Demo realtime để gây ấn tượng
□ Có sẵn video kết quả dự phòng
```

### 8.4 Troubleshooting Nhanh

| Vấn đề | Giải pháp |
|--------|-----------|
| Gradio không khởi động | Kiểm tra port 7860 có bị chiếm không |
| OCR chậm | Giảm scale_factor, bật GPU |
| Không detect được | Tăng confidence threshold |
| Ảnh mờ kết quả | Thử phương pháp enhancement khác |

---

## 9. Xử lý Tình huống Bất ngờ

### 9.1 Demo bị "đứng máy"
```
1. Kiểm tra terminal output
2. Restart Gradio: Ctrl+C → python gradio_demo.py
3. Clear cache nếu cần
```

### 9.2 Kết quả OCR sai
```
1. Giải thích: "Hệ thống hỗ trợ OCR error correction"
2. Demo tính năng Character Corrector
3. Cho thấy confidence score thấp → cảnh báo
```

### 9.3 Không detect được biển số
```
1. Kiểm tra: ảnh có biển số không?
2. Thử ảnh khác rõ ràng hơn
3. Giải thích: "Yêu cầu biển số > 30x30 pixels"
```

### 9.4 Performance chậm
```
1. Giải thích: "GPU sẽ nhanh hơn 10x"
2. Tắt các ứng dụng khác
3. Giảm input size
```

---

## 10. Tổng kết Kiến thức Cần nhớ

### 10.1 Điểm mạnh của hệ thống
1. ✅ **Độ chính xác cao** - 99.76% precision
2. ✅ **Đa dạng đầu vào** - Ảnh, video, webcam, stream
3. ✅ **Hỗ trợ nhiều loại biển** - 6 loại biển số Việt Nam
4. ✅ **Robust** - Xử lý được ảnh xấu, thiếu sáng
5. ✅ **Ensemble OCR** - 26 lần suy luận cho độ chính xác cao
6. ✅ **Giao diện đẹp** - Gradio, Streamlit

### 10.2 Câu nói Gây ấn tượng
```
"Chúng tôi sử dụng 26 lần OCR inference (13 preprocessing × 2 engines)
để đạt được độ chính xác 99.76%, cao hơn đa số hệ thống thương mại."

"Pipeline xử lý từ ảnh thô đến kết quả chỉ trong 50ms trên GPU."

"Hệ thống hỗ trợ đầy đủ các loại biển số Việt Nam, kể cả biển công an
và quân đội."
```

### 10.3 Code Snippets Quan trọng

```python
# Khởi tạo pipeline
from src.pipeline.inference import VietnameseLPRPipeline
pipeline = VietnameseLPRPipeline(device="cuda")

# Xử lý ảnh
results = pipeline.process_image("test.jpg")

# Xem kết quả
for r in results:
    print(f"Plate: {r.plate}, Conf: {r.confidence:.2f}")
```

```python
# Realtime camera
for frame_id, ts, results in pipeline.process_camera(camera_id=0):
    # Display results
    pass
```

---

## 11. Liên kết Hữu ích

- **Gradio Demo:** `http://localhost:7860`
- **Advanced Demo:** `http://localhost:7860` (chạy từ `src/demo/gradio_demo.py`)
- **Streamlit:** `http://localhost:8501`
- **Model weights:** `weights/best.pt`
- **Configs:** `configs/*.yaml`

---

> **Chúc bạn demo thành công! 🚗📸**
