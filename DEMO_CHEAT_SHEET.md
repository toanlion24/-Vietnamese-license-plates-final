# 📋 HƯỚNG DẪN DEMO NHANH - GIẤY NOTE

## 🚀 Khởi động Demo

```bash
# Demo cơ bản
python gradio_demo.py

# Demo nâng cao (26 OCR)
python -m src.demo.gradio_demo
```

**Mở trình duyệt:** http://localhost:7860

---

## 🎯 6 Điểm USP để Trình bày

| # | Điểm mạnh | Nói gì |
|---|-----------|--------|
| 1 | **99.76% Precision** | "Độ chính xác 99.76%, cao hơn đa số hệ thống thương mại" |
| 2 | **26 OCR Inferences** | "13 phương pháp × 2 engine = 26 lần suy luận để chọn kết quả tốt nhất" |
| 3 | **Đa dạng biển số** | "Hỗ trợ 6 loại: Ô tô, xe máy, công an, quân đội, nước ngoài, tạm trú" |
| 4 | **Xử lý ảnh xấu** | "Tự động phát hiện ảnh mờ/thiếu sáng và chọn phương pháp phù hợp" |
| 5 | **Realtime** | "50ms/ảnh trên GPU, hỗ trợ webcam trực tiếp" |
| 6 | **Tracking** | "ByteTrack + Voting ổn định kết quả qua nhiều frame" |

---

## 🎬 Script Demo

### 1. Demo Upload Ảnh (30s)
```
1. Upload ảnh xe Việt Nam
2. Nhấn "🤖 Tự động nhận diện"
3. Chờ kết quả (~3-5s)
4. Giải thích: "Biển số + confidence + thời gian xử lý"
```

### 2. Demo So sánh Methods (1 phút)
```
1. Upload ảnh khó (mờ/tối)
2. Tab "26 OCR Inferences"
3. Show từng candidate
4. Giải thích: "Ensemble chọn kết quả có confidence cao nhất"
```

### 3. Demo Camera Realtime (1 phút)
```python
# Code để demo
from src.pipeline.inference import VietnameseLPRPipeline
pipeline = VietnameseLPRPipeline()
for frame_id, ts, results in pipeline.process_camera(0):
    print([r.plate for r in results])
```

### 4. Demo Video (30s)
```
1. Upload video giao thông
2. Nhấn "▶️ Xử lý video"
3. Show video kết quả với bounding box
4. Giải thích: "Tracking + lọc nhiễu"
```

---

## 🔧 Troubleshooting

| Vấn đề | Fix |
|--------|-----|
| Gradio không mở | `python -m src.demo.gradio_demo` |
| OCR sai | "Hệ thống có OCR error correction tự động" |
| Không detect | "Tăng confidence threshold" |
| Chậm | "Dùng GPU sẽ nhanh hơn 10x" |

---

## 📊 Performance Numbers

| Metric | Value |
|--------|-------|
| Precision | 99.76% |
| Recall | 99.83% |
| mAP@50 | 99.48% |
| mAP@50-95 | 99.43% |
| Speed (GPU) | 50ms/ảnh |
| Speed (CPU) | 200ms/ảnh |

---

## 📝 Ghi nhớ Quan trọng

### Định dạng Biển số
```
Ô tô:      30A-1234.56
Xe máy:    43-12345 hoặc 80-NG-63  
Công an:   60-1234-56
Quân đội:  123456-78
```

### Pipeline Flow
```
Ảnh → CLAHE → YOLOv11 → Crop → Perspective → 26 OCR → Validation → Kết quả
```

---

## 🎤 Câu Nói Kết

> "Hệ thống của chúng tôi sử dụng ensemble learning với 26 lần OCR inference để đạt 99.76% precision, cao hơn đa số giải pháp thương mại, trong khi vẫn đảm bảo tốc độ xử lý realtime trên GPU."

---

## 📱 Liên hệ

- **Demo:** http://localhost:7860
- **Weights:** weights/best.pt
- **Source:** src/pipeline/inference.py
