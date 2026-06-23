# Vietnamese LPR - Full Pipeline Demo

Gradio demo cho nhận diện biển số xe Việt Nam với **full LPR pipeline**.

## Advanced OCR Pipeline

```
Input Plate Image
       │
       ▼
┌─────────────────────────────────────────────────┐
│  PREPROCESSING (7 Methods)                       │
├─────────────────────────────────────────────────┤
│  1. Colab Style 2.5x (CLAHE + Sharpen)         │
│  2. Colab Style 3.0x                           │
│  3. Colab Style 4.0x                           │
│  4. Dark Condition Gamma 2.5                    │
│  5. Dark Condition Gamma 3.0                    │
│  6. Grayscale + Binarize                        │
│  7. Low-light Enhancement                       │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  ENSEMBLE OCR (2 Engines)                       │
├─────────────────────────────────────────────────┤
│  Engine 1: PaddleOCR SVTR_LCNet                 │
│  Engine 2: PaddleOCR CRNN                       │
│                                                 │
│  Total: 7 preprocessing × 2 engines = 14 runs   │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  BEST SELECTION                                 │
│  - Select result with highest confidence         │
│  - Validate against plate format rules           │
│  - Return top 3 candidates for review            │
└─────────────────────────────────────────────────┘
       │
       ▼
Output: [Best Text, Confidence, All Candidates]
```

## Model Performance

| Metric | Training Value |
|:------:|:--------------:|
| **Precision** | 99.76% |
| **Recall** | 99.83% |
| **mAP@50** | 99.48% |
| **mAP@50-95** | 99.43% |

**Training Details:**
- Model: YOLO11s (pretrained)
- Epochs: 100
- Optimizer: AdamW
- Batch size: 16
- Image size: 640x640

## Chạy demo

```bash
# 1. Cài dependencies
pip install ultralytics gradio opencv-python paddleocr paddlepaddle

# 2. Chạy demo
python gradio_demo.py
```

Server sẽ chạy tại: **http://localhost:7860**

## Tính năng

### Tab 1: Full LPR Pipeline (YOLO + OCR)
- **Detection**: YOLO11s phát hiện vị trí biển số
- **OCR**: PaddleOCR đọc text từ biển số
- **Validation**: Kiểm tra format biển số Việt Nam
- **Gallery**: Hiển thị các biển số đã cắt

### Tab 2: Detection Only
- Chỉ phát hiện vị trí biển số (không đọc text)

### Tab 3: Folder Watch
- Tự động xử lý ảnh trong thư mục

## Sử dụng qua Gradio Client API

```python
from gradio_client import Client, handle_file

client = Client("http://localhost:7860/")

# Full LPR Pipeline
result = client.predict(
    image=handle_file("path/to/car.jpg"),
    conf_threshold=0.25,
    scale_factor=4,
    min_ocr_conf=0.3,
    api_name="/process_full_lpr"
)

# result[0] = annotated image with bboxes + OCR text
# result[1] = gallery of cropped plates
# result[2] = detailed results text
```

## Vietnamese License Plate Types

| Type | Format | Example |
|------|--------|---------|
| Private Car | XX-YYYY.NN | 30A-1234.56 |
| Motorcycle | YY-NNNNN / YY-XXX-YY | 43-12345 / 80-NG-63 |
| Police | XX-YYYY-NN | 60-1234.56 |
| Army | YYYYYY-NN | 123456-78 |
| Commercial | XX-YYYY.NN | 51-1234.56 |

## Model Files

| File | Size | Description |
|------|------|-------------|
| `weights/best.pt` | 18.3 MB | Best model weights (PyTorch) |
| `weights/best.onnx` | 36.1 MB | ONNX format for deployment |
| `weights/last.pt` | 18.3 MB | Last checkpoint |

## Dependencies

```
ultralytics>=8.0.0
gradio>=4.0.0
opencv-python>=4.8.0
paddleocr>=2.7.0
paddlepaddle>=2.5.0
numpy>=1.24.0
torch>=2.0.0
```

## Ports

- Default: 7860
