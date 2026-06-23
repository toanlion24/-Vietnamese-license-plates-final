# Vietnamese LPR - Gradio OCR Demo

Gradio demo cho nhận diện biển số xe Việt Nam với PaddleOCR.

## Chạy demo

```bash
# 1. Đảm bảo đã cài dependencies
pip install paddlepaddle==2.6.2 "paddleocr>=2.7,<3.0" "albumentations<1.5" "gradio<6"

# 2. Chạy demo
python src/modules/gradio_lpr_ocr.py
```

Server sẽ chạy tại: **http://localhost:7870**

## Tính năng

- **Tải ảnh lên chủ động**: Upload ảnh từ máy tính hoặc paste từ clipboard
- **Demo images**: 2 ảnh mẫu để test nhanh
- **Tùy chỉnh OCR**:
  - Scale Factor (1-8): Phóng to ảnh trước khi OCR
  - Min Confidence (0.3-0.95): Ngưỡng confidence tối thiểu
- **Hiển thị kết quả**:
  - Ảnh đã xử lý (annotated)
  - Gallery các biển số đã cắt
  - Chi tiết text OCR (gốc, chuẩn hóa, loại biển số, tỉnh/TP)

## Sử dụng qua Gradio Client API

```python
from gradio_client import Client, handle_file

client = Client("http://localhost:7870/")

# Upload và xử lý ảnh
result = client.predict(
    image=handle_file("path/to/plate.jpg"),
    scale_factor=4,
    min_confidence=0.5,
    api_name="/process_image"
)

# result[0] = annotated image path
# result[1] = gallery of plates
# result[2] = detailed text results
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `/process_image` | Xử lý ảnh OCR với PaddleOCR |
| `/load_demo1` | Load ảnh demo (full image) |
| `/load_demo2` | Load ảnh demo (cropped plate) |

## Kiến trúc

```
User Upload → Image Input → OCR Processor → Results
                                    ↓
                            [Enhancement]
                            [Upscaling]
                            [PaddleOCR]
                            [Validation]
                                    ↓
                            [Annotated Image]
                            [Plate Gallery]
                            [Results Text]
```

## Demo Test Results

| Image | Detected Text | Confidence | Type |
|-------|--------------|------------|------|
| plate_01_crop.jpg | 80-NG-63 | 90.8% | motorcycle |

## Ports

- Default: 7870 (configurable in `gradio_lpr_ocr.py`)