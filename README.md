# Vietnamese LPR - YOLOv11 + PaddleOCR

Production-ready License Plate Recognition system for Vietnamese vehicles.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT (Image/Video)                                │
│                           ↓                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                   STAGE 1: PREPROCESSING                               │  │
│  │   CLAHE Enhancement → Resize → Normalize                              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                           ↓                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                   STAGE 2: YOLOv11 DETECTION                          │  │
│  │   License Plate Detection → Bounding Box + Confidence                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                           ↓                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                   STAGE 3: PLATE PROCESSING                           │  │
│  │   Crop → Perspective Transform → Type Classification                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                           ↓                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                   STAGE 4: PaddleOCR RECOGNITION                       │  │
│  │   Text Detection → Character Recognition → CTC Decode                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                           ↓                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                   STAGE 5: POST-PROCESSING                            │  │
│  │   Normalize → Format Validation → Confidence Scoring                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                           ↓                                                  │
│                           OUTPUT                                              │
│              { plate: "30A-1234.56", confidence: 0.95 }                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download pretrained models
python scripts/download_models.py

# Run inference on image
python -m src.pipeline.inference --image test.jpg

# Run inference on video
python -m src.pipeline.inference --video traffic.mp4

# Run webcam real-time
python -m src.pipeline.inference --camera 0
```

## Project Structure

```
vietnamese-lpr/
├── src/
│   ├── detection/          # YOLOv11 detection module
│   ├── recognition/       # PaddleOCR recognition module
│   ├── pipeline/          # End-to-end pipeline
│   └── utils/             # Common utilities
├── configs/               # Model configurations
├── models/               # Trained weights
├── data/                 # Dataset storage
├── scripts/              # Utility scripts
├── notebooks/            # Jupyter notebooks
├── tests/                # Unit & integration tests
└── docs/                 # Documentation
```

## Performance

| Metric | Target | Status |
|--------|--------|--------|
| Detection mAP@0.5 | > 0.95 | 🚀 |
| Recognition Accuracy | > 90% | 🚀 |
| Inference Speed (GPU) | < 50ms | 🚀 |
| End-to-End Accuracy | > 85% | 🚀 |

## License Plate Formats

- **Private Car**: `30A-1234.56`
- **Motorcycle**: `43-12345`
- **Police**: `60-1234-56`
- **Army**: `123456-78`

---
Built with YOLOv11 + PaddleOCR for Vietnamese License Plate Recognition
