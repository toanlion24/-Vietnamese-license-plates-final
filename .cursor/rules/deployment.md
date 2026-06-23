# Vietnamese LPR Deployment Guide

## Deployment Options

### 1. Python CLI

**Best for:** Batch processing, scripts, automation

```bash
# Single image
python -m src.pipeline.inference --image test.jpg

# Video
python -m src.pipeline.inference --video traffic.mp4

# With output
python -m src.pipeline.inference \
    --image test.jpg \
    --output result.jpg
```

### 2. Gradio Web App

**Best for:** Demo, testing, quick validation

```bash
python -m src.demo.gradio_demo
```

Then open http://localhost:7860

### 3. FastAPI Server

**Best for:** Production API, integration

```python
# app.py
from fastapi import FastAPI, UploadFile
from src.pipeline import VietnameseLPRPipeline

app = FastAPI()
pipeline = VietnameseLPRPipeline()

@app.post("/recognize")
async def recognize(file: UploadFile):
    contents = await file.read()
    results = pipeline.process_image(contents)
    return {"plates": [r.to_dict() for r in results]}
```

Run with:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 4. ONNX Export

**Best for:** Cross-platform, mobile, edge devices

```python
from ultralytics import YOLO

# Export YOLO
model = YOLO("models/yolov11/best.pt")
model.export(format="onnx", imgsz=640)
```

## Model Optimization

### TensorRT (NVIDIA GPUs)

```python
# Export with TensorRT
model = YOLO("models/yolov11/best.pt")
model.export(format="engine", half=True)
```

### ONNX Optimization

```bash
# Install onnxruntime
pip install onnx onnxruntime-gpu onnxoptimizer

# Optimize
python -c "
import onnx
from onnxoptimizer import optimize
model = onnx.load('yolov11.onnx')
optimized = optimize(model)
onnx.save(optimized, 'yolov11_optimized.onnx')
"
```

## Performance Targets by Platform

| Platform | Latency | Throughput |
|----------|---------|------------|
| RTX 3080+ | < 30ms | > 30 FPS |
| RTX 2060 | < 50ms | > 20 FPS |
| GTX 1080 | < 80ms | > 12 FPS |
| CPU (i7) | < 300ms | > 3 FPS |
| Jetson Nano | < 200ms | > 5 FPS |

## Docker Deployment

### Dockerfile

```dockerfile
FROM pytorch/pytorch:2.0-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY models/ ./models
COPY src/ ./src
COPY configs/ ./configs

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "src.pipeline.inference"]
```

### Build and Run

```bash
# Build
docker build -t vietnamese-lpr .

# Run
docker run --gpus all -v /data:/data vietnamese-lpr \
    python -m src.pipeline.inference --image /data/test.jpg
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LPR_DEVICE` | cuda | cuda or cpu |
| `LPR_CONFIDENCE` | 0.7 | Minimum confidence |
| `LPR_MODEL_PATH` | models/ | Model directory |
| `LPR_LOG_LEVEL` | INFO | Logging level |

## API Reference

### REST API

```
POST /api/v1/recognize
Content-Type: multipart/form-data

Request:
  image: file (jpg, png)

Response:
{
  "plates": [
    {
      "plate": "30A-1234.56",
      "confidence": 0.95,
      "bbox": [x1, y1, x2, y2],
      "type": "private_car"
    }
  ],
  "processing_time_ms": 42.5
}
```

### Python API

```python
from src.pipeline import VietnameseLPRPipeline

pipeline = VietnameseLPRPipeline(
    detector_weights="models/yolov11/best.pt",
    device="cuda"
)

# Single image
results = pipeline.process_image("test.jpg")

# Batch
results = pipeline.process_batch(["img1.jpg", "img2.jpg"])

# Video
for frame_result in pipeline.process_video("video.mp4"):
    print(frame_result.timestamp, frame_result.results)
```

## Monitoring

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter('lpr_requests_total', 'Total LPR requests')
PROCESSING_TIME = Histogram('lpr_processing_seconds', 'Processing time')

@app.middleware("http")
async def monitor(request, call_next):
    REQUEST_COUNT.inc()
    with PROCESSING_TIME.time():
        response = await call_next(request)
    return response
```

### Health Check

```bash
curl http://localhost:8000/health
# {"status": "healthy", "model": "loaded"}
```

## Troubleshooting Deployment

| Issue | Solution |
|-------|----------|
| Model loading slow | Pre-load model at startup |
| Memory leak | Use process pool, limit batch size |
| GPU not used | Check CUDA availability |
| Port conflict | Change port or use Docker |
| Slow cold start | Pre-warm model on startup |

## Security

### Input Validation

```python
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_upload(file: UploadFile):
    if Path(file.filename).suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Invalid file type")
    
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large")
    
    return contents
```

### Rate Limiting

```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/recognize")
@limiter.limit("100/minute")
async def recognize(request: Request):
    ...
```
