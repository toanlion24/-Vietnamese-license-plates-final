# Vietnamese LPR Pipeline Architecture

## System Overview

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                        VIETNAMESE LICENSE PLATE RECOGNITION                     │
│                              END-TO-END PIPELINE                               │
└────────────────────────────────────────────────────────────────────────────────┘

                                    ┌─────────────┐
                                    │   INPUT     │
                                    │  (Image/    │
                                    │   Video/    │
                                    │   Stream)   │
                                    └──────┬──────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
           ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
           │   CAMERA      │      │    IMAGE     │      │    VIDEO     │
           │   STREAM      │      │    FILE      │      │    FILE      │
           │   (RTSP)      │      │    (JPG)     │      │    (MP4)     │
           └───────────────┘      └───────────────┘      └───────────────┘
                    │                      │                      │
                    └──────────────────────┼──────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              STAGE 1: PREPROCESSING                              │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                        IMAGE ENHANCEMENT                                  │   │
│   │                                                                          │   │
│   │   • CLAHE (Contrast Limited Adaptive Histogram Equalization)             │   │
│   │   • Denoising (Non-local means)                                          │   │
│   │   • Deblurring (Blind deconvolution for motion blur)                     │   │
│   │   • Super-resolution (ESRGAN optional for low-quality inputs)           │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                           │                                     │
│                                           ▼                                     │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                           RESIZE & NORMALIZE                             │   │
│   │                                                                          │   │
│   │   Input Size: Variable (up to 4096x4096)                                 │   │
│   │   Target Size: 1920x1080 (maintain aspect ratio)                         │   │
│   │   Normalization: ImageNet stats [mean, std] or /255.0                   │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              STAGE 2: DETECTION (YOLOv11)                        │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                           YOLOv11 DETECTOR                                │   │
│   │                                                                          │   │
│   │   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐              │   │
│   │   │ Backbone│───▶│  Neck  │───▶│  Head   │───▶│   NMS   │              │   │
│   │   │ CSPDarknet│   │  PAA   │    │  Detect │    │  Filter │              │   │
│   │   └─────────┘    └─────────┘    └─────────┘    └─────────┘              │   │
│   │                                                                          │   │
│   │   Model Sizes:                                                            │   │
│   │   ├── yolov11n.pt (nano)    → Fast, lower accuracy                       │   │
│   │   ├── yolov11s.pt (small)   → Balanced ✓ RECOMMENDED                    │   │
│   │   ├── yolov11m.pt (medium)  → Higher accuracy                            │   │
│   │   └── yolov11l.pt (large)   → Highest accuracy                            │   │
│   │                                                                          │   │
│   │   Output: List of detections [x1, y1, x2, y2, conf, class_id]           │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              STAGE 3: PLATE PROCESSING                           │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   For each detected plate:                                                      │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                        CROP & PERSPECTIVE TRANSFORM                      │   │
│   │                                                                          │   │
│   │   Original Detection                                                     │   │
│   │   ┌─────────────────────────────────┐                                     │   │
│   │   │                                 │                                     │   │
│   │   │     ╔═══════════════════╗        │                                     │   │
│   │   │     ║                   ║        │  ──▶  ┌──────────────────────┐   │   │
│   │   │     ║   LICENSE PLATE   ║        │       │                      │   │   │
│   │   │     ║                   ║        │       │   LICENSE PLATE      │   │   │
│   │   │     ╚═══════════════════╝        │       │      CROPPED         │   │   │
│   │   │                                 │       │                      │   │   │
│   │   └─────────────────────────────────┘       └──────────────────────┘   │   │
│   │       (Skewed/angled)                         (Aligned/Strait)         │   │
│   │                                                                          │   │
│   │   Algorithm: Perspective transform using corner detection              │   │
│   │   Padding: 10% border added around cropped region                     │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                           │                                     │
│                                           ▼                                     │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                        PLATE TYPE CLASSIFICATION                        │   │
│   │                                                                          │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │   │
│   │   │   PRIVATE    │  │  COMMERCIAL  │  │   MOTORCYCLE │                 │   │
│   │   │   (White)    │  │   (Yellow)   │  │   (White)    │                 │   │
│   │   └──────────────┘  └──────────────┘  └──────────────┘                 │   │
│   │                                                                          │   │
│   │   ┌──────────────┐  ┌──────────────┐                                    │   │
│   │   │   POLICE     │  │    ARMY     │                                     │   │
│   │   │   (White)    │  │   (Red)     │                                     │   │
│   │   └──────────────┘  └──────────────┘                                    │   │
│   │                                                                          │   │
│   │   Purpose: Select appropriate recognition model/config                   │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           STAGE 4: RECOGNITION (PaddleOCR)                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                        PADDLEOCR PIPELINE                               │   │
│   │                                                                          │   │
│   │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐            │   │
│   │   │   Text       │───▶│   Text       │───▶│   CTC        │            │   │
│   │   │   Detection  │    │   Recognition│    │   Decode     │            │   │
│   │   │   (DB/ARSR)  │    │   (CRNN)     │    │              │            │   │
│   │   └──────────────┘    └──────────────┘    └──────────────┘            │   │
│   │          │                   │                                         │   │
│   │          ▼                   ▼                                         │   │
│   │   ┌──────────────┐    ┌──────────────┐                               │   │
│   │   │   Text       │    │   Character  │                               │   │
│   │   │   Polygons   │    │   Sequence   │                               │   │
│   │   └──────────────┘    └──────────────┘                               │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                     CHARACTER SET CONFIGURATION                         │   │
│   │                                                                          │   │
│   │   Standard: A-Z, 0-9                                                    │   │
│   │   Vietnamese Extended: AÀĂẮẰẦẨẪẬẮẰẶáàăắặấầẩẫậ (FULL ALPHABET)           │   │
│   │   Special: Hyphen (-), Period (.)                                       │   │
│   │                                                                          │   │
│   │   Dictionary Size: 165 characters (standard) / 230 (extended)          │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              STAGE 5: POST-PROCESSING                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                        TEXT NORMALIZATION                               │   │
│   │                                                                          │   │
│   │   1. Remove noise characters                                            │   │
│   │   2. Correct common OCR errors                                          │   │
│   │   3. Normalize Vietnamese diacritics                                     │   │
│   │   4. Convert to standard case                                            │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                           │                                     │
│                                           ▼                                     │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                      FORMAT VALIDATION                                   │   │
│   │                                                                          │   │
│   │   Pattern Matching with Regex:                                          │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │  Private Car:  /^\d{2}[A-Z]-\d{4}\.\d{2}$/                    │   │   │
│   │   │  Motorcycle:   /^\d{2}-\d{5}(\.\d{2})?$/                       │   │   │
│   │   │  Police:       /^\d{2}-\d{4}-\d{2}$/                           │   │   │
│   │   │  Army:         /^\d{6}-\d{2}$/                                  │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                           │                                     │
│                                           ▼                                     │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                       CONFIDENCE SCORING                                │   │
│   │                                                                          │   │
│   │   Combined Score = Detection_Conf × Recognition_Conf × Format_Score   │   │
│   │                                                                          │   │
│   │   Acceptance Threshold: 0.7 (configurable)                             │   │
│   │   High Confidence: > 0.9                                                │   │
│   │   Medium Confidence: 0.7 - 0.9                                          │   │
│   │   Low Confidence: < 0.7 (flag for review)                               │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                    OUTPUT                                        │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                           RESULT STRUCTURE                               │   │
│   │                                                                          │   │
│   │   {                                                                       │   │
│   │     "plate": "30A-1234.56",           // Recognized text               │   │
│   │     "confidence": 0.95,              // Combined confidence           │   │
│   │     "plate_type": "private_car",       // Classification result        │   │
│   │     "bbox": [x1, y1, x2, y2],         // Detection bounding box        │   │
│   │     "processing_time_ms": 45,        // Total inference time          │   │
│   │     "timestamp": "2024-01-15T10:30:00Z"  // For video streams          │   │
│   │   }                                                                       │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                          │
└─────────────────────────────────────────────────────────────────────────────────┘

IMAGE INPUT
    │
    ▼
[Preprocessing] ──────────────────────────────────────────▶ [Original for display]
    │                                                             │
    ▼                                                             │
[YOLOv11 Detection]                                             │
    │                                                             │
    ├──▶ [No plates detected] ──▶ [Return empty result]         │
    │                                                             │
    ▼                                                             │
[Crop Plates] ──────────────────────────────────────────▶ [Cropped plates for debug]
    │                                                             │
    ▼                                                             │
[Plate Alignment]                                                │
    │                                                             │
    ▼                                                             │
[Type Classification]                                           │
    │                                                             │
    ├──▶ [Type: Private] ──▶ [Config: Standard dictionary]      │
    ├──▶ [Type: Motorcycle] ──▶ [Config: Motorcycle dictionary] │
    └──▶ [Type: Police/Army] ──▶ [Config: Numeric-heavy dict]   │
    │                                                             │
    ▼                                                             │
[PaddleOCR Recognition]                                          │
    │                                                             │
    ├──▶ [Multiple candidates] ──▶ [Select best by confidence]  │
    └──▶ [Single candidate] ──▶ [Use directly]                   │
    │                                                             │
    ▼                                                             │
[Post-processing]                                                │
    │                                                             │
    ├──▶ [Format valid] ──▶ [Accept result]                       │
    └──▶ [Format invalid] ──▶ [Flag for review / Use raw OCR]    │
    │                                                             │
    ▼                                                             │
[Output Result]                                                  │
```

---

## Performance Targets

| Metric | Target | Measured At |
|--------|--------|-------------|
| Detection mAP@0.5 | > 0.95 | GPU (RTX 3080+) |
| Detection mAP@0.5:0.95 | > 0.85 | GPU (RTX 3080+) |
| Recognition Accuracy (char) | > 95% | Full dataset |
| End-to-End Accuracy | > 85% | Full dataset |
| Inference Time (GPU) | < 50ms | Per image |
| Inference Time (CPU) | < 200ms | Per image |
| Video FPS (GPU) | > 20 FPS | 1080p input |
| Memory Usage (GPU) | < 2GB | Peak during inference |

---

## Error Handling

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ERROR HANDLING STRATEGY                            │
└─────────────────────────────────────────────────────────────────────────────────┘

Detection Failures:
├── No plates detected
│   └── Action: Return empty list, log confidence scores
├── Multiple plates (> 10)
│   └── Action: Return top 10 by confidence, warn about complexity
└── Very small detection (< 20px)
    └── Action: Skip, warn about resolution

Recognition Failures:
├── OCR returns empty
│   └── Action: Skip plate, mark as "unreadable"
├── OCR confidence < 0.3
│   └── Action: Mark as low confidence, include in review queue
└── Invalid format after normalization
    └── Action: Keep raw OCR result, flag for validation

System Errors:
├── GPU out of memory
│   └── Action: Fallback to CPU mode, reduce batch size
├── Model file missing
│   └── Action: Raise clear error with download instructions
└── Input file corrupt
    └── Action: Return error result, log file path
```

---

## Configuration Schema

```yaml
# configs/pipeline.yaml
pipeline:
  name: "VietnameseLPR"
  version: "1.0.0"

detection:
  model_path: "models/yolov11/best.pt"
  confidence_threshold: 0.25
  nms_threshold: 0.45
  input_size: 640
  max_detections: 10

recognition:
  config_path: "configs/recognition.yaml"
  model_path: "models/paddleocr/best"
  confidence_threshold: 0.5
  dictionary: "vietnamese_full"  # or "motorcycle", "numeric"
  
preprocessing:
  enhance: true
  clahe_clip_limit: 2.0
  clahe_grid_size: [8, 8]
  denoise: false
  resize_max: [1920, 1080]

postprocessing:
  normalize_text: true
  validate_format: true
  min_confidence: 0.7
  spell_check: true

output:
  include_debug_images: false
  include_processing_time: true
  log_level: "INFO"
```
