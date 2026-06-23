# Vietnamese LPR - Recognition Module Skill

---
name: vietnamese-lpr-recognition
description: >
  PaddleOCR character recognition module. Use when modifying OCR,
  training, or debugging recognition issues.
---

## Overview

The recognition module uses PaddleOCR with CRNN architecture to recognize characters on detected license plates. Supports full Vietnamese alphabet.

## Module Structure

```
src/recognition/
├── recognizer.py    # PlateRecognizer class
├── train.py        # Training script
└── __init__.py
```

## PlateRecognizer Class

```python
from src.recognition import PlateRecognizer

recognizer = PlateRecognizer(
    config_path="configs/recognition.yaml",
    model_path="models/paddleocr/best",
    dictionary_path="configs/vietnamese_dict.txt",
    use_gpu=True,
    lang="vi"
)
```

## Methods

### recognize()

```python
results = recognizer.recognize(
    image,                  # Cropped plate image
    return_confidence=True
)
```

Returns: `List[RecognitionResult]`

### recognize_plate()

Full plate recognition with post-processing:
```python
text, confidence = recognizer.recognize_plate(
    image,
    plate_type="car"  # Optional hint
)
```

## RecognitionResult Dataclass

```python
@dataclass
class RecognitionResult:
    text: str                    # "30A-1234.56"
    confidence: float             # 0.95
    bbox: List[List[float]]      # [[x1,y1], [x2,y2], ...]
```

## Preprocessing

Critical for OCR accuracy:

```python
from src.recognition import preprocess_for_recognition

# Enhance and resize
processed = preprocess_for_recognition(
    image,
    target_height=48,
    target_width=320,
    enhance=True  # CLAHE
)
```

### Preprocessing Steps

1. **CLAHE Enhancement** - Improves contrast
2. **Resize** - Standard size for model
3. **Padding** - Center on canvas

## Vietnamese Character Set

```
AÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬBCCDĐEÉÈẺẼẸÊẾỀỂỄỆ
FGHIÍÌỈĨỊJKLMNOÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢP
QRSTUÚÙỦŨỤƯỨỪỬỮỰVXY
0123456789
- .
```

Dictionary file: `configs/vietnamese_dict.txt`

## Plate Type Dictionaries

| Type | Characters | Example |
|------|------------|---------|
| Car | Full alphabet | 30A-1234.56 |
| Motorcycle | Subset + digits | 43-12345 |
| Police | Numeric heavy | 60-1234-56 |
| Army | All digits | 123456-78 |

## Post-processing

```python
from src.recognition import normalize_vietnamese_plate, validate_plate_format

# Normalize
normalized = normalize_vietnamese_plate("30A-1234.56")
# "30A-1234.56"

# Validate
is_valid = validate_plate_format(
    "30A-1234.56",
    plate_type="private_car"
)
# True
```

### Normalization Rules

```python
replacements = {
    'O': '0',   # Letter O to number 0
    'D': '0',   # Letter D to 0
    'I': '1',   # Letter I to 1
    'l': '1',   # lowercase l to 1
    'S': '5',   # Letter S to 5
    'B': '8',   # Letter B to 8
}
```

## Training

### Prepare Data

```bash
# Extract cropped plates using trained detector
python scripts/prepare_ocr_data.py \
    --detection-model models/yolov11/best.pt \
    --output data/datasets/ocr_recognition/
```

### Label Format

Create `labels.txt`:
```
image001.jpg	30A-1234.56
image002.jpg	51B-5678.90
```

### Train

```bash
python -m src.recognition.train \
    --config configs/recognition.yaml \
    --epochs 200
```

### Configuration

```yaml
# configs/recognition.yaml
Global:
  algorithm: CRNN
  max_text_length: 25
  character_dict_path: ../configs/vietnamese_dict.txt

Architecture:
  Backbone:
    name: ResNet34_vd
  Neck:
    name: RNNEncoder
  Head:
    name: CTCHead
```

## Evaluation

### Metrics

| Metric | Target | Acceptable |
|--------|--------|------------|
| Character Accuracy | > 95% | > 92% |
| Word Accuracy | > 90% | > 85% |
| Mean Confidence | > 0.90 | > 0.85 |

### Manual Test

```python
from src.recognition import PlateRecognizer, preprocess_for_recognition
import cv2

recognizer = PlateRecognizer()
image = cv2.imread("cropped_plate.jpg")
processed = preprocess_for_recognition(image)
results = recognizer.recognize(processed)

print(f"Text: {results[0].text}")
print(f"Confidence: {results[0].confidence}")
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Empty result | Image quality | Preprocess, enhance |
| Wrong characters | Model not fine-tuned | Train with Vietnamese data |
| Low confidence | Unclear image | Improve preprocessing |
| Missing diacritics | Dictionary incomplete | Update vietnamese_dict.txt |

## Common OCR Errors

```python
# Known confusions
known_errors = {
    '0': ['O', 'D'],
    '1': ['I', 'l', '|'],
    '5': ['S', 's'],
    '8': ['B', 'b'],
    'A': ['4'],  # In some fonts
}
```

Apply correction after recognition.

## Red Flags

- Using English-only OCR without fine-tuning
- Ignoring preprocessing
- No confidence threshold
- Training without Vietnamese data
- Single character output only
