# Vietnamese LPR Data Processing Rules

## Overview

These rules govern data collection, preprocessing, and dataset preparation for the Vietnamese LPR project.

## Dataset Structure

```
data/
├── raw/                    # Original collected data
├── processed/              # Preprocessed data
│   ├── train/
│   ├── val/
│   └── test/
└── datasets/
    ├── yolo_detection/     # For detection training
    │   ├── train/images/
    │   ├── train/labels/
    │   ├── val/images/
    │   └── val/labels/
    └── ocr_recognition/    # For OCR training
        ├── train/
        └── val/
```

## Image Requirements

### Resolution
- Minimum: 640x480
- Recommended: 1920x1080
- Maximum: 4096x4096

### Quality
- Format: JPG, PNG, BMP
- Color: RGB (convert BGR in OpenCV)
- No watermarks or overlays

### Diversity
- Various lighting conditions (day, night, indoor, outdoor)
- Multiple angles (front, side, angled)
- Different plate types balanced
- Various camera resolutions

## YOLO Annotation Format

```
<class_id> <x_center> <y_center> <width> <height>
```

- All values normalized to [0, 1]
- Single class: `0` for license_plate
- One annotation file per image

## Annotation Tools

Use LabelImg for detection annotations:
```bash
pip install labelImg
labelImg --yolo  # YOLO format output
```

## Data Augmentation

### During Training
- Rotation: ±5°
- Brightness: ±10%
- Noise: Gaussian, σ=0.01
- No blur (will reduce OCR accuracy)

### NOT Recommended
- Heavy occlusion
- Extreme angles (>30°)
- Very low resolution (<200px width)

## Validation Checklist

Before training:
- [ ] Minimum 1000 training images
- [ ] Minimum 200 validation images
- [ ] Annotations verified (sample 5%)
- [ ] Class distribution balanced
- [ ] No corrupted images

## Preprocessing Pipeline

```python
def preprocess_image(image: np.ndarray) -> np.ndarray:
    # 1. Resize to max 1920x1080
    # 2. Apply CLAHE (clip_limit=2.0, grid=(8,8))
    # 3. Convert BGR to RGB
    # 4. Normalize with ImageNet stats
    return processed
```

## Character Dictionary

Vietnamese plates contain:
```
AÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬBCCDĐEÉÈẺẼẸÊẾỀỂỄỆ
FGHIÍÌỈĨỊJKLMNOÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢP
QRSTUÚÙỦŨỤƯỨỪỬỮỰVXY0123456789- .
```

Total: ~165 characters
