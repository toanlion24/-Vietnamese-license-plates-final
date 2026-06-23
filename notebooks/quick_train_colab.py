# Quick Setup Script for Google Colab Training
# ==========================================

# 1. INSTALL DEPENDENCIES
# =========================
!pip install ultralytics gdown -q

# Check GPU
!nvidia-smi

# 2. UPLOAD DATASET
# ===================
# Option A: Upload ZIP file
from google.colab import files
uploaded = files.upload()

# Extract
import zipfile
for filename in uploaded.keys():
    print(f"Extracting {filename}...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('/content/dataset/')
    print("Done!")

# Option B: Download from Google Drive
# Uncomment and set FILE_ID
# FILE_ID = "YOUR_FILE_ID"
# !gdown --id {FILE_ID}
# !unzip -q "*.zip" -d /content/dataset/

# 3. CREATE DATA.YAML
# ====================
data_yaml = """
path: /content/dataset/License Plate Detection Dataset
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: license_plate
"""

with open('/content/data.yaml', 'w') as f:
    f.write(data_yaml)

# 4. VERIFY DATASET
# ==================
import os
from pathlib import Path

base = Path('/content/dataset/License Plate Detection Dataset')
for split in ['train', 'val', 'test']:
    img_dir = base / 'images' / split
    lbl_dir = base / 'labels' / split
    
    if img_dir.exists() and lbl_dir.exists():
        imgs = len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))
        lbls = len(list(lbl_dir.glob('*.txt')))
        print(f"{split.upper()}: {imgs} images, {lbls} labels")
    else:
        print(f"{split.upper()}: NOT FOUND")

# 5. TRAIN YOLOv11
# ==================
from ultralytics import YOLO

# Load model (s = small, recommended)
model = YOLO('yolov11s.pt')

# Train
results = model.train(
    data='/content/data.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    device=0,
    project='/content/runs',
    name='plate_detection',
    exist_ok=True,
    
    # Optimizer
    optimizer='AdamW',
    lr0=0.001,
    
    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    
    # Settings
    save=True,
    save_json=True,
    patience=50,
)

# 6. VALIDATE
# ============
best_model = YOLO('/content/runs/plate_detection/weights/best.pt')
val_results = best_model.val(data='/content/data.yaml', split='test')

print(f"\nTest Set Results:")
print(f"mAP@0.5: {val_results.box.map50:.4f}")
print(f"mAP@0.5:0.95: {val_results.box.map:.4f}")

# 7. DOWNLOAD MODEL
# ==================
from google.colab import files

# Download best.pt
!cp /content/runs/plate_detection/weights/best.pt /content/best_plate_detector.pt
files.download('/content/best_plate_detector.pt')

# Or export to ONNX
best_model.export(format='onnx')
files.download('/content/runs/plate_detection/weights/best.onnx')
