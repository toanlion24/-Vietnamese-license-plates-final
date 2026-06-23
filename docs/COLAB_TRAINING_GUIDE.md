# Hướng Dẫn Train YOLOv11 Trên Google Colab

## 📋 Mục Lục

1. [Tổng Quan](#tổng-quan)
2. [Cách 1: Upload Dataset ZIP](#cách-1-upload-dataset-zip-lên-colab)
3. [Cách 2: Google Drive (Nhanh hơn)](#cách-2-upload-lên-google-drive)
4. [Cách 3: Kaggle Dataset](#cách-3-kaggle-dataset)
5. [Chạy Training](#chạy-training)
6. [Theo Dõi Training](#theo-dõi-training)
7. [Tải Model Về](#tải-model-về)
8. [Xử Lý Lỗi Thường Gặp](#xử-lý-lỗi-thường-gặp)

---

## 🎯 Tổng Quan

### Tại Sao Nên Dùng Google Colab?

| Tiêu chí | Local PC | Google Colab Free |
|-----------|----------|------------------|
| GPU | Phụ thuộc máy bạn | **NVIDIA T4 (Free)** |
| VRAM | 4-8GB thường | **15GB** |
| Cài đặt | Phải tự làm | **Đã có sẵn** |
| Chi phí | Điện + Hardware | **Miễn phí** |
| Giới hạn | Không | ~12h/phiên |

### GPU Available trên Colab

```
Colab Free:     NVIDIA T4 (16GB VRAM)  ← Recommended
Colab Pro:      NVIDIA V100 (16GB) / A100 (40GB)
```

---

## 📦 Cách 1: Upload Dataset ZIP lên Colab

### Bước 1: Nén Dataset trên Windows

```powershell
# Mở PowerShell hoặc Terminal
cd D:\ComputerVisionLasted

# Nén thư mục dataset thành ZIP
Compress-Archive -Path "License Plate Detection Dataset" -DestinationPath "D:\dataset.zip" -Force

# Kiểm tra dung lượng
Get-Item "D:\dataset.zip" | Select-Object Name, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB,2)}}
```

**Output mẫu:**
```
Name      Size(MB)
----      ---------
dataset.zip    245.78
```

### Bước 2: Upload lên Google Colab

1. Mở Google Colab: https://colab.research.google.com/

2. Tạo Notebook mới hoặc mở file `yolov11_training_colab.ipynb` trong project

3. Upload file ZIP:
```python
from google.colab import files
uploaded = files.upload()  # Click "Choose Files" và chọn dataset.zip
```

4. Đợi upload xong (tùy tốc độ mạng, ~245MB mất ~2-5 phút)

### Bước 3: Giải nén

```python
import zipfile
import os

# Tìm file ZIP đã upload
for filename in uploaded.keys():
    print(f"Extracting {filename}...")
    
    # Giải nén
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('/content/dataset/')
    
    print("Done!")

# Kiểm tra
print("\nDataset structure:")
!ls -la /content/dataset/
```

### Bước 4: Tạo data.yaml

```python
data_yaml = """
# Vietnamese License Plate Detection Dataset
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

print("data.yaml created!")
```

---

## ☁️ Cách 2: Upload lên Google Drive

**Cách này nhanh hơn và không bị giới hạn upload size**

### Bước 1: Upload lên Google Drive

1. Mở https://drive.google.com/
2. Upload folder `License Plate Detection Dataset` lên Drive
3. Click chuột phải → "Share" → "Anyone with the link"

### Bước 2: Lấy File ID

URL sau khi share sẽ có dạng:
```
https://drive.google.com/file/d/1A2B3C4D5E6F7G8H9I0J/view
                                      ↑
                                Đây là FILE ID
```

Copy FILE ID đó (`1A2B3C4D5E6F7G8H9I0J`)

### Bước 3: Download trong Colab

```python
# Cài đặt gdown
!pip install gdown -q

# Download từ Google Drive
FILE_ID = "YOUR_FILE_ID_HERE"  # Thay bằng FILE ID của bạn
!gdown --id {FILE_ID}

# Giải nén
!unzip -q "License Plate Detection Dataset.zip" -d /content/dataset/

# Kiểm tra
!ls -la /content/dataset/
```

---

## 📊 Cách 3: Kaggle Dataset

Nếu dataset nằm trên Kaggle:

```python
# Cài đặt Kaggle
!pip install kaggle -q

# Upload Kaggle API key (từ https://www.kaggle.com/account)
from google.colab import files
files.upload()  # Upload kaggle.json

# Setup
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download dataset
!kaggle datasets download -d USERNAME/DATASET-NAME -p /content/dataset/

# Giải nén
!unzip -q /content/dataset/*.zip -d /content/dataset/
```

---

## 🚀 Chạy Training

### Cài đặt Dependencies

```python
# Install ultralytics (YOLOv11)
!pip install ultralytics -q

# Kiểm tra GPU
!nvidia-smi
```

### Training với YOLOv11s (Recommended)

```python
from ultralytics import YOLO

# Load pretrained model
# n = nano, s = small, m = medium, l = large, x = xlarge
model = YOLO('yolov11s.pt')  # Recommended: balanced speed/accuracy

# Training
results = model.train(
    data='/content/data.yaml',
    epochs=100,              # Số epoch (100 là standard)
    batch=16,                # Batch size (điều chỉnh theo GPU)
    imgsz=640,              # Image size
    device=0,               # GPU 0
    
    # Output
    project='/content/runs',
    name='plate_detection',
    exist_ok=True,
    
    # Optimizer settings
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    
    # Augmentation (tăng cường dữ liệu)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    
    # Early stopping
    patience=50,
    
    # Save settings
    save=True,
    save_json=True,
    verbose=True,
)
```

### Các Model Size khác

```python
# YOLOv11 Nano (nhanh nhất, accuracy thấp hơn)
model = YOLO('yolov11n.pt')

# YOLOv11 Small (recommended)
model = YOLO('yolov11s.pt')

# YOLOv11 Medium (chậm hơn, accuracy cao hơn)
model = YOLO('yolov11m.pt')

# YOLOv11 Large (chậm, accuracy cao nhất)
model = YOLO('yolov11l.pt')
```

### Điều chỉnh Batch Size theo GPU

| GPU | VRAM | Batch Size |
|-----|------|------------|
| T4 (Colab Free) | 15GB | 16-32 |
| V100 | 16GB | 16-32 |
| A100 | 40GB | 32-64 |
| RTX 3060 | 12GB | 8-16 |
| RTX 3080 | 10GB | 8-16 |

---

## 📈 Theo Dõi Training

### Trong Colab Output

Khi training chạy, bạn sẽ thấy:
```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/100     5.3G      1.245      0.893      1.102        156        640: 100%|██████████| 414/414 [00:32<00:00, 12.84it/s]
               Class     Images  Instances      Box(P)      Box(R)      Box(mAP50)  Box(mAP50-95): 100%|██████████| 52/52 [00:05<00:00,  9.47it/s]
                 all       814       1042      0.823      0.798      0.856      0.612
```

### Metrics quan trọng

| Metric | Ý nghĩa | Target |
|--------|----------|--------|
| `box_loss` | Detection box loss | Càng thấp越好 |
| `cls_loss` | Classification loss | Càng thấp越好 |
| `Box(mAP50)` | **Độ chính xác chính** | > 0.95 |
| `Box(mAP50-95)` | mAP trung bình | > 0.75 |

### Training Logs

Sau khi train xong:
```
/content/runs/plate_detection/
├── weights/
│   ├── best.pt      ← Model tốt nhất
│   └── last.pt      ← Model cuối cùng
├── results.csv      ← Metrics log
├── results.png      ← Loss curves
└── args.yaml        ← Training config
```

---

## 💾 Tải Model Về

### Tải best.pt (Recommended)

```python
from google.colab import files

# Copy model
!cp /content/runs/plate_detection/weights/best.pt /content/best_plate_detector.pt

# Download
files.download('/content/best_plate_detector.pt')
```

### Export sang ONNX

```python
# Load best model
best_model = YOLO('/content/runs/plate_detection/weights/best.pt')

# Export to ONNX
best_model.export(format='onnx')

# Download
files.download('/content/runs/plate_detection/weights/best.onnx')
```

### Copy vào Project

1. Tải model về local
2. Copy vào folder `models/yolov11/` trong project

```
D:\ComputerVisionLasted\models\yolov11\
├── best.pt       ← Copy vào đây
└── best.onnx     ← (optional)
```

---

## ❌ Xử Lý Lỗi Thường Gặp

### Lỗi 1: "CUDA out of memory"

```python
# Giảm batch size
results = model.train(
    batch=8,    # Thay vì 16
    ...
)
```

### Lỗi 2: "No module named 'ultralytics'"

```python
!pip install ultralytics -q
```

### Lỗi 3: "Session crashed"

Thường do GPU bị disconnect. Giải pháp:
- Lưu checkpoint thường xuyên
- Dùng `patience` để early stopping
- Training với ít epochs hơn

### Lỗi 4: Dataset not found

```python
# Kiểm tra đường dẫn
import os
print(os.path.exists('/content/dataset/License Plate Detection Dataset/images/train'))

# In ra cấu trúc
!find /content/dataset -type d
```

### Lỗi 5: Runtime disconnected

Để tránh disconnect:
1. Nhấn `Ctrl+Shift+I` → Console
2. Paste:
```javascript
function KeepClicking(){
   document.querySelector("colab-connect-button").click()
}
setInterval(KeepClicking, 60000)
```

---

## 🎓 Quick Reference

### Commands để copy nhanh

```python
# ====================
# SETUP
# ====================
!pip install ultralytics -q
!nvidia-smi

# ====================
# TRAINING
# ====================
from ultralytics import YOLO
model = YOLO('yolov11s.pt')
results = model.train(
    data='/content/data.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    device=0,
    patience=50,
)

# ====================
# VALIDATE
# ====================
model = YOLO('/content/runs/plate_detection/weights/best.pt')
model.val(data='/content/data.yaml', split='test')

# ====================
# DOWNLOAD
# ====================
from google.colab import files
files.download('/content/runs/plate_detection/weights/best.pt')
```

---

## ✅ Checklist

- [ ] Dataset đã nén (.zip)
- [ ] Upload lên Colab
- [ ] Giải nén thành công
- [ ] Tạo data.yaml đúng
- [ ] Chạy training
- [ ] Model đạt mAP@0.5 > 0.90
- [ ] Tải best.pt về
- [ ] Copy vào project

---

## 📞 Hỗ Trợ

Nếu gặp lỗi, kiểm tra:
1. Console output trong Colab
2. File `results.csv` trong output folder
3. Thử restart runtime và chạy lại

---

*Document created: June 2026*
