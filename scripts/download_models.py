"""
Download pretrained models for Vietnamese LPR
"""

import os
import sys
from pathlib import Path
import urllib.request
import zipfile
import shutil

try:
    from ultralytics import YOLO
    from paddleocr import PaddleOCR
    print("[OK] Ultralytics and PaddleOCR packages available")
except ImportError as e:
    print(f"[WARN] Missing packages: {e}")
    print("Install with: pip install ultralytics paddlepaddle-gpu paddleocr")


def download_yolov11(model_name: str = "yolov11s"):
    """Download YOLOv11 pretrained model"""
    print(f"\n[1/2] Downloading YOLOv11 {model_name}...")
    
    models_dir = Path("models/yolov11")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / f"{model_name}.pt"
    
    if model_path.exists():
        print(f"  Model already exists: {model_path}")
        return str(model_path)
    
    try:
        print(f"  Downloading {model_name} from Ultralytics...")
        model = YOLO(f"{model_name}.pt")
        model.save(str(model_path))
        print(f"  Saved to: {model_path}")
        return str(model_path)
    except Exception as e:
        print(f"  [ERROR] Failed to download: {e}")
        
        fallback_url = f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{model_name}.pt"
        print(f"  Trying fallback: {fallback_url}")
        
        try:
            urllib.request.urlretrieve(fallback_url, str(model_path))
            print(f"  Downloaded to: {model_path}")
            return str(model_path)
        except Exception as e2:
            print(f"  [ERROR] Fallback failed: {e2}")
            return None


def download_paddleocr_models():
    """Download PaddleOCR models"""
    print(f"\n[2/2] Checking PaddleOCR models...")
    
    models_dir = Path("models/paddleocr")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False)
        print("  PaddleOCR initialized successfully")
        return True
    except Exception as e:
        print(f"  [WARN] PaddleOCR initialization issue: {e}")
        print("  This may be due to missing model files.")
        print("  Models will be downloaded on first use.")
        return True


def create_sample_dict():
    """Create Vietnamese character dictionary"""
    print("\n[+] Creating Vietnamese character dictionary...")
    
    dict_path = Path("configs/vietnamese_dict.txt")
    dict_path.parent.mkdir(parents=True, exist_ok=True)
    
    vietnamese_chars = """AÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ
BCDĐEÉÈẺẼẸÊẾỀỂỄỆ
FGHIÍÌỈĨỊJKLM
NOÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ
PQRSTUÚÙỦŨỤƯỨỪỬỮỰ
VXY
aáàảãạăắằẳẵặâấầẩẫậ
bcdđeéèẻẽẹêếềểễệ
fghiíìỉĩịjklm
noóòỏõọôốồổỗộơớờởỡợ
pqrstuúùủũụưứừửữự
vxy
0123456789
- ."""
    
    with open(dict_path, 'w', encoding='utf-8') as f:
        f.write(vietnamese_chars)
    
    print(f"  Saved to: {dict_path}")


def main():
    print("=" * 50)
    print("Vietnamese LPR - Model Downloader")
    print("=" * 50)
    
    print("\nThis script will download:")
    print("  - YOLOv11 detection model")
    print("  - PaddleOCR recognition models")
    print("  - Vietnamese character dictionary")
    
    yolo_path = download_yolov11("yolov11s")
    paddleocr_ok = download_paddleocr_models()
    create_sample_dict()
    
    print("\n" + "=" * 50)
    print("Download complete!")
    print("=" * 50)
    
    print(f"\nModels location:")
    print(f"  YOLOv11: {yolo_path or 'Not downloaded'}")
    print(f"  PaddleOCR: Auto-downloaded on first use")
    print(f"  Dictionary: configs/vietnamese_dict.txt")
    
    print("\nNext steps:")
    print("  1. Add training data to data/")
    print("  2. Train models with: python -m src.detection.train")
    print("  3. Run inference: python -m src.pipeline.inference --image test.jpg")


if __name__ == "__main__":
    main()
