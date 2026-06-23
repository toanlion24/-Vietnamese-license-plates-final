"""
02 - Model Training
Vietnamese LPR - Training Pipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    print("=" * 60)
    print("Vietnamese LPR - Model Training")
    print("=" * 60)
    
    print("\nThis notebook demonstrates:")
    print("  1. YOLOv11 detection model training")
    print("  2. PaddleOCR recognition model training")
    print("  3. End-to-end pipeline evaluation")
    
    print("\n" + "-" * 40)
    print("Step 1: Prepare Training Data")
    print("-" * 40)
    
    print("""
    Organize your data as YOLO format:
    
    data/
    └── datasets/
        └── yolo_detection/
            ├── train/
            │   ├── images/
            │   │   ├── img001.jpg
            │   │   └── img002.jpg
            │   └── labels/
            │       ├── img001.txt
            │       └── img002.txt
            └── val/
                ├── images/
                └── labels/
    
    Label format (YOLO):
      class_id x_center y_center width height
      (normalized to 0-1)
    """)
    
    print("-" * 40)
    print("Step 2: Train Detection Model")
    print("-" * 40)
    
    train_cmd = """
    python -m src.detection.train \\
        --data data/datasets/yolo_detection/data.yaml \\
        --model yolov11s \\
        --epochs 100 \\
        --batch 16 \\
        --device cuda
    """
    print(train_cmd)
    
    print("-" * 40)
    print("Step 3: Train Recognition Model")
    print("-" * 40)
    
    print("""
    After detection training, extract cropped plates:
    
    python scripts/prepare_ocr_data.py \\
        --detection-model models/yolov11/best.pt \\
        --output data/datasets/ocr_recognition/
    
    Then train OCR:
    
    python -m src.recognition.train \\
        --config configs/recognition.yaml
    """)
    
    print("-" * 40)
    print("Step 4: Evaluate Pipeline")
    print("-" * 40)
    
    print("""
    python -m scripts.evaluate_pipeline \\
        --detector models/yolov11/best.pt \\
        --recognizer models/paddleocr/best \\
        --test-data data/test/
    """)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
