"""
Training script for YOLOv11 plate detection
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ultralytics import YOLO
import torch


def train_yolo(
    data_config: str = "data/datasets/yolo_detection",
    model_size: str = "yolov11s",
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = 640,
    device: str = "cuda",
    project: str = "runs/train",
    name: str = "yolov11_plates",
):
    """
    Train YOLOv11 for plate detection.
    
    Args:
        data_config: Path to data config YAML
        model_size: YOLOv11 model size (n, s, m, l, x)
        epochs: Number of training epochs
        batch_size: Batch size
        image_size: Input image size
        device: Training device
        project: Project directory
        name: Experiment name
    """
    print("=" * 60)
    print("YOLOv11 Plate Detection Training")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_size}")
    print(f"  Device: {device}")
    print(f"  Data: {data_config}")
    
    model_name = f"{model_size}.pt"
    print(f"\n[1/3] Loading model: {model_name}")
    
    model = YOLO(model_name)
    
    print(f"\n[2/3] Starting training...")
    
    results = model.train(
        data=data_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        
        # Optimizer settings
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        
        # Other settings
        patience=50,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )
    
    print(f"\n[3/3] Training complete!")
    
    best_model_path = Path(project) / name / "weights" / "best.pt"
    if best_model_path.exists():
        print(f"\nBest model saved to: {best_model_path}")
    else:
        last_model_path = Path(project) / name / "weights" / "last.pt"
        print(f"\nLast model saved to: {last_model_path}")
    
    print("\nTraining results:")
    print(f"  Final mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  Final mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 for plate detection")
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/datasets/yolo_detection/data.yaml',
        help='Path to data config'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolov11s',
        choices=['yolov11n', 'yolov11s', 'yolov11m', 'yolov11l', 'yolov11x'],
        help='Model size'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda or cpu)'
    )
    
    parser.add_argument(
        '--project',
        type=str,
        default='runs/train',
        help='Project directory'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='yolov11_plates',
        help='Experiment name'
    )
    
    args = parser.parse_args()
    
    train_yolo(
        data_config=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
