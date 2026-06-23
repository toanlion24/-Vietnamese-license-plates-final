"""
YOLOv11 Plate Detection Training Script
Trains YOLOv11 on Vietnamese License Plate Dataset
"""

import os
import sys
from pathlib import Path
import argparse
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def train_yolov11(
    data_yaml: str,
    model_size: str = "s",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "0",
    project_name: str = "yolov11_plates",
    resume: bool = False,
    weights: str = None,
):
    """
    Train YOLOv11 for plate detection.
    
    Args:
        data_yaml: Path to data.yaml
        model_size: Model size (n, s, m, l, x)
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size
        device: Device (0, 1, cpu)
        project_name: Project name for runs
        resume: Resume from last checkpoint
        weights: Custom weights path
    """
    from ultralytics import YOLO
    
    print("=" * 60)
    print("YOLOV11 PLATE DETECTION TRAINING")
    print("=" * 60)
    
    # Validate data.yaml
    if not Path(data_yaml).exists():
        print(f"[ERROR] data.yaml not found: {data_yaml}")
        return
    
    # Load and display config
    with open(data_yaml) as f:
        config = yaml.safe_load(f)
    
    print(f"\n[*] Dataset: {config.get('dataset_info', {}).get('name', 'Unknown')}")
    print(f"[*] Total images: {config.get('dataset_info', {}).get('total_images', 'N/A')}")
    print(f"[*] Model: yolov11{model_size}")
    print(f"[*] Epochs: {epochs}")
    print(f"[*] Batch size: {batch_size}")
    print(f"[*] Image size: {img_size}")
    print(f"[*] Device: {device}")
    
    # Model selection
    if weights:
        print(f"[*] Loading custom weights: {weights}")
        model = YOLO(weights)
    else:
        model_name = f"yolov11{model_size}.pt"
        print(f"[*] Loading pretrained: {model_name}")
        model = YOLO(model_name)
    
    # Training
    print("\n[*] Starting training...")
    print("-" * 60)
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project_name,
        name="exp",
        exist_ok=True,
        resume=resume,
        
        # Optimizer settings
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Scheduler
        scheduler='cosine',
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        
        # Validation
        val=True,
        save=True,
        save_json=True,
        save_hybrid=True,
        conf=0.001,
        iou=0.6,
        max_det=300,
        
        # Other
        verbose=True,
        workers=8,
        seed=0,
        pretrained=True,
        patience=50,
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    # Print results
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"\n[*] Best mAP@0.5: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"[*] Best mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    
    # Export model
    print("\n[*] Exporting model...")
    export_path = model.export(format="onnx")
    print(f"[+] Model exported to: {export_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 for plate detection")
    
    # Data
    parser.add_argument(
        "--data",
        type=str,
        default="D:/ComputerVisionLasted/data/datasets/plate_detection/data.yaml",
        help="Path to data.yaml"
    )
    
    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="s",
        choices=["n", "s", "m", "l", "x"],
        help="Model size: n(nano), s(small), m(medium), l(large), x(xlarge)"
    )
    
    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="Device (0, 1, cpu)")
    
    # Paths
    parser.add_argument("--project", type=str, default="runs/detect", help="Project folder")
    parser.add_argument("--weights", type=str, default=None, help="Custom weights path")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    
    args = parser.parse_args()
    
    # Train
    train_yolov11(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        project_name=args.project,
        resume=args.resume,
        weights=args.weights,
    )


if __name__ == "__main__":
    main()
