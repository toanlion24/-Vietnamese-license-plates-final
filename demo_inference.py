"""
Demo script to test trained models on Colab
Usage:
    python demo_inference.py --image path/to/image.jpg
    python demo_inference.py --image path/to/image.jpg --full-pipeline
"""

import argparse
import cv2
import sys
import json
import importlib.util
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

spec = importlib.util.spec_from_file_location('detector', 'src/detection/detector.py')
detector_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detector_module)
PlateDetector = detector_module.PlateDetector
visualize_detections = detector_module.visualize_detections


def main():
    parser = argparse.ArgumentParser(description="Test trained models")
    parser.add_argument("--image", type=str, default=None, help="Path to test image")
    parser.add_argument("--detector", type=str, default="data/best_plate_detector.pt", 
                        help="Path to detector model (.pt)")
    parser.add_argument("--onnx", type=str, default="data/best.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--use-onnx", action="store_true", help="Use ONNX model")
    parser.add_argument("--full-pipeline", action="store_true", help="Run full LPR pipeline (detection + recognition)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", type=str, default=None, help="Output folder name")
    args = parser.parse_args()

    if args.image is None:
        test_images = list(Path("LicensePlateDetectionDataset/images/test").glob("*.jpg"))[:1]
        if test_images:
            args.image = str(test_images[0])
        else:
            print("Error: No test image found. Please specify --image")
            sys.exit(1)

    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    if args.use_onnx:
        print("[Demo] ONNX support requires onnxruntime. Using PyTorch model instead.")
    
    # Create output folder
    input_stem = Path(args.image).stem
    if args.output:
        output_dir = Path("outputs") / args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"{input_stem}_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Demo] Using PyTorch model: {args.detector}")
    print(f"[Demo] Image: {args.image}")
    print(f"[Demo] Output: {output_dir}")
    
    detector = PlateDetector(
        model_path=args.detector,
        confidence_threshold=args.conf,
        device=args.device
    )

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not read image: {args.image}")
        sys.exit(1)

    print(f"\n[Demo] Running detection...")
    
    detections = detector.detect(img, return_cropped=True)
    
    results_data = {
        "image": args.image,
        "model": args.detector,
        "timestamp": datetime.now().isoformat(),
        "total_detections": len(detections),
        "detections": []
    }
    
    print("\n" + "="*50)
    print("DETECTION RESULTS:")
    print("="*50)
    
    for i, det in enumerate(detections):
        det_info = {
            "plate_id": i + 1,
            "bbox": [float(v) for v in det.bbox],
            "confidence": float(det.confidence),
        }
        
        print(f"  Plate {i+1}:")
        print(f"    BBox: [{det.bbox[0]:.0f}, {det.bbox[1]:.0f}, {det.bbox[2]:.0f}, {det.bbox[3]:.0f}]")
        print(f"    Confidence: {det.confidence:.2%}")
        
        if hasattr(det, 'cropped_image') and det.cropped_image is not None:
            crop_filename = f"plate_{i+1:02d}_crop.jpg"
            crop_path = output_dir / crop_filename
            cv2.imwrite(str(crop_path), det.cropped_image)
            det_info["crop_file"] = crop_filename
            print(f"    Saved crop: {crop_filename}")
        
        results_data["detections"].append(det_info)
    
    if len(detections) == 0:
        print("  No plates detected!")
    
    print("="*50)
    
    # Save detected image
    debug_img = visualize_detections(img, detections)
    detected_path = output_dir / "detected.jpg"
    cv2.imwrite(str(detected_path), debug_img)
    
    # Save original image
    original_path = output_dir / "original.jpg"
    cv2.imwrite(str(original_path), img)
    
    # Save results JSON
    results_path = output_dir / "results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[DEMO] Results saved to: {output_dir}/")
    print(f"  - detected.jpg")
    print(f"  - original.jpg")
    print(f"  - results.json")
    if len(detections) > 0:
        print(f"  - plate_01_crop.jpg ... plate_{len(detections):02d}_crop.jpg")
    print("[DEMO] Done!")


if __name__ == "__main__":
    main()
