"""
Phase 6: Model Evaluation Script
Run: python evaluate_model.py
"""

import sys
import cv2
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import importlib.util

sys.path.insert(0, str(Path(__file__).parent))

spec = importlib.util.spec_from_file_location('detector', 'src/detection/detector.py')
detector_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detector_module)
PlateDetector = detector_module.PlateDetector


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def evaluate_on_test_set(model_path, test_images_dir, ground_truth_dir=None, iou_threshold=0.5):
    """
    Evaluate model on test set.
    
    If ground_truth_dir is provided, calculates precision/recall.
    Otherwise, just runs detection and reports results.
    """
    print("="*60)
    print("PHASE 6: MODEL EVALUATION")
    print("="*60)
    
    detector = PlateDetector(model_path=model_path, confidence_threshold=0.25, device="cuda")
    
    test_images = list(Path(test_images_dir).glob("*.jpg")) + list(Path(test_images_dir).glob("*.png"))
    
    if len(test_images) == 0:
        print(f"No images found in {test_images_dir}")
        return
    
    print(f"\nFound {len(test_images)} test images")
    print(f"Model: {model_path}")
    print(f"IoU Threshold: {iou_threshold}")
    
    total_detections = 0
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    
    all_results = []
    inference_times = []
    
    for img_path in test_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        start_time = time.time()
        detections = detector.detect(img, return_cropped=True)
        inference_time = (time.time() - start_time) * 1000
        inference_times.append(inference_time)
        
        result = {
            "image": img_path.name,
            "detections": len(detections),
            "inference_ms": inference_time,
            "plates": []
        }
        
        for det in detections:
            result["plates"].append({
                "bbox": [float(v) for v in det.bbox],
                "confidence": float(det.confidence)
            })
        
        total_detections += len(detections)
        
        # If we have ground truth, calculate TP/FP/FN
        if ground_truth_dir:
            gt_path = Path(ground_truth_dir) / (img_path.stem + ".txt")
            if gt_path.exists():
                with open(gt_path) as f:
                    gt_boxes = [line.strip().split() for line in f.readlines()]
                
                matched_gt = set()
                for det in detections:
                    best_iou = 0
                    best_gt_idx = -1
                    for i, gt in enumerate(gt_boxes):
                        if i in matched_gt:
                            continue
                        # Convert YOLO format to pixels
                        # Format: class x_center y_center width height (normalized)
                        img_h, img_w = img.shape[:2]
                        x_c, y_c, w, h = map(float, gt[1:5])
                        gt_box = [
                            (x_c - w/2) * img_w,
                            (y_c - h/2) * img_h,
                            (x_c + w/2) * img_w,
                            (y_c + h/2) * img_h
                        ]
                        iou = calculate_iou(det.bbox, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = i
                    
                    if best_iou >= iou_threshold:
                        total_true_positives += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        total_false_positives += 1
                
                total_false_negatives += len(gt_boxes) - len(matched_gt)
        
        all_results.append(result)
    
    # Calculate metrics
    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    avg_inference_time = np.mean(inference_times)
    fps = 1000 / avg_inference_time if avg_inference_time > 0 else 0
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    if ground_truth_dir:
        print(f"\nDetection Metrics:")
        print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"\nCounts:")
        print(f"  True Positives:  {total_true_positives}")
        print(f"  False Positives: {total_false_positives}")
        print(f"  False Negatives: {total_false_negatives}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Total Images:      {len(test_images)}")
    print(f"  Total Detections:  {total_detections}")
    print(f"  Avg Inference:     {avg_inference_time:.2f} ms")
    print(f"  FPS:               {fps:.2f}")
    
    # Create summary report
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "test_images_dir": str(test_images_dir),
        "iou_threshold": iou_threshold,
        "metrics": {
            "total_images": len(test_images),
            "total_detections": total_detections,
            "precision": float(precision) if ground_truth_dir else None,
            "recall": float(recall) if ground_truth_dir else None,
            "f1_score": float(f1) if ground_truth_dir else None,
            "avg_inference_ms": float(avg_inference_time),
            "fps": float(fps)
        },
        "per_image_results": all_results
    }
    
    # Save report
    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nReport saved: {report_path}")
    print("="*60)
    
    return report


if __name__ == "__main__":
    import sys
    
    # Default paths
    model_path = "weights/best.pt"
    test_images_dir = "LicensePlateDetectionDataset/images/test"
    ground_truth_dir = None  # Set to labels dir if available
    
    # Check if labels exist
    labels_dir = "LicensePlateDetectionDataset/labels/test"
    if Path(labels_dir).exists():
        ground_truth_dir = labels_dir
        print(f"Found labels at {labels_dir}")
    
    evaluate_on_test_set(model_path, test_images_dir, ground_truth_dir)
