"""
Metrics and evaluation utilities for Vietnamese LPR
"""

import json
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class DetectionMetrics:
    """Detection evaluation metrics"""
    precision: float
    recall: float
    f1_score: float
    map50: float
    map50_95: float
    total_gt: int
    total_pred: int
    tp: int
    fp: int
    fn: int


@dataclass  
class RecognitionMetrics:
    """Recognition evaluation metrics"""
    accuracy: float
    character_accuracy: float
    total_samples: int
    correct_samples: int
    mean_confidence: float
    confidence_std: float


@dataclass
class LPRMetrics:
    """End-to-end LPR evaluation metrics"""
    detection: DetectionMetrics
    recognition: RecognitionMetrics
    end_to_end_accuracy: float
    avg_processing_time_ms: float
    fps: float


def calculate_detection_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5
) -> DetectionMetrics:
    """
    Calculate detection metrics.
    
    Args:
        predictions: List of predictions with 'bbox' and 'confidence'
        ground_truth: List of ground truth with 'bbox'
        iou_threshold: IoU threshold for TP/FP/FN
        
    Returns:
        DetectionMetrics object
    """
    from ..utils.image_utils import calculate_iou
    
    tp = 0
    fp = 0
    matched_gt = set()
    
    for pred in predictions:
        pred_bbox = pred.get('bbox', [])
        if not pred_bbox:
            continue
            
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
                
            gt_bbox = gt.get('bbox', [])
            if not gt_bbox:
                continue
                
            iou = calculate_iou(pred_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
    
    fn = len(ground_truth) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1,
        map50=0.0,  # Would need full evaluation
        map50_95=0.0,
        total_gt=len(ground_truth),
        total_pred=len(predictions),
        tp=tp,
        fp=fp,
        fn=fn
    )


def calculate_recognition_metrics(
    predictions: List[str],
    ground_truth: List[str],
    confidences: Optional[List[float]] = None
) -> RecognitionMetrics:
    """
    Calculate recognition metrics.
    
    Args:
        predictions: List of predicted texts
        ground_truth: List of ground truth texts
        confidences: Optional confidence scores
        
    Returns:
        RecognitionMetrics object
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    accuracy = correct / len(predictions) if len(predictions) > 0 else 0
    
    total_chars = 0
    correct_chars = 0
    for p, g in zip(predictions, ground_truth):
        min_len = min(len(p), len(g))
        correct_chars += sum(1 for i in range(min_len) if p[i] == g[i])
        total_chars += max(len(p), len(g))
    
    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    
    conf_mean = 0
    conf_std = 0
    if confidences:
        conf_mean = np.mean(confidences)
        conf_std = np.std(confidences)
    
    return RecognitionMetrics(
        accuracy=accuracy,
        character_accuracy=char_accuracy,
        total_samples=len(predictions),
        correct_samples=correct,
        mean_confidence=conf_mean,
        confidence_std=conf_std
    )


def calculate_lpr_metrics(
    detection_results: List[Dict],
    ground_truth: List[Dict],
    processing_times: Optional[List[float]] = None
) -> LPRMetrics:
    """
    Calculate comprehensive LPR metrics.
    
    Args:
        detection_results: Detection + recognition results
        ground_truth: Ground truth annotations
        processing_times: Optional processing times per image
        
    Returns:
        LPRMetrics object
    """
    pred_plates = [r.get('plate', '') for r in detection_results]
    gt_plates = [g.get('plate', '') for g in ground_truth]
    confidences = [r.get('confidence', 0) for r in detection_results]
    
    detection_metrics = calculate_detection_metrics(
        detection_results, ground_truth
    )
    
    recognition_metrics = calculate_recognition_metrics(
        pred_plates, gt_plates, confidences
    )
    
    end_to_end = recognition_metrics.accuracy
    
    avg_time = 0
    fps = 0
    if processing_times:
        avg_time = np.mean(processing_times)
        fps = 1000 / avg_time if avg_time > 0 else 0
    
    return LPRMetrics(
        detection=detection_metrics,
        recognition=recognition_metrics,
        end_to_end_accuracy=end_to_end,
        avg_processing_time_ms=avg_time,
        fps=fps
    )


def generate_evaluation_report(
    metrics: LPRMetrics,
    output_path: Optional[str] = None
) -> str:
    """
    Generate evaluation report text.
    
    Args:
        metrics: LPRMetrics object
        output_path: Optional path to save report
        
    Returns:
        Report text
    """
    report = []
    report.append("=" * 60)
    report.append("VIETNAMESE LPR EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    report.append("DETECTION METRICS:")
    report.append("-" * 40)
    report.append(f"  Precision:      {metrics.detection.precision:.4f}")
    report.append(f"  Recall:         {metrics.detection.recall:.4f}")
    report.append(f"  F1 Score:       {metrics.detection.f1_score:.4f}")
    report.append(f"  True Positives: {metrics.detection.tp}")
    report.append(f"  False Positives:{metrics.detection.fp}")
    report.append(f"  False Negatives:{metrics.detection.fn}")
    report.append("")
    
    report.append("RECOGNITION METRICS:")
    report.append("-" * 40)
    report.append(f"  Word Accuracy:        {metrics.recognition.accuracy:.4f}")
    report.append(f"  Character Accuracy:   {metrics.recognition.character_accuracy:.4f}")
    report.append(f"  Mean Confidence:      {metrics.recognition.mean_confidence:.4f}")
    report.append(f"  Confidence Std:       {metrics.recognition.confidence_std:.4f}")
    report.append("")
    
    report.append("END-TO-END PERFORMANCE:")
    report.append("-" * 40)
    report.append(f"  End-to-End Accuracy: {metrics.end_to_end_accuracy:.4f}")
    report.append(f"  Avg Processing Time: {metrics.avg_processing_time_ms:.2f} ms")
    report.append(f"  FPS:                 {metrics.fps:.2f}")
    report.append("")
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
    
    return report_text


class PerformanceProfiler:
    """Profile LPR pipeline performance"""
    
    def __init__(self):
        self.timings = {
            'detection': [],
            'preprocessing': [],
            'recognition': [],
            'postprocessing': [],
            'total': []
        }
    
    def record(self, stage: str, duration_ms: float):
        """Record timing for a stage"""
        if stage in self.timings:
            self.timings[stage].append(duration_ms)
    
    def get_summary(self) -> Dict:
        """Get timing summary"""
        summary = {}
        for stage, times in self.timings.items():
            if times:
                summary[stage] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'count': len(times)
                }
        return summary
    
    def reset(self):
        """Reset all timings"""
        for stage in self.timings:
            self.timings[stage] = []
