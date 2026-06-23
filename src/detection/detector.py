"""
Detection Module - YOLOv11 License Plate Detection
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Please install ultralytics: pip install ultralytics")


@dataclass
class DetectionResult:
    """Single detection result"""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int = 0
    class_name: str = "license_plate"
    
    @property
    def xyxy(self) -> List[float]:
        return self.bbox
    
    @property
    def xywh(self) -> List[float]:
        """Convert to xywh format [x_center, y_center, width, height]"""
        x1, y1, x2, y2 = self.bbox
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        return [x_center, y_center, width, height]
    
    @property
    def area(self) -> float:
        """Calculate bounding box area"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class PlateDetector:
    """
    YOLOv11-based License Plate Detector
    
    Handles plate detection in images using YOLOv11 model.
    """
    
    def __init__(
        self,
        model_path: str = "models/yolov11/best.pt",
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        input_size: int = 640,
        device: str = "cuda"
    ):
        """
        Initialize the plate detector.
        
        Args:
            model_path: Path to YOLOv11 weights
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS IoU threshold
            input_size: Model input size (640 recommended for speed)
            device: 'cuda' or 'cpu'
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.device = device
        
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv11 model"""
        if not Path(self.model_path).exists():
            print(f"Warning: Model not found at {self.model_path}")
            print("Using pretrained yolov11s as fallback...")
            self.model = YOLO("yolov11s.pt")
        else:
            self.model = YOLO(self.model_path)
        
        self.model.to(self.device)
    
    def detect(
        self, 
        image: Union[str, np.ndarray, Path],
        return_cropped: bool = False,
        padding_percent: float = 0.1
    ) -> List[DetectionResult]:
        """
        Detect license plates in an image.
        
        Args:
            image: Image path, numpy array, or Path object
            return_cropped: If True, include cropped plate images
            padding_percent: Padding around detected plates (0-1)
            
        Returns:
            List of DetectionResult objects
        """
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not read image: {image}")
        else:
            img = image.copy()
        
        original_h, original_w = img.shape[:2]
        
        results = self.model.predict(
            img,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            imgsz=self.input_size,
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                box = boxes[i]
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                if padding_percent > 0:
                    pad_x = (x2 - x1) * padding_percent
                    pad_y = (y2 - y1) * padding_percent
                    
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(original_w, x2 + pad_x)
                    y2 = min(original_h, y2 + pad_y)
                
                detection = DetectionResult(
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    confidence=conf,
                    class_id=cls_id
                )
                
                if return_cropped:
                    detection.cropped_image = img[int(y1):int(y2), int(x1):int(x2)]
                
                detections.append(detection)
        
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        return detections
    
    def detect_batch(
        self, 
        images: List[Union[str, np.ndarray, Path]],
        **kwargs
    ) -> List[List[DetectionResult]]:
        """
        Detect plates in multiple images.
        
        Args:
            images: List of images
            **kwargs: Arguments passed to detect()
            
        Returns:
            List of detection lists
        """
        results = []
        for img in images:
            results.append(self.detect(img, **kwargs))
        return results
    
    def detect_from_video_frame(
        self,
        frame: np.ndarray,
        max_detections: int = 10
    ) -> List[DetectionResult]:
        """
        Fast detection optimized for video frames.
        
        Args:
            frame: Video frame as numpy array
            max_detections: Maximum number of plates to detect
            
        Returns:
            List of DetectionResult objects
        """
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            imgsz=self.input_size,
            device=self.device,
            max_det=max_detections,
            verbose=False
        )
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                detections.append(DetectionResult(
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    confidence=conf
                ))
        
        return detections


def visualize_detections(
    image: np.ndarray,
    detections: List[DetectionResult],
    show_confidence: bool = True,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw detection bounding boxes on image.
    
    Args:
        image: Input image
        detections: List of DetectionResult objects
        show_confidence: Show confidence score
        color: Box color (B, G, R)
        thickness: Line thickness
        
    Returns:
        Image with drawn boxes
    """
    img = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        if show_confidence:
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                img, 
                (x1, y1 - label_size[1] - 4), 
                (x1 + label_size[0], y1), 
                color, 
                -1
            )
            cv2.putText(
                img, 
                label, 
                (x1, y1 - 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
    
    return img
