"""
Module 2: YOLOv11 Detection
Detects vehicles and license plates in images
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError:
    logger.warning("ultralytics not installed. Using fallback detection.")


@dataclass
class BoundingBox:
    """Bounding box representation"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.0
    class_id: int = 0
    class_name: str = ""
    
    @property
    def xyxy(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]
    
    @property
    def xywh(self) -> List[float]:
        """Center x, center y, width, height"""
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        return [cx, cy, w, h]
    
    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate IoU with another box"""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.area + other.area - inter
        
        return inter / union if union > 0 else 0
    
    def contains(self, other: 'BoundingBox') -> float:
        """Calculate how much of 'other' is inside this box (0-1)"""
        if not self.intersects(other):
            return 0.0
        
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        return inter / other.area if other.area > 0 else 0
    
    def intersects(self, other: 'BoundingBox') -> bool:
        return not (self.x2 < other.x1 or self.x1 > other.x2 or
                    self.y2 < other.y1 or self.y1 > other.y2)


@dataclass
class DetectionResult:
    """Detection result container"""
    vehicles: List[BoundingBox] = field(default_factory=list)
    plates: List[BoundingBox] = field(default_factory=list)
    timestamp: float = 0.0
    frame_id: int = 0
    image_shape: Tuple[int, int] = (0, 0)
    
    @property
    def total_detections(self) -> int:
        return len(self.vehicles) + len(self.plates)
    
    def has_plates(self) -> bool:
        return len(self.plates) > 0
    
    def has_vehicles(self) -> bool:
        return len(self.vehicles) > 0


class YOLODetector:
    """
    YOLOv11-based detector for vehicles and license plates.
    
    Supports:
    - Vehicle detection (car, truck, bus, motorcycle, etc.)
    - License plate detection
    - Multi-class detection
    """
    
    # COCO vehicle class IDs
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle', 
        4: 'airplane',
        5: 'bus',
        6: 'train',
        7: 'truck',
        8: 'boat',
    }
    
    # Plate class (custom, use 0 if trained on custom model)
    PLATE_CLASS = 0
    
    def __init__(
        self,
        vehicle_model: str = "yolov11s.pt",
        plate_model: Optional[str] = None,
        device: str = "cuda",
        conf_vehicle: float = 0.25,
        conf_plate: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 640,
    ):
        """
        Initialize detector.
        
        Args:
            vehicle_model: Path to vehicle detection model
            plate_model: Path to plate detection model (optional)
            device: 'cuda' or 'cpu'
            conf_vehicle: Confidence threshold for vehicles
            conf_plate: Confidence threshold for plates
            iou_threshold: NMS IoU threshold
            input_size: Model input size
        """
        self.vehicle_model_path = vehicle_model
        self.plate_model_path = plate_model
        self.device = device
        self.conf_vehicle = conf_vehicle
        self.conf_plate = conf_plate
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        
        self._vehicle_model = None
        self._plate_model = None
        
        self._load_models()
    
    def _load_models(self):
        """Load YOLO models"""
        # Load vehicle model
        try:
            logger.info(f"Loading vehicle model: {self.vehicle_model_path}")
            self._vehicle_model = YOLO(self.vehicle_model_path)
            self._vehicle_model.to(self.device)
            logger.info("Vehicle model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load vehicle model: {e}")
            self._vehicle_model = None
        
        # Load plate model if provided
        if self.plate_model_path:
            try:
                logger.info(f"Loading plate model: {self.plate_model_path}")
                self._plate_model = YOLO(self.plate_model_path)
                self._plate_model.to(self.device)
                logger.info("Plate model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load plate model: {e}")
                self._plate_model = None
    
    def detect(
        self,
        image: np.ndarray,
        detect_vehicles: bool = True,
        detect_plates: bool = True,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> DetectionResult:
        """
        Detect vehicles and plates in image.
        
        Args:
            image: Input image (BGR format)
            detect_vehicles: Enable vehicle detection
            detect_plates: Enable plate detection
            frame_id: Frame ID for tracking
            timestamp: Frame timestamp
            
        Returns:
            DetectionResult with vehicles and plates
        """
        result = DetectionResult(
            frame_id=frame_id,
            timestamp=timestamp,
            image_shape=image.shape[:2]
        )
        
        if detect_vehicles and self._vehicle_model:
            result.vehicles = self._detect_vehicles(image)
        
        if detect_plates and (self._plate_model or self._vehicle_model):
            result.plates = self._detect_plates(image)
        
        return result
    
    def _detect_vehicles(self, image: np.ndarray) -> List[BoundingBox]:
        """Detect vehicles using YOLO"""
        if self._vehicle_model is None:
            return []
        
        results = self._vehicle_model.predict(
            image,
            conf=self.conf_vehicle,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            device=self.device,
            verbose=False,
        )
        
        vehicles = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                if cls_id not in self.VEHICLE_CLASSES:
                    continue
                
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                
                vehicles.append(BoundingBox(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=self.VEHICLE_CLASSES[cls_id]
                ))
        
        return vehicles
    
    def _detect_plates(self, image: np.ndarray) -> List[BoundingBox]:
        """Detect license plates using YOLO"""
        if self._plate_model:
            model = self._plate_model
            conf = self.conf_plate
        elif self._vehicle_model:
            model = self._vehicle_model
            conf = self.conf_plate
        else:
            return []
        
        results = model.predict(
            image,
            conf=conf,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            device=self.device,
            verbose=False,
        )
        
        plates = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                # For custom plate model, class 0 is plate
                # For vehicle model, skip vehicle classes
                if self._plate_model is None and cls_id in self.VEHICLE_CLASSES:
                    continue
                
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                plate_conf = float(boxes.conf[i].cpu().numpy())
                
                plates.append(BoundingBox(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    confidence=plate_conf,
                    class_id=cls_id,
                    class_name='license_plate'
                ))
        
        return plates
    
    def detect_vehicles_only(self, image: np.ndarray) -> List[BoundingBox]:
        """Detect only vehicles"""
        return self._detect_vehicles(image)
    
    def detect_plates_only(self, image: np.ndarray) -> List[BoundingBox]:
        """Detect only plates"""
        return self._detect_plates(image)
    
    def warmup(self, image_size: Tuple[int, int] = (640, 640)):
        """Warmup models with dummy input"""
        dummy = np.zeros((*image_size, 3), dtype=np.uint8)
        logger.info("Warming up models...")
        self.detect(dummy, detect_vehicles=True, detect_plates=True)
        logger.info("Warmup complete")


class FallbackDetector:
    """
    Fallback detector when YOLO is not available.
    Uses basic image processing for demo purposes.
    """
    
    def __init__(self, **kwargs):
        self.conf_vehicle = kwargs.get('conf_vehicle', 0.3)
        self.conf_plate = kwargs.get('conf_plate', 0.3)
    
    def detect(self, image: np.ndarray, **kwargs) -> DetectionResult:
        """Simple edge-based detection demo"""
        result = DetectionResult(
            frame_id=kwargs.get('frame_id', 0),
            timestamp=kwargs.get('timestamp', 0.0),
            image_shape=image.shape[:2]
        )
        
        # Placeholder - return empty results
        # In production, this should use actual detection
        return result


def visualize_detections(
    image: np.ndarray,
    detections: DetectionResult,
    show_vehicles: bool = True,
    show_plates: bool = True,
    show_ids: bool = True,
    vehicle_color: Tuple[int, int, int] = (0, 255, 0),
    plate_color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw detection bounding boxes on image.
    
    Args:
        image: Input image
        detections: DetectionResult
        show_vehicles: Draw vehicle boxes
        show_plates: Draw plate boxes
        show_ids: Show class names
        vehicle_color: Vehicle box color (BGR)
        plate_color: Plate box color (BGR)
        thickness: Line thickness
        
    Returns:
        Image with drawn boxes
    """
    img = image.copy()
    
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Draw vehicles
    if show_vehicles:
        for box in detections.vehicles:
            x1, y1, x2, y2 = map(int, box.xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), vehicle_color, thickness)
            
            if show_ids:
                label = f"{box.class_name}: {box.confidence:.2f}"
                label_size = cv2.getTextSize(label, font, 0.5, 1)[0]
                cv2.rectangle(
                    img,
                    (x1, y1 - label_size[1] - 4),
                    (x1 + label_size[0], y1),
                    vehicle_color,
                    -1
                )
                cv2.putText(
                    img, label,
                    (x1, y1 - 2),
                    font, 0.5, (255, 255, 255), 1
                )
    
    # Draw plates
    if show_plates:
        for box in detections.plates:
            x1, y1, x2, y2 = map(int, box.xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), plate_color, thickness)
            
            if show_ids:
                label = f"PLATE: {box.confidence:.2f}"
                label_size = cv2.getTextSize(label, font, 0.5, 1)[0]
                cv2.rectangle(
                    img,
                    (x1, y1 - label_size[1] - 4),
                    (x1 + label_size[0], y1),
                    plate_color,
                    -1
                )
                cv2.putText(
                    img, label,
                    (x1, y1 - 2),
                    font, 0.5, (255, 255, 255), 1
                )
    
    # Add summary
    summary = f"Vehicles: {len(detections.vehicles)} | Plates: {len(detections.plates)}"
    cv2.putText(
        img, summary,
        (10, 25),
        font, 0.7, (255, 255, 255), 2
    )
    
    return img
