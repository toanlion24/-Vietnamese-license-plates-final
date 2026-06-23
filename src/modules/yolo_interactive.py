"""
Module 2: YOLOv11 Detection - Interactive Wrapper
Chủ động điều chỉnh input source và detection parameters
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError:
    logger.warning("ultralytics not installed")


class DetectionMode(Enum):
    """Chế độ detection"""
    PLATE_ONLY = "plate_only"
    VEHICLE_ONLY = "vehicle_only"
    BOTH = "both"


class InputSource(Enum):
    """Nguồn input"""
    IMAGE = "image"
    VIDEO = "video"
    WEBCAM = "webcam"
    FOLDER = "folder"
    FRAME = "frame"


@dataclass
class BoundingBox:
    """Bounding box representation"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.0
    class_id: int = 0
    class_name: str = "plate"
    
    @property
    def xyxy(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]
    
    @property
    def xywh(self) -> List[float]:
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2
        return [cx, cy, self.width, self.height]
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def crop_image(self, image: np.ndarray) -> np.ndarray:
        """Crop vùng bbox từ ảnh gốc"""
        h, w = image.shape[:2]
        x1, y1 = int(max(0, self.x1)), int(max(0, self.y1))
        x2, y2 = int(min(w, self.x2)), int(min(h, self.y2))
        return image[y1:y2, x1:x2]


@dataclass
class DetectionResult:
    """Kết quả detection cho một frame"""
    frame_id: int = 0
    timestamp: float = 0.0
    source: str = ""
    plates: List[BoundingBox] = field(default_factory=list)
    vehicles: List[BoundingBox] = field(default_factory=list)
    image_shape: Tuple[int, int] = (0, 0)
    raw_image: np.ndarray = None
    
    @property
    def total_detections(self) -> int:
        return len(self.plates) + len(self.vehicles)
    
    @property
    def has_plates(self) -> bool:
        return len(self.plates) > 0
    
    def get_plates_cropped(self, image: np.ndarray) -> List[np.ndarray]:
        """Trả về list ảnh biển số đã crop"""
        return [plate.crop_image(image) for plate in self.plates]
    
    def to_dict(self) -> Dict:
        """Convert sang dictionary để save JSON"""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "plates": [
                {
                    "bbox": p.xyxy,
                    "confidence": p.confidence,
                    "class_name": p.class_name
                } for p in self.plates
            ],
            "vehicles": [
                {
                    "bbox": v.xyxy,
                    "confidence": v.confidence,
                    "class_name": v.class_name
                } for v in self.vehicles
            ],
            "total_detections": self.total_detections
        }


class YOLOInteractiveDetector:
    """
    Wrapper linh hoạt cho YOLOv11 Detection
    
    Features:
    - Chọn input source: image, video, webcam, folder
    - Điều chỉnh parameters: conf, iou, device
    - Chế độ detection: plate only, vehicle only, both
    - Tùy chỉnh output: crop, visualize, JSON
    """
    
    # COCO vehicle classes
    VEHICLE_CLASSES = {
        2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat'
    }
    
    def __init__(
        self,
        model_path: str = "weights/best.pt",
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        input_size: int = 640,
        detection_mode: DetectionMode = DetectionMode.PLATE_ONLY,
        warmup: bool = True,
    ):
        """
        Khởi tạo detector với các tham số tùy chỉnh
        
        Args:
            model_path: Đường dẫn model weights
            confidence: Ngưỡng confidence (0.0 - 1.0)
            iou_threshold: Ngưỡng NMS IOU
            device: 'cuda' hoặc 'cpu'
            input_size: Kích thước input model
            detection_mode: PLATE_ONLY, VEHICLE_ONLY, hoặc BOTH
            warmup: Warmup model khi khởi tạo
        """
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self.input_size = input_size
        self.detection_mode = detection_mode
        
        self._model = None
        self._load_model()
        
        if warmup:
            self.warmup()
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            logger.info(f"Loading model: {self.model_path}")
            if Path(self.model_path).exists():
                self._model = YOLO(self.model_path)
            else:
                logger.warning(f"Model not found: {self.model_path}, using fallback")
                self._model = YOLO("yolov8n.pt")
            self._model.to(self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._model = None
    
    def warmup(self):
        """Warmup model"""
        if self._model:
            dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            self._model.predict(dummy, verbose=False)
            logger.info("Model warmed up")
    
    def set_confidence(self, conf: float):
        """Thay đổi confidence threshold"""
        self.confidence = max(0.0, min(1.0, conf))
    
    def set_iou(self, iou: float):
        """Thay đổi NMS IOU threshold"""
        self.iou_threshold = max(0.0, min(1.0, iou))
    
    def set_mode(self, mode: DetectionMode):
        """Thay đổi detection mode"""
        self.detection_mode = mode
    
    def detect(
        self,
        image: np.ndarray,
        return_cropped: bool = True,
        return_visualized: bool = False,
        frame_id: int = 0,
        timestamp: float = 0.0,
        source: str = "frame"
    ) -> Union[DetectionResult, Tuple[DetectionResult, np.ndarray], Tuple[DetectionResult, List[np.ndarray]]]:
        """
        Detect objects in image
        
        Args:
            image: Input image (BGR format)
            return_cropped: Trả về ảnh đã crop
            return_visualized: Trả về ảnh có vẽ bbox
            frame_id: Frame ID cho video
            timestamp: Timestamp cho video
            source: Tên nguồn input
            
        Returns:
            DetectionResult hoặc tuple (DetectionResult, cropped_images/visualized_image)
        """
        result = DetectionResult(
            frame_id=frame_id,
            timestamp=timestamp,
            source=source,
            image_shape=image.shape[:2],
            raw_image=image.copy() if return_visualized else None
        )
        
        if self._model is None:
            return result
        
        # Run inference
        results = self._model.predict(
            image,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            device=self.device,
            verbose=False
        )
        
        if not results or len(results) == 0:
            return self._format_output(result, return_cropped, return_visualized)
        
        boxes = results[0].boxes
        
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].cpu().numpy())
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            
            bbox = BoundingBox(
                x1=float(x1), y1=float(y1),
                x2=float(x2), y2=float(y2),
                confidence=conf, class_id=cls_id
            )
            
            # Phân loại vehicle vs plate
            if cls_id in self.VEHICLE_CLASSES:
                if self.detection_mode in [DetectionMode.VEHICLE_ONLY, DetectionMode.BOTH]:
                    bbox.class_name = self.VEHICLE_CLASSES[cls_id]
                    result.vehicles.append(bbox)
            else:
                # Plate class
                if self.detection_mode in [DetectionMode.PLATE_ONLY, DetectionMode.BOTH]:
                    bbox.class_name = "plate"
                    result.plates.append(bbox)
        
        return self._format_output(result, return_cropped, return_visualized)
    
    def _format_output(
        self,
        result: DetectionResult,
        return_cropped: bool,
        return_visualized: bool
    ):
        """Format output theo yêu cầu"""
        outputs = [result]
        
        if return_cropped and result.has_plates:
            cropped = result.get_plates_cropped(result.raw_image or np.zeros(result.image_shape, dtype=np.uint8))
            outputs.append(cropped)
        elif return_cropped:
            outputs.append([])
        
        if return_visualized and result.raw_image is not None:
            vis = self.visualize(result, result.raw_image)
            outputs.append(vis)
        
        if len(outputs) == 1:
            return outputs[0]
        elif len(outputs) == 2:
            return tuple(outputs)
        else:
            return tuple(outputs)
    
    def detect_from_image(
        self,
        image_path: str,
        **kwargs
    ) -> DetectionResult:
        """Detect từ file ảnh"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return self.detect(image, source=Path(image_path).name, **kwargs)
    
    def detect_from_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        callback: Optional[Callable[[DetectionResult], None]] = None,
        **kwargs
    ) -> List[DetectionResult]:
        """
        Detect từ video
        
        Args:
            video_path: Đường dẫn video
            max_frames: Số frame tối đa (None = all)
            callback: Hàm gọi sau mỗi frame (để real-time processing)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        results = []
        frame_id = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_id / fps if fps > 0 else frame_id
            result = self.detect(frame, frame_id=frame_id, timestamp=timestamp, source=video_path, **kwargs)
            results.append(result)
            
            if callback:
                callback(result)
            
            frame_id += 1
            if max_frames and frame_id >= max_frames:
                break
        
        cap.release()
        return results
    
    def detect_from_webcam(
        self,
        camera_id: int = 0,
        max_frames: Optional[int] = None,
        callback: Optional[Callable[[DetectionResult], bool]] = None,
        **kwargs
    ) -> List[DetectionResult]:
        """
        Detect từ webcam
        
        Args:
            camera_id: Camera index
            max_frames: Số frame tối đa
            callback: Trả về True để tiếp tục, False để dừng
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")
        
        results = []
        frame_id = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                result = self.detect(frame, frame_id=frame_id, source=f"webcam_{camera_id}", **kwargs)
                results.append(result)
                
                if callback:
                    should_continue = callback(result)
                    if not should_continue:
                        break
                
                frame_id += 1
                if max_frames and frame_id >= max_frames:
                    break
        finally:
            cap.release()
        
        return results
    
    def detect_from_folder(
        self,
        folder_path: str,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp'),
        **kwargs
    ) -> Dict[str, DetectionResult]:
        """
        Detect tất cả ảnh trong folder
        
        Args:
            folder_path: Đường dẫn folder
            extensions: Các extension cần xử lý
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        results = {}
        for img_path in folder.rglob('*'):
            if img_path.suffix.lower() in extensions:
                try:
                    result = self.detect_from_image(str(img_path), **kwargs)
                    results[str(img_path)] = result
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")
        
        return results
    
    def visualize(
        self,
        result: DetectionResult,
        image: np.ndarray,
        show_plates: bool = True,
        show_vehicles: bool = True,
        plate_color: Tuple[int, int, int] = (0, 255, 0),
        vehicle_color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
        show_conf: bool = True,
        show_label: bool = True,
    ) -> np.ndarray:
        """Visualize detections lên ảnh"""
        img = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw plates
        if show_plates:
            for plate in result.plates:
                x1, y1, x2, y2 = map(int, plate.xyxy)
                cv2.rectangle(img, (x1, y1), (x2, y2), plate_color, thickness)
                
                if show_label:
                    label = f"PLATE: {plate.confidence:.0%}" if show_conf else "PLATE"
                    cv2.putText(img, label, (x1, y1 - 5), font, 0.5, plate_color, 1)
        
        # Draw vehicles
        if show_vehicles:
            for vehicle in result.vehicles:
                x1, y1, x2, y2 = map(int, vehicle.xyxy)
                cv2.rectangle(img, (x1, y1), (x2, y2), vehicle_color, thickness)
                
                if show_label:
                    label = f"{vehicle.class_name}: {vehicle.confidence:.0%}" if show_conf else vehicle.class_name
                    cv2.putText(img, label, (x1, y1 - 5), font, 0.5, vehicle_color, 1)
        
        # Summary
        summary = f"Plates: {len(result.plates)} | Vehicles: {len(result.vehicles)}"
        cv2.putText(img, summary, (10, 25), font, 0.6, (255, 255, 255), 2)
        
        return img
    
    def save_results(
        self,
        results: Union[DetectionResult, List[DetectionResult]],
        output_path: str,
        include_images: bool = False
    ):
        """
        Save results ra JSON
        
        Args:
            results: DetectionResult hoặc list
            output_path: Đường dẫn output JSON
            include_images: Có lưu ảnh (base64) không
        """
        if isinstance(results, DetectionResult):
            results = [results]
        
        output = {
            "total_frames": len(results),
            "detections": [r.to_dict() for r in results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved {len(results)} results to {output_path}")


# ============================================================
# EASY API - Cách sử dụng nhanh
# ============================================================

def quick_detect(
    image: Union[str, np.ndarray],
    model_path: str = "weights/best.pt",
    confidence: float = 0.25,
    device: str = "cuda",
    return_cropped: bool = True,
    return_visualized: bool = False,
) -> Union[DetectionResult, Tuple]:
    """
    Quick detection - interface đơn giản nhất
    
    Args:
        image: Ảnh (path hoặc numpy array)
        model_path: Đường dẫn model
        confidence: Ngưỡng confidence
        device: cuda/cpu
        return_cropped: Trả về ảnh crop
        return_visualized: Trả về ảnh visualize
        
    Returns:
        DetectionResult hoặc tuple (result, cropped_images, visualized_image)
    """
    detector = YOLOInteractiveDetector(
        model_path=model_path,
        confidence=confidence,
        device=device,
        warmup=False
    )
    
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Cannot read: {image}")
    else:
        img = image
    
    return detector.detect(
        img,
        return_cropped=return_cropped,
        return_visualized=return_visualized,
        source=image if isinstance(image, str) else "array"
    )


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MODULE 2: YOLOv11 Interactive Detection")
    print("=" * 60)
    print()
    
    # Ví dụ 1: Quick detect
    print("[1] Quick Detect - Đơn giản nhất")
    print("-" * 40)
    result, cropped = quick_detect(
        "LicensePlateDetectionDataset/images/test/boderngoaigiao1.jpg",
        confidence=0.25
    )
    print(f"Found: {len(result.plates)} plate(s)")
    for p in result.plates:
        print(f"  - Bbox: {[round(x, 1) for x in p.xyxy]}, Conf: {p.confidence:.2%}")
    print()
    
    # Ví dụ 2: Tùy chỉnh nhiều tham số
    print("[2] Custom Detector - Nhiều tùy chỉnh")
    print("-" * 40)
    detector = YOLOInteractiveDetector(
        model_path="weights/best.pt",
        confidence=0.20,  # Giảm để detect nhiều hơn
        iou_threshold=0.50,
        device="cuda",
        detection_mode=DetectionMode.PLATE_ONLY
    )
    
    # Detect từ webcam (10 frames)
    print("Webcam detection (10 frames):")
    # results = detector.detect_from_webcam(0, max_frames=10)
    print("  (Webcam demo disabled - uncomment to use)")
    print()
    
    # Ví dụ 3: Detect video
    print("[3] Video Detection")
    print("-" * 40)
    print("  detector.detect_from_video('video.mp4', max_frames=100)")
    print()
    
    # Ví dụ 4: Detect folder
    print("[4] Folder Detection")
    print("-" * 40)
    print("  results = detector.detect_from_folder('test_images/')")
    print("  detector.save_results(results, 'output.json')")
    print()
    
    print("=" * 60)
    print("✅ Module 2 ready!")
    print("=" * 60)
