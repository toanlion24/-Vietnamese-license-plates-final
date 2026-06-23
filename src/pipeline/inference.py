"""
Vietnamese LPR Pipeline - End-to-End License Plate Recognition
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import yaml

from ..detection import PlateDetector, DetectionResult
from ..recognition import (
    PlateRecognizer, 
    preprocess_for_recognition,
    normalize_vietnamese_plate,
    validate_plate_format,
)


@dataclass
class LPRResult:
    """Complete LPR result for a single plate"""
    plate: str
    confidence: float
    detection_confidence: float
    recognition_confidence: float
    bbox: List[float]
    plate_type: Optional[str] = None
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def __str__(self) -> str:
        """String representation"""
        return f"LPRResult(plate='{self.plate}', conf={self.confidence:.2f})"


@dataclass
class LPRFrameResult:
    """Results for a single frame"""
    frame_id: int
    timestamp: float
    results: List[LPRResult]
    total_processing_time_ms: float
    
    def to_dict(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "plates": [r.to_dict() for r in self.results],
            "processing_time_ms": self.total_processing_time_ms,
        }


class VietnameseLPRPipeline:
    """
    End-to-End Vietnamese License Plate Recognition Pipeline
    
    Combines YOLOv11 detection with PaddleOCR recognition
    for complete plate recognition.
    """
    
    def __init__(
        self,
        detector_weights: str = "models/yolov11/best.pt",
        recognizer_config: Optional[str] = None,
        recognizer_weights: Optional[str] = None,
        dictionary_path: str = "configs/vietnamese_dict.txt",
        device: str = "cuda",
        detection_conf: float = 0.25,
        recognition_conf: float = 0.5,
        min_confidence: float = 0.7,
        config_path: Optional[str] = None,
    ):
        """
        Initialize the LPR pipeline.
        
        Args:
            detector_weights: Path to YOLOv11 detection model
            recognizer_config: Path to PaddleOCR config
            recognizer_weights: Path to PaddleOCR recognition model
            dictionary_path: Path to Vietnamese character dictionary
            device: 'cuda' or 'cpu'
            detection_conf: Detection confidence threshold
            recognition_conf: Recognition confidence threshold
            min_confidence: Minimum combined confidence for output
            config_path: Optional path to pipeline config YAML
        """
        self.device = device
        self.detection_conf = detection_conf
        self.recognition_conf = recognition_conf
        self.min_confidence = min_confidence
        
        if config_path and Path(config_path).exists():
            self._load_config(config_path)
        
        print(f"[VietnameseLPR] Initializing pipeline on {device}...")
        print(f"[VietnameseLPR] Loading detector: {detector_weights}")
        
        self.detector = PlateDetector(
            model_path=detector_weights,
            confidence_threshold=detection_conf,
            device=device,
        )
        
        print(f"[VietnameseLPR] Loading recognizer...")
        self.recognizer = PlateRecognizer(
            config_path=recognizer_config,
            model_path=recognizer_weights,
            dictionary_path=dictionary_path,
            use_gpu=(device == "cuda"),
        )
        
        print("[VietnameseLPR] Pipeline initialized successfully!")
    
    def _load_config(self, config_path: str):
        """Load configuration from YAML"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'detection' in config:
            self.detection_conf = config['detection'].get('confidence_threshold', self.detection_conf)
        if 'recognition' in config:
            self.recognition_conf = config['recognition'].get('confidence_threshold', self.recognition_conf)
        if 'postprocessing' in config:
            self.min_confidence = config['postprocessing'].get('min_confidence', self.min_confidence)
    
    def process_image(
        self,
        image: Union[str, np.ndarray, Path],
        return_debug_image: bool = False,
        padding_percent: float = 0.1,
    ) -> Union[List[LPRResult], Tuple[List[LPRResult], np.ndarray]]:
        """
        Process a single image.
        
        Args:
            image: Input image (path or numpy array)
            return_debug_image: Return annotated image
            padding_percent: Padding around detected plates
            
        Returns:
            List of LPRResult objects, optionally with debug image
        """
        start_time = time.time()
        
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not read image: {image}")
            original_img = img.copy()
        else:
            img = image.copy()
            original_img = img.copy()
        
        detections = self.detector.detect(
            img, 
            return_cropped=False,
            padding_percent=padding_percent,
        )
        
        results = []
        
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cropped = img[y1:y2, x1:x2]
            
            if cropped.size == 0:
                continue
            
            preprocessed = preprocess_for_recognition(cropped)
            
            rec_results = self.recognizer.recognize(preprocessed)
            
            if not rec_results:
                plate_text = ""
                rec_conf = 0.0
            else:
                raw_text = rec_results[0].text
                plate_text = normalize_vietnamese_plate(raw_text)
                rec_conf = rec_results[0].confidence
            
            detection_confidence = det.confidence
            combined_confidence = detection_confidence * rec_conf
            
            plate_type = self._classify_plate_type(plate_text)
            
            result = LPRResult(
                plate=plate_text,
                confidence=combined_confidence,
                detection_confidence=detection_confidence,
                recognition_confidence=rec_conf,
                bbox=det.bbox,
                plate_type=plate_type,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
            
            results.append(result)
        
        results = [r for r in results if r.confidence >= self.min_confidence]
        results.sort(key=lambda r: r.confidence, reverse=True)
        
        if return_debug_image:
            debug_img = self._draw_debug_image(original_img, results)
            return results, debug_img
        
        return results
    
    def process_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        show_progress: bool = True,
        frame_skip: int = 0,
    ) -> List[LPRFrameResult]:
        """
        Process a video file.
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save annotated video
            show_progress: Show progress bar
            frame_skip: Process every N frames (0 = all)
            
        Returns:
            List of frame results
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_results = []
        frame_id = 0
        
        pbar = None
        if show_progress:
            from tqdm import tqdm
            pbar = tqdm(total=total_frames, desc="Processing video")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_skip > 0 and frame_id % (frame_skip + 1) != 0:
                frame_id += 1
                if pbar:
                    pbar.update(1)
                continue
            
            timestamp = frame_id / fps
            
            start_time = time.time()
            results = self.process_image(frame)
            processing_time = (time.time() - start_time) * 1000
            
            frame_result = LPRFrameResult(
                frame_id=frame_id,
                timestamp=timestamp,
                results=results,
                total_processing_time_ms=processing_time,
            )
            frame_results.append(frame_result)
            
            if writer or show_progress:
                debug_img = self._draw_debug_image(frame, results)
                if writer:
                    writer.write(debug_img)
            
            frame_id += 1
            if pbar:
                pbar.update(1)
        
        cap.release()
        if writer:
            writer.release()
        if pbar:
            pbar.close()
        
        return frame_results
    
    def process_camera(
        self,
        camera_id: int = 0,
        window_name: str = "Vietnamese LPR",
        buffer_size: int = 30,
    ):
        """
        Process camera stream in real-time.
        
        Args:
            camera_id: Camera device ID
            window_name: Display window name
            buffer_size: Frame buffer size
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        print(f"[VietnameseLPR] Press 'q' to quit, 's' to save frame")
        
        frame_buffer = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.process_image(frame)
            debug_img = self._draw_debug_image(frame, results)
            
            cv2.imshow(window_name, debug_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = f"captured_{timestamp}.jpg"
                cv2.imwrite(save_path, debug_img)
                print(f"[VietnameseLPR] Saved: {save_path}")
            
            frame_buffer.append((frame, results))
            if len(frame_buffer) > buffer_size:
                frame_buffer.pop(0)
        
        cap.release()
        cv2.destroyAllWindows()
        
        return frame_buffer
    
    def _classify_plate_type(self, plate_text: str) -> Optional[str]:
        """Classify plate type based on text format"""
        if validate_plate_format(plate_text, 'private_car'):
            return 'private_car'
        elif validate_plate_format(plate_text, 'motorcycle'):
            return 'motorcycle'
        elif validate_plate_format(plate_text, 'police'):
            return 'police'
        elif validate_plate_format(plate_text, 'army'):
            return 'army'
        return None
    
    def _draw_debug_image(
        self,
        image: np.ndarray,
        results: List[LPRResult],
    ) -> np.ndarray:
        """Draw detection and recognition results on image"""
        img = image.copy()
        
        for result in results:
            x1, y1, x2, y2 = [int(v) for v in result.bbox]
            
            color = (0, 255, 0) if result.confidence > 0.8 else (0, 255, 255)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            label = f"{result.plate} ({result.confidence:.2f})"
            if result.plate_type:
                label += f" [{result.plate_type}]"
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                img,
                (x1, y1 - label_size[1] - 8),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            cv2.putText(
                img,
                label,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        h, w = img.shape[:2]
        
        status_y = h - 30
        cv2.putText(
            img,
            f"Plates: {len(results)} | Device: {self.device}",
            (10, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        return img


def create_pipeline(**kwargs) -> VietnameseLPRPipeline:
    """Factory function to create LPR pipeline"""
    return VietnameseLPRPipeline(**kwargs)
