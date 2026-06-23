"""
Recognition Module - PaddleOCR Character Recognition
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

try:
    from paddleocr import PaddleOCR
except ImportError:
    raise ImportError("Please install paddleocr: pip install paddleocr")


@dataclass
class RecognitionResult:
    """Single recognition result"""
    text: str
    confidence: float
    bbox: List[List[float]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    
    def __post_init__(self):
        """Clean text"""
        self.text = self.text.strip()
    
    @property
    def char_confidences(self) -> List[float]:
        """Per-character confidence (if available)"""
        return [self.confidence] * len(self.text)


class PlateRecognizer:
    """
    PaddleOCR-based License Plate Character Recognizer
    
    Handles text recognition in cropped license plate images.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_path: Optional[str] = None,
        dictionary_path: str = "configs/vietnamese_dict.txt",
        use_angle_cls: bool = True,
        use_gpu: bool = True,
        lang: str = "vi"
    ):
        """
        Initialize the plate recognizer.
        
        Args:
            config_path: Path to PaddleOCR config
            model_path: Path to custom recognition model
            dictionary_path: Path to Vietnamese character dictionary
            use_angle_cls: Use angle classification for rotated text
            use_gpu: Use GPU acceleration
            lang: Language ('vi', 'en', 'ch')
        """
        self.config_path = config_path
        self.model_path = model_path
        self.dictionary_path = dictionary_path
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self.lang = lang
        
        self._load_model()
    
    def _load_model(self):
        """Initialize PaddleOCR engine"""
        self.ocr = PaddleOCR(
            use_angle_cls=self.use_angle_cls,
            lang=self.lang,
            use_gpu=self.use_gpu,
            show_log=False,
            rec_model_dir=self.model_path,
            rec_char_dict_path=self.dictionary_path,
        )
    
    def recognize(
        self, 
        image: Union[np.ndarray, str, Path],
        return_confidence: bool = True
    ) -> List[RecognitionResult]:
        """
        Recognize text from a license plate image.
        
        Args:
            image: Cropped plate image (numpy array) or image path
            return_confidence: Include confidence scores
            
        Returns:
            List of RecognitionResult objects
        """
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not read image: {image}")
        else:
            img = image.copy()
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        results = self.ocr.ocr(img, cls=self.use_angle_cls)
        
        recognitions = []
        
        if results and results[0]:
            for line in results[0]:
                bbox = line[0]
                text = line[1][0]
                confidence = float(line[1][1])
                
                recognitions.append(RecognitionResult(
                    text=text,
                    confidence=confidence,
                    bbox=bbox
                ))
        
        recognitions.sort(key=lambda r: r.confidence, reverse=True)
        
        return recognitions
    
    def recognize_plate(
        self,
        image: np.ndarray,
        plate_type: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Recognize full plate text with post-processing.
        
        Args:
            image: Cropped plate image
            plate_type: Type of plate ('car', 'motorcycle', 'police', 'army')
            
        Returns:
            Tuple of (recognized_text, combined_confidence)
        """
        results = self.recognize(image)
        
        if not results:
            return "", 0.0
        
        if len(results) == 1:
            return results[0].text, results[0].confidence
        
        combined_text = "".join([r.text for r in results])
        avg_confidence = np.mean([r.confidence for r in results])
        
        return combined_text, avg_confidence


def preprocess_for_recognition(
    image: np.ndarray,
    target_height: int = 48,
    target_width: int = 320,
    enhance: bool = True
) -> np.ndarray:
    """
    Preprocess plate image for better OCR recognition.
    
    Args:
        image: Input plate image
        target_height: Target height for model input
        target_width: Target width for model input
        enhance: Apply CLAHE enhancement
        
    Returns:
        Preprocessed image
    """
    img = image.copy()
    
    if enhance:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        if len(img.shape) == 3:
            img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            img = enhanced
    
    h, w = img.shape[:2]
    aspect_ratio = w / h
    
    if aspect_ratio > target_width / target_height:
        new_w = target_width
        new_h = int(target_width / aspect_ratio)
    else:
        new_h = target_height
        new_w = int(target_height * aspect_ratio)
    
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
    
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img
    
    return canvas


def normalize_vietnamese_plate(text: str) -> str:
    """
    Normalize Vietnamese license plate text.
    
    Args:
        text: Raw OCR text
        
    Returns:
        Normalized plate string
    """
    replacements = {
        'O': '0',
        'D': '0',
        'I': '1',
        'l': '1',
        'S': '5',
        'B': '8',
    }
    
    result = text.upper()
    
    for old, new in replacements.items():
        result = result.replace(old, new)
    
    result = ''.join(c for c in result if c.isalnum() or c in '- .')
    
    return result


def validate_plate_format(text: str, plate_type: Optional[str] = None) -> bool:
    """
    Validate Vietnamese plate format.
    
    Args:
        text: Plate text
        plate_type: Optional plate type hint
        
    Returns:
        True if format is valid
    """
    import re
    
    patterns = {
        'private_car': r'^\d{2}[A-Z]-\d{4}\.\d{2}$',
        'motorcycle': r'^\d{2}-\d{5}(\.\d{2})?$',
        'police': r'^\d{2}-\d{4}-\d{2}$',
        'army': r'^\d{6}-\d{2}$',
    }
    
    if plate_type and plate_type in patterns:
        return bool(re.match(patterns[plate_type], text))
    
    for pattern in patterns.values():
        if re.match(pattern, text):
            return True
    
    return False
