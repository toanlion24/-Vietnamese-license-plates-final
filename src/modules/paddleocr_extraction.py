"""
Module 6: PaddleOCR Extraction
Character recognition for license plates
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

try:
    from paddleocr import PaddleOCR
except ImportError:
    logger.warning("PaddleOCR not installed. Using fallback recognizer.")


@dataclass
class OCRResult:
    """Single OCR recognition result"""
    text: str
    confidence: float
    bbox: List[List[float]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    char_confidences: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        self.text = self.text.strip()
    
    @property
    def char_count(self) -> int:
        return len(self.text)
    
    @property
    def avg_char_confidence(self) -> float:
        if self.char_confidences:
            return np.mean(self.char_confidences)
        return self.confidence


@dataclass
class PlateOCRResult:
    """Complete OCR result for a plate"""
    results: List[OCRResult]
    full_text: str = ""
    avg_confidence: float = 0.0
    char_confidence: float = 0.0
    
    def __post_init__(self):
        if self.results:
            self.full_text = "".join(r.text for r in self.results)
            self.avg_confidence = np.mean([r.confidence for r in self.results])
            
            all_chars = []
            for r in self.results:
                all_chars.extend(r.char_confidences if r.char_confidences else [r.confidence])
            if all_chars:
                self.char_confidence = np.mean(all_chars)
    
    @property
    def has_result(self) -> bool:
        return len(self.results) > 0 and len(self.full_text) > 0


class PaddleOCRRecognizer:
    """
    PaddleOCR-based character recognition for license plates.
    
    Features:
    - Multi-language support (Vietnamese, English)
    - Angle classification
    - Character-level confidence
    - Batch processing
    """
    
    def __init__(
        self,
        lang: str = "vi",
        use_angle_cls: bool = True,
        use_gpu: bool = True,
        config_path: Optional[str] = None,
        model_path: Optional[str] = None,
        dictionary_path: Optional[str] = None,
        show_log: bool = False,
    ):
        """
        Initialize PaddleOCR recognizer.
        
        Args:
            lang: Language ('vi', 'en', 'ch', 'mixed')
            use_angle_cls: Use angle classification
            use_gpu: Use GPU acceleration
            config_path: Custom config path
            model_path: Custom model path
            dictionary_path: Custom character dictionary
            show_log: Show PaddleOCR logs
        """
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self.config_path = config_path
        self.model_path = model_path
        self.dictionary_path = dictionary_path
        self.show_log = show_log
        
        self._ocr = None
        self._init_ocr()
    
    def _init_ocr(self):
        """Initialize PaddleOCR engine"""
        try:
            # PaddleOCR 2.10 API
            self._ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=self.show_log,
                rec_model_dir=self.model_path,
                rec_char_dict_path=self.dictionary_path,
            )
            logger.info(f"PaddleOCR initialized (lang={self.lang}, gpu={self.use_gpu})")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            self._ocr = None
    
    def recognize(
        self,
        image: np.ndarray,
        return_confidence: bool = True,
        min_confidence: float = 0.0,
    ) -> PlateOCRResult:
        """
        Recognize text from license plate image.
        
        Args:
            image: Input plate image
            return_confidence: Include confidence scores
            min_confidence: Minimum confidence threshold
            
        Returns:
            PlateOCRResult with recognized text
        """
        if self._ocr is None:
            logger.warning("PaddleOCR not available, returning empty result")
            return PlateOCRResult(results=[])
        
        img = self._prepare_image(image)
        
        try:
            # PaddleOCR 2.10 returns: [[line1, line2, ...]]
            # Each line: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ('text', confidence)]
            results = self._ocr.ocr(img, cls=self.use_angle_cls)
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return PlateOCRResult(results=[])
        
        if not results or not results[0]:
            return PlateOCRResult(results=[])
        
        ocr_results = []
        
        for line in results[0]:
            if line is None:
                continue
            bbox = line[0]
            text = line[1][0] if isinstance(line[1], tuple) else str(line[1])
            confidence = float(line[1][1]) if isinstance(line[1], tuple) else 1.0
            
            if confidence < min_confidence:
                continue
            
            ocr_results.append(OCRResult(
                text=text,
                confidence=confidence,
                bbox=bbox,
                char_confidences=[confidence] * len(text)
            ))
        
        ocr_results.sort(key=lambda r: r.confidence, reverse=True)
        
        return PlateOCRResult(results=ocr_results)
    
    def recognize_single(
        self,
        image: np.ndarray,
    ) -> Tuple[str, float]:
        """
        Recognize single line of text.
        
        Args:
            image: Input plate image
            
        Returns:
            Tuple of (recognized_text, confidence)
        """
        result = self.recognize(image)
        
        if not result.has_result:
            return "", 0.0
        
        if len(result.results) == 1:
            return result.results[0].text, result.results[0].confidence
        
        return result.full_text, result.avg_confidence
    
    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Prepare image for OCR"""
        img = image.copy()
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        h, w = img.shape[:2]
        if w < 50 or h < 10:
            img = cv2.resize(img, (max(w, 300), max(h, 60)))
        
        return img
    
    def recognize_batch(
        self,
        images: List[np.ndarray],
        **kwargs
    ) -> List[PlateOCRResult]:
        """
        Recognize multiple images.
        
        Args:
            images: List of plate images
            
        Returns:
            List of PlateOCRResult
        """
        results = []
        for img in images:
            results.append(self.recognize(img, **kwargs))
        return results


class FallbackRecognizer:
    """
    Fallback recognizer when PaddleOCR is not available.
    Uses template matching for demo purposes.
    """
    
    def __init__(self, **kwargs):
        self.characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
        self._template_cache = {}
    
    def recognize(
        self,
        image: np.ndarray,
        **kwargs
    ) -> PlateOCRResult:
        """Fallback recognition - returns empty result"""
        return PlateOCRResult(results=[])


class CharacterExtractor:
    """
    Extracts individual characters from plate for analysis.
    """
    
    def __init__(
        self,
        min_char_width: int = 10,
        min_char_height: int = 20,
        max_char_width: int = 100,
    ):
        self.min_char_width = min_char_width
        self.min_char_height = min_char_height
        self.max_char_width = max_char_width
    
    def extract(
        self,
        image: np.ndarray,
    ) -> List[Tuple[np.ndarray, List[float]]]:
        """
        Extract individual character images.
        
        Args:
            image: Plate image
            
        Returns:
            List of (character_image, bounding_box) tuples
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((2, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        chars = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if (w < self.min_char_width or h < self.min_char_height or
                w > self.max_char_width):
                continue
            
            char_img = binary[y:y+h, x:x+w]
            chars.append((char_img, [x, y, x+w, y+h]))
        
        chars.sort(key=lambda c: c[1][0])
        
        return chars


def preprocess_for_ocr(
    image: np.ndarray,
    target_height: int = 48,
    target_width: int = 320,
    enhance: bool = True,
) -> np.ndarray:
    """
    Preprocess image specifically for OCR.
    
    Args:
        image: Input plate image
        target_height: Target height
        target_width: Target width
        enhance: Apply enhancement
        
    Returns:
        Preprocessed image ready for OCR
    """
    from .image_enhancement import ImageEnhancer
    
    img = image.copy()
    
    if enhance:
        enhancer = ImageEnhancer()
        img = enhancer.enhance_for_ocr(img)
    else:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    h, w = img.shape[:2]
    
    if h == 0 or w == 0:
        return np.zeros((target_height, target_width), dtype=np.uint8)
    
    aspect = w / h
    target_aspect = target_width / target_height
    
    if aspect > target_aspect:
        new_w = target_width
        new_h = int(target_width / aspect)
    else:
        new_h = target_height
        new_w = int(target_height * aspect)
    
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    canvas = np.ones((target_height, target_width), dtype=np.uint8) * 255
    
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img
    
    return canvas


def visualize_ocr_results(
    image: np.ndarray,
    result: PlateOCRResult,
    show_chars: bool = True,
) -> np.ndarray:
    """
    Visualize OCR results on image.
    
    Args:
        image: Input image
        result: PlateOCRResult
        show_chars: Show character bounding boxes
        
    Returns:
        Image with visualizations
    """
    img = image.copy()
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if show_chars:
        for ocr_result in result.results:
            bbox = np.array(ocr_result.bbox, dtype=np.int32)
            cv2.polylines(img, [bbox], True, (0, 255, 0), 2)
            
            for i, point in enumerate(bbox):
                cv2.circle(img, tuple(point), 3, (0, 0, 255), -1)
    
    text_y = h - 20 if h > 100 else h - 5
    
    label = result.full_text if result.has_result else "[No text detected]"
    conf_label = f" ({result.avg_confidence:.2f})" if result.has_result else ""
    
    label_size = cv2.getTextSize(label, font, 0.7, 2)[0]
    cv2.rectangle(
        img,
        (5, text_y - label_size[1] - 5),
        (10 + label_size[0], text_y + 5),
        (0, 0, 0),
        -1
    )
    
    cv2.putText(
        img,
        label + conf_label,
        (10, text_y),
        font,
        0.7,
        (0, 255, 0),
        2
    )
    
    return img
