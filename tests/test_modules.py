"""
Unit tests for Vietnamese LPR
"""

import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection import PlateDetector, DetectionResult, visualize_detections
from src.recognition import (
    PlateRecognizer,
    preprocess_for_recognition,
    normalize_vietnamese_plate,
    validate_plate_format,
)
from src.utils import apply_clahe, calculate_iou


class TestDetection:
    """Test detection module"""
    
    def test_detection_result_bbox(self):
        """Test DetectionResult bbox properties"""
        bbox = [100, 100, 200, 200]
        result = DetectionResult(bbox=bbox, confidence=0.95)
        
        assert result.bbox == bbox
        assert result.xywh == [150, 150, 100, 100]
        assert result.area == 10000
    
    def test_iou_calculation(self):
        """Test IoU calculation"""
        box1 = [0, 0, 100, 100]
        box2 = [50, 50, 150, 150]
        
        iou = calculate_iou(box1, box2)
        
        assert 0 < iou < 1
        assert iou > 0


class TestRecognition:
    """Test recognition module"""
    
    def test_normalize_plate(self):
        """Test plate text normalization"""
        assert normalize_vietnamese_plate("30A-1234.56") == "30A-1234.56"
        assert normalize_vietnamese_plate("30a-1234.56") == "30A-1234.56"
        assert normalize_vietnamese_plate("O0A-1234.56") == "00A-1234.56"
    
    def test_validate_plate_formats(self):
        """Test plate format validation"""
        assert validate_plate_format("30A-1234.56", "private_car") == True
        assert validate_plate_format("43-12345", "motorcycle") == True
        assert validate_plate_format("60-1234-56", "police") == True
        assert validate_plate_format("123456-78", "army") == True
        
        assert validate_plate_format("invalid", "private_car") == False
        assert validate_plate_format("ABC-1234.56", "private_car") == False


class TestPreprocessing:
    """Test preprocessing functions"""
    
    def test_clahe(self):
        """Test CLAHE enhancement"""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        enhanced = apply_clahe(img)
        
        assert enhanced.shape == img.shape
        assert enhanced.dtype == np.uint8
    
    def test_recognition_preprocessing(self):
        """Test OCR preprocessing"""
        img = np.random.randint(0, 255, (100, 400, 3), dtype=np.uint8)
        processed = preprocess_for_recognition(img, target_height=48, target_width=320)
        
        assert processed.shape == (48, 320, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
