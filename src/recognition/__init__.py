"""
Recognition Module - PaddleOCR Character Recognition
"""

from .recognizer import (
    PlateRecognizer,
    RecognitionResult,
    preprocess_for_recognition,
    normalize_vietnamese_plate,
    validate_plate_format,
)

__all__ = [
    "PlateRecognizer",
    "RecognitionResult",
    "preprocess_for_recognition",
    "normalize_vietnamese_plate",
    "validate_plate_format",
]
