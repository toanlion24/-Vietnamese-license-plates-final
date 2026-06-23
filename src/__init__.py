"""
Vietnamese LPR - YOLOv11 + PaddleOCR Pipeline
Main package initialization
"""

__version__ = "1.0.0"
__author__ = "Vietnamese LPR Team"

from .detection import PlateDetector
from .recognition import PlateRecognizer
from .pipeline import VietnameseLPRPipeline

__all__ = [
    "PlateDetector",
    "PlateRecognizer",
    "VietnameseLPRPipeline",
]
