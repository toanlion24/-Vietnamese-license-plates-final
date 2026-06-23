"""
Vietnamese LPR Pipeline - End-to-End License Plate Recognition
"""

from .inference import (
    VietnameseLPRPipeline,
    LPRResult,
    LPRFrameResult,
    create_pipeline,
)

__all__ = [
    "VietnameseLPRPipeline",
    "LPRResult",
    "LPRFrameResult",
    "create_pipeline",
]
