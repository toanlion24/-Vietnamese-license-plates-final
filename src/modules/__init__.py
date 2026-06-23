"""
Vietnamese LPR - Modules Package
================================

Mục lục các module trong pipeline nhận diện biển số xe:

┌─────────────────────────────────────────────────────────────────────────────┐
│                            PIPELINE OVERVIEW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  MODULE 1: Input Stream          →  src/modules/input_stream.py            │
│  MODULE 2: YOLOv11 Detection     →  src/modules/yolo_interactive.py        │
│  MODULE 3: Vehicle-Plate Link    →  src/modules/vehicle_plate_association.py│
│  MODULE 4: Perspective Correct   →  src/modules/rectify_perspective.py    │
│  MODULE 5: Image Enhancement     →  src/modules/image_enhancement.py       │
│  MODULE 6: OCR Extraction        →  src/modules/paddleocr_extraction.py    │
│  MODULE 7: Rule Engine           →  src/modules/rule_engine.py             │
│  MODULE 8: Tracking & Voting     →  src/modules/bytetrack_voting.py       │
│  MODULE 9: Database              →  src/modules/database_manager.py        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Data Flow:
  Image → Detect (bbox) → Associate → Rectify → Enhance → OCR → Validate → Track → Store
"""

# =============================================================================
# MODULE 1: INPUT STREAM
# =============================================================================
from .input_stream import (
    InputStream,
    WebcamStream,
    Frame,
    load_image,
    load_video_info,
    list_cameras,
)

# =============================================================================
# MODULE 2: YOLOv11 DETECTION
# =============================================================================
from .yolo_interactive import (
    YOLOInteractiveDetector,
    quick_detect,
    DetectionMode,
    InputSource,
    BoundingBox,
    DetectionResult,
)
from .yolo_detection import (
    YOLODetector,  # Legacy detector
)

# =============================================================================
# MODULE 3: VEHICLE-PLATE ASSOCIATION
# =============================================================================
from .vehicle_plate_association import (
    VehiclePlateAssociator,
    VehiclePlatePair,
)

# =============================================================================
# MODULE 4: RECTIFY & PERSPECTIVE
# =============================================================================
from .rectify_perspective import (
    PlateRectifier,
    RectificationResult,
)

# =============================================================================
# MODULE 5: IMAGE ENHANCEMENT
# =============================================================================
from .image_enhancement import (
    ImageEnhancer,
    EnhancementConfig,
)

# =============================================================================
# MODULE 6: PADDLEOCR EXTRACTION
# =============================================================================
from .paddleocr_extraction import (
    PaddleOCRRecognizer,
    PlateOCRResult,
)

# =============================================================================
# MODULE 7: RULE ENGINE & REGEX
# =============================================================================
from .rule_engine import (
    PlateValidator,
    PlateRuleEngine,
    PlateType,
)

# =============================================================================
# MODULE 8: BYTETRACK & VOTING
# =============================================================================
from .bytetrack_voting import (
    ByteTracker,
    VotingSystem,
    Tracklet,
)

# =============================================================================
# MODULE 9: DATABASE
# =============================================================================
from .database_manager import (
    DatabaseManager,
    RecognitionRecord,
)

# =============================================================================
# EXPORTS
# =============================================================================
__all__ = [
    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 1: INPUT STREAM
    # ─────────────────────────────────────────────────────────────────────────
    "InputStream",
    "WebcamStream",
    "Frame",
    "load_image",
    "load_video_info",
    "list_cameras",
    
    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 2: YOLOv11 DETECTION (Interactive - Recommended)
    # ─────────────────────────────────────────────────────────────────────────
    "YOLOInteractiveDetector",  # ✅ Recommended
    "quick_detect",
    "DetectionMode",
    "InputSource",
    # ─── Legacy ───
    "YOLODetector",
    
    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 3: VEHICLE-PLATE ASSOCIATION
    # ─────────────────────────────────────────────────────────────────────────
    "VehiclePlateAssociator",
    "VehiclePlatePair",
    
    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 4: RECTIFY & PERSPECTIVE
    # ─────────────────────────────────────────────────────────────────────────
    "PlateRectifier",
    "RectificationResult",
    
    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 5: IMAGE ENHANCEMENT
    # ─────────────────────────────────────────────────────────────────────────
    "ImageEnhancer",
    "EnhancementConfig",
    
    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 6: PADDLEOCR
    # ─────────────────────────────────────────────────────────────────────────
    "PaddleOCRRecognizer",
    "PlateOCRResult",
    
    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 7: RULE ENGINE
    # ─────────────────────────────────────────────────────────────────────────
    "PlateValidator",
    "PlateRuleEngine",
    "PlateType",
    
    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 8: BYTETRACK & VOTING
    # ─────────────────────────────────────────────────────────────────────────
    "ByteTracker",
    "VotingSystem",
    "Tracklet",
    
    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 9: DATABASE
    # ─────────────────────────────────────────────────────────────────────────
    "DatabaseManager",
    "RecognitionRecord",
    
    # ─────────────────────────────────────────────────────────────────────────
    # SHARED CLASSES
    # ─────────────────────────────────────────────────────────────────────────
    "BoundingBox",
    "DetectionResult",
]

# =============================================================================
# MODULE INFO
# =============================================================================
MODULES = {
    1: {"name": "Input Stream", "file": "input_stream.py", "status": "ready"},
    2: {"name": "YOLOv11 Detection", "file": "yolo_interactive.py", "status": "ready"},
    3: {"name": "Vehicle-Plate Association", "file": "vehicle_plate_association.py", "status": "ready"},
    4: {"name": "Rectify & Perspective", "file": "rectify_perspective.py", "status": "ready"},
    5: {"name": "Image Enhancement", "file": "image_enhancement.py", "status": "ready"},
    6: {"name": "PaddleOCR Extraction", "file": "paddleocr_extraction.py", "status": "ready"},
    7: {"name": "Rule Engine", "file": "rule_engine.py", "status": "ready"},
    8: {"name": "ByteTrack & Voting", "file": "bytetrack_voting.py", "status": "ready"},
    9: {"name": "Database", "file": "database_manager.py", "status": "ready"},
}


def list_modules():
    """Liệt kê tất cả modules"""
    print("\n📦 Vietnamese LPR - Available Modules")
    print("=" * 60)
    for num, info in MODULES.items():
        status_icon = "✅" if info["status"] == "ready" else "⏳"
        print(f"  [{num}] {status_icon} {info['name']:<30} ({info['file']})")
    print("=" * 60)
    print(f"  Total: {len(MODULES)} modules\n")


if __name__ == "__main__":
    list_modules()
