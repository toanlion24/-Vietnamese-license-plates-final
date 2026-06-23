"""
OCR Demo Script
==============
Demo script for PaddleOCR text recognition on Vietnamese license plates.

Usage:
    python scripts/ocr_demo.py
    python scripts/ocr_demo.py --image path/to/image.jpg
    python scripts/ocr_demo.py --plate "80-NG-63"
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix torch import order issue on Windows
import torch  # noqa: F401

def test_ocr_with_image(image_path: str = None):
    """Test OCR with an actual license plate image."""
    from paddleocr import PaddleOCR
    from src.modules.image_enhancement import ImageEnhancer
    from src.modules.rule_engine import PlateValidator, validate_vietnamese_plate
    
    # Initialize OCR
    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='vi',
        use_gpu=False,
        show_log=False
    )
    
    # Initialize enhancer
    enhancer = ImageEnhancer()
    
    # Initialize validator
    validator = PlateValidator()
    
    # Load image
    if image_path is None:
        image_path = "outputs/boderngoaigiao1_20260621_220652/original.jpg"
    
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Image shape: {img.shape}")
    
    # Crop plate region from bbox in results.json
    # bbox: [255.11, 170.49, 335.80, 206.01]
    x1, y1, x2, y2 = 255, 170, 336, 206
    plate = img[y1:y2, x1:x2]
    print(f"Plate crop shape: {plate.shape}")
    
    # Enhance image for better OCR
    print("Enhancing image...")
    plate_enhanced = enhancer.enhance_for_ocr(plate)
    
    # Resize for better OCR (small images are hard to OCR)
    scale_factor = 4
    plate_resized = cv2.resize(
        plate_enhanced,
        (plate_enhanced.shape[1] * scale_factor, plate_enhanced.shape[0] * scale_factor),
        interpolation=cv2.INTER_CUBIC
    )
    
    # Convert back to BGR if grayscale
    if len(plate_resized.shape) == 2:
        plate_resized = cv2.cvtColor(plate_resized, cv2.COLOR_GRAY2BGR)
    
    print(f"Enhanced plate shape: {plate_resized.shape}")
    
    # Run OCR
    print("Running OCR...")
    result = ocr.ocr(plate_resized)
    
    print("\n" + "=" * 50)
    print("OCR RESULTS")
    print("=" * 50)
    
    if result and result[0]:
        for line in result[0]:
            if line:
                bbox = line[0]
                text = line[1][0]
                conf = line[1][1]
                
                print(f"Detected Text: {text}")
                print(f"Confidence: {conf:.3f} ({conf*100:.1f}%)")
                
                # Validate plate format
                validation = validator.validate(text)
                print(f"\nValidation:")
                print(f"  Is Valid: {validation.is_valid}")
                print(f"  Plate Type: {validation.plate_type.value}")
                print(f"  Normalized: {validation.normalized_text}")
                
                if validation.errors:
                    print(f"  Errors: {validation.errors}")
                if validation.warnings:
                    print(f"  Warnings: {validation.warnings}")
                
                # Get province
                province = validator.get_province(validation.normalized_text)
                if province:
                    print(f"  Province: {province}")
    
    print("=" * 50)
    
    return result


def test_ocr_with_text():
    """Test plate validation with sample texts."""
    from src.modules.rule_engine import PlateValidator
    
    print("\n" + "=" * 50)
    print("PLATE VALIDATION TESTS")
    print("=" * 50)
    
    validator = PlateValidator()
    
    test_plates = [
        # Valid plates
        "30A-1234.56",    # Private car
        "43-12345",       # Motorcycle
        "80-NG-63",       # Motorcycle (detected format)
        "60-1234-56",     # Police
        "123456-78",      # Army
        
        # Invalid plates
        "ABC-1234.56",    # Invalid (letter in wrong position)
        "12-ABCDE",       # Invalid (letters in motorcycle)
        "12345",          # Too short
        "ABCD-EFGH-IJKL", # Wrong format
    ]
    
    for plate in test_plates:
        result = validator.validate(plate)
        status = "✓" if result.is_valid else "✗"
        print(f"{status} {plate:20s} -> {result.normalized_text:15s} ({result.plate_type.value})")
    
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="OCR Demo for Vietnamese License Plates")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to image file")
    parser.add_argument("--text", "-t", action="store_true", help="Run plate validation tests")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Vietnamese License Plate OCR Demo")
    print("=" * 60)
    
    if args.text:
        test_ocr_with_text()
    else:
        test_ocr_with_image(args.image)


if __name__ == "__main__":
    main()
