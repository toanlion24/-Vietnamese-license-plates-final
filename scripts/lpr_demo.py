"""
End-to-End LPR Demo
====================
Complete demo combining YOLO detection + PaddleOCR recognition.

Usage:
    python scripts/lpr_demo.py
    python scripts/lpr_demo.py --image path/to/image.jpg
"""

import cv2
import numpy as np
import argparse
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix torch import order issue on Windows
import torch  # noqa: F401

from paddleocr import PaddleOCR
from src.modules.image_enhancement import ImageEnhancer
from src.modules.rule_engine import PlateValidator


class LPRDemo:
    """End-to-end License Plate Recognition Demo."""
    
    def __init__(self, use_gpu: bool = False):
        """Initialize the LPR demo."""
        print("Initializing LPR Demo...")
        
        # Initialize PaddleOCR
        print("  - Initializing PaddleOCR...")
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='vi',
            use_gpu=use_gpu,
            show_log=False
        )
        
        # Initialize enhancer
        print("  - Initializing Image Enhancer...")
        self.enhancer = ImageEnhancer()
        
        # Initialize validator
        print("  - Initializing Plate Validator...")
        self.validator = PlateValidator()
        
        print("LPR Demo initialized successfully!\n")
    
    def process_image(
        self,
        image_path: str,
        bbox: list = None,
        scale_factor: int = 4
    ) -> dict:
        """
        Process an image and recognize license plates.
        
        Args:
            image_path: Path to the image file
            bbox: Optional bounding box [x1, y1, x2, y2] to crop plate
            scale_factor: Scale factor for upscaling small plates
            
        Returns:
            Dictionary with recognition results
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        result = {
            'image_path': image_path,
            'image_shape': img.shape,
            'plates': []
        }
        
        # Use provided bbox or full image
        if bbox is None:
            # Default bbox from demo image
            bbox = [255, 170, 336, 206]
        
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Crop plate region
        plate = img[y1:y2, x1:x2]
        result['plate_shape'] = plate.shape
        
        # Enhance image
        plate_enhanced = self.enhancer.enhance_for_ocr(plate)
        
        # Resize for better OCR
        plate_resized = cv2.resize(
            plate_enhanced,
            (plate_enhanced.shape[1] * scale_factor, plate_enhanced.shape[0] * scale_factor),
            interpolation=cv2.INTER_CUBIC
        )
        
        if len(plate_resized.shape) == 2:
            plate_resized = cv2.cvtColor(plate_resized, cv2.COLOR_GRAY2BGR)
        
        result['plate_enhanced_shape'] = plate_resized.shape
        
        # Run OCR
        ocr_result = self.ocr.ocr(plate_resized)
        
        # Parse OCR results
        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                if line:
                    text = line[1][0]
                    conf = line[1][1]
                    
                    # Validate plate
                    validation = self.validator.validate(text)
                    
                    plate_info = {
                        'raw_text': text,
                        'confidence': conf,
                        'normalized_text': validation.normalized_text,
                        'is_valid': validation.is_valid,
                        'plate_type': validation.plate_type.value,
                    }
                    
                    if validation.is_valid:
                        province = self.validator.get_province(validation.normalized_text)
                        if province:
                            plate_info['province'] = province
                    
                    if validation.errors:
                        plate_info['errors'] = validation.errors
                    
                    result['plates'].append(plate_info)
        
        return result
    
    def visualize_result(
        self,
        image_path: str,
        result: dict,
        output_path: str = None
    ) -> np.ndarray:
        """Visualize the recognition result."""
        img = cv2.imread(image_path)
        
        bbox = [255, 170, 336, 206]
        x1, y1, x2, y2 = bbox
        
        # Draw plate bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if result['plates']:
            plate = result['plates'][0]
            text = plate['normalized_text']
            conf = plate['confidence']
            
            label = f"{text} ({conf:.2f})"
            if plate.get('province'):
                label += f" [{plate['province']}]"
            
            # Draw label
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                img,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        else:
            cv2.putText(
                img,
                "No plate detected",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )
        
        if output_path:
            cv2.imwrite(output_path, img)
            print(f"Result saved to: {output_path}")
        
        return img
    
    def print_result(self, result: dict):
        """Print the recognition result."""
        print("=" * 60)
        print("RECOGNITION RESULTS")
        print("=" * 60)
        print(f"Image: {result['image_path']}")
        print(f"Image Shape: {result['image_shape']}")
        print(f"Plate Crop Shape: {result['plate_shape']}")
        print("-" * 60)
        
        if result['plates']:
            for i, plate in enumerate(result['plates'], 1):
                print(f"\nPlate #{i}:")
                print(f"  Raw Text:     {plate['raw_text']}")
                print(f"  Confidence:   {plate['confidence']:.3f} ({plate['confidence']*100:.1f}%)")
                print(f"  Normalized:   {plate['normalized_text']}")
                print(f"  Valid:        {'Yes' if plate['is_valid'] else 'No'}")
                print(f"  Type:         {plate['plate_type']}")
                if 'province' in plate:
                    print(f"  Province:     {plate['province']}")
                if 'errors' in plate:
                    print(f"  Errors:       {plate['errors']}")
        else:
            print("\nNo plates detected!")
        
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="End-to-End LPR Demo")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to image file")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output image path")
    parser.add_argument("--scale", "-s", type=int, default=4, help="Scale factor for OCR")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Vietnamese License Plate Recognition Demo")
    print("=" * 60 + "\n")
    
    # Initialize LPR Demo
    lpr = LPRDemo(use_gpu=not args.no_gpu)
    
    # Default image path
    if args.image is None:
        args.image = "outputs/boderngoaigiao1_20260621_220652/original.jpg"
    
    # Process image
    print(f"Processing: {args.image}\n")
    result = lpr.process_image(args.image, scale_factor=args.scale)
    
    # Print results
    lpr.print_result(result)
    
    # Visualize and save
    output_path = args.output
    if output_path is None:
        output_path = "outputs/ocr_result.jpg"
    
    lpr.visualize_result(args.image, result, output_path)
    
    return result


if __name__ == "__main__":
    main()
