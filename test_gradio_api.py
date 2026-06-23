# -*- coding: utf-8 -*-
"""Test Gradio LPR Demo API"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from gradio_client import Client, handle_file


def main():
    print("=" * 60)
    print("Testing Gradio LPR Demo API")
    print("=" * 60)
    
    client = Client('http://localhost:7870/')
    
    # Test 1: Demo image 1 (full image)
    print("\n[Test 1] Process with demo image 1 (full)")
    print("Loading demo image 1...")
    result1 = client.predict(
        api_name='/load_demo1'
    )
    print(f"  -> Image loaded: {type(result1).__name__}")
    
    # Test 2: Demo image 2 (crop)
    print("\n[Test 2] Process with demo image 2 (cropped plate)")
    print("Loading demo image 2...")
    result2 = client.predict(
        api_name='/load_demo2'
    )
    print(f"  -> Image loaded: {type(result2).__name__}")
    
    # Test 3: Process with custom image
    print("\n[Test 3] Process with custom image file")
    image_path = 'outputs/boderngoaigiao1_20260621_220652/plate_01_crop.jpg'
    print(f"  Image: {image_path}")
    
    result3 = client.predict(
        image=handle_file(image_path),
        scale_factor=4,
        min_confidence=0.5,
        api_name='/process_image'
    )
    
    print(f"\n  Result:")
    print(f"    Total outputs: {len(result3)}")
    
    if len(result3) >= 3:
        annotated_img = result3[0]
        gallery = result3[1]
        results_text = result3[2]
        
        print(f"\n  [1] Annotated image: {type(annotated_img).__name__}")
        if isinstance(annotated_img, dict):
            print(f"      Path: {annotated_img.get('path', 'N/A')}")
        
        print(f"\n  [2] Plates gallery: {type(gallery).__name__}")
        if isinstance(gallery, list):
            for i, g in enumerate(gallery):
                print(f"      [{i}] {g.get('path', 'N/A') if isinstance(g, dict) else g}")
        
        print(f"\n  [3] Results text:")
        print(results_text)
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()