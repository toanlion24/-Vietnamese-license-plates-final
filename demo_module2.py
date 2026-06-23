"""
Module 2: YOLOv11 Detection - Quick Demo
Input: RGB Frame
Output: BoundingBox (xe, biển số)
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import importlib.util

# Setup path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import detector
spec = importlib.util.spec_from_file_location('detector', 'src/detection/detector.py')
detector_module = importlib.util.spec_from_file_location(spec.name, spec.origin)
detector_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detector_module)

PlateDetector = detector_module.PlateDetector
visualize_detections = detector_module.visualize_detections


def demo_single_image(image_path="LicensePlateDetectionDataset/images/test/boderngoaigiao1.jpg"):
    """Demo detection on single image"""
    print("=" * 60)
    print("Module 2: YOLOv11 Detection")
    print("=" * 60)
    print(f"Input: {image_path}")
    print()
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return
    
    print(f"Image shape: {img.shape}")
    print()
    
    # Initialize detector
    print("[1/3] Loading model...")
    detector = PlateDetector(
        model_path="weights/best.pt",
        confidence_threshold=0.25,
        device="cuda"
    )
    print("[OK] Model loaded")
    print()
    
    # Detect
    print("[2/3] Running detection...")
    detections = detector.detect(img, return_cropped=True)
    print(f"[OK] Found {len(detections)} plate(s)")
    print()
    
    # Show results
    print("[3/3] Results:")
    print("-" * 60)
    print(f"{'#':<3} {'x1':<8} {'y1':<8} {'x2':<8} {'y2':<8} {'Conf':<10} {'Size':<15}")
    print("-" * 60)
    
    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det.bbox
        w = x2 - x1
        h = y2 - y1
        print(f"{i:<3} {x1:<8.1f} {y1:<8.1f} {x2:<8.1f} {y2:<8.1f} {det.confidence:<10.2%} {int(w)}x{int(h)}px")
    
    print("-" * 60)
    print()
    
    # Visualize
    result_img = visualize_detections(img, detections)
    
    # Save result
    output_path = "outputs/module2_detection_result.jpg"
    cv2.imwrite(output_path, result_img)
    print(f"[OK] Visualization saved to: {output_path}")
    
    # Also save cropped plates
    for i, det in enumerate(detections, 1):
        if hasattr(det, 'cropped_image') and det.cropped_image is not None:
            crop_path = f"outputs/module2_plate_{i}.jpg"
            cv2.imwrite(crop_path, det.cropped_image)
            print(f"[OK] Cropped plate {i} saved to: {crop_path}")
    
    return detections


def demo_video_frames():
    """Demo detection on video frames"""
    print("=" * 60)
    print("Module 2: Video Stream Detection")
    print("=" * 60)
    
    video_path = "test.mp4"  # Change to your video path
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return
    
    detector = PlateDetector(model_path="weights/best.pt", device="cuda")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        detections = detector.detect(frame)
        
        if detections:
            print(f"Frame {frame_count}: {len(detections)} plate(s)")
    
    cap.release()
    print(f"\nTotal frames: {frame_count}")


def demo_with_input_stream():
    """Demo using input_stream module"""
    print("=" * 60)
    print("Module 2: Integration with InputStream")
    print("=" * 60)
    
    # Import input stream
    spec = importlib.util.spec_from_file_location('input_stream', 'src/modules/input_stream.py')
    input_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(input_module)
    
    InputStream = input_module.InputStream
    PlateDetector = detector_module.PlateDetector
    
    # Create stream
    stream = InputStream("test.jpg")
    detector = PlateDetector(model_path="weights/best.pt", device="cuda")
    
    # Process frames
    for frame in stream:
        detections = detector.detect(frame.image)
        print(f"Frame {frame.frame_id}: {len(detections)} detection(s)")
    
    print("\n[OK] Demo complete!")


def create_test_image():
    """Create a test image with boxes for visualization"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (100, 100, 100)  # Gray background
    
    # Draw a simulated car
    cv2.rectangle(img, (100, 150), (540, 400), (50, 50, 200), -1)
    cv2.putText(img, "CAR", (250, 280), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Draw a simulated plate
    cv2.rectangle(img, (200, 320), (440, 370), (200, 200, 50), -1)
    cv2.putText(img, "30A-1234", (210, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    cv2.imwrite("outputs/test_vehicle.jpg", img)
    print("[OK] Test image created: outputs/test_vehicle.jpg")
    return img


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    
    print()
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "MODULE 2: YOLOv11 DETECTION" + " " * 19 + "║")
    print("║" + " " * 15 + "Input: RGB Frame" + " " * 29 + "║")
    print("║" + " " * 15 + "Output: BoundingBox" + " " * 26 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # Check if test image exists (prioritize dataset image)
    test_image = "LicensePlateDetectionDataset/images/test/boderngoaigiao1.jpg"
    if Path(test_image).exists():
        print(f"Using test image: {test_image}")
        demo_single_image(test_image)
    elif Path("outputs/boderngoaigiao1_20260621_220652/detected.jpg").exists():
        test_image = "outputs/boderngoaigiao1_20260621_220652/detected.jpg"
        print(f"Using test image: {test_image}")
        demo_single_image(test_image)
    else:
        print("[INFO] No test image found. Creating demo image...")
        create_test_image()
        print("\n[INFO] Please add a real image to test with.")
