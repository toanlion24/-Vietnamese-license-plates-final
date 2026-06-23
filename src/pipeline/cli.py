"""
Command-line interface for Vietnamese LPR Pipeline
"""

import argparse
import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import VietnameseLPRPipeline, create_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Vietnamese License Plate Recognition - YOLOv11 + PaddleOCR"
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to input image'
    )
    
    parser.add_argument(
        '--video',
        type=str,
        help='Path to input video'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        help='Camera device ID (e.g., 0 for webcam)'
    )
    
    parser.add_argument(
        '--detector',
        type=str,
        default='models/yolov11/best.pt',
        help='Path to detection model'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.7,
        help='Minimum confidence threshold'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for annotated image/video'
    )
    
    parser.add_argument(
        '--save-crops',
        action='store_true',
        help='Save cropped plates'
    )
    
    args = parser.parse_args()
    
    if not any([args.image, args.video, args.camera is not None]):
        parser.print_help()
        print("\n\nExample usage:")
        print("  python -m src.pipeline.inference --image test.jpg")
        print("  python -m src.pipeline.inference --video traffic.mp4")
        print("  python -m src.pipeline.inference --camera 0")
        sys.exit(1)
    
    print("=" * 60)
    print("Vietnamese License Plate Recognition")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Confidence threshold: {args.conf}")
    print()
    
    try:
        pipeline = create_pipeline(
            detector_weights=args.detector,
            device=args.device,
            min_confidence=args.conf,
        )
        
        if args.image:
            print(f"\nProcessing image: {args.image}")
            
            results, debug_img = pipeline.process_image(
                args.image,
                return_debug_image=True
            )
            
            print(f"\nFound {len(results)} plate(s):")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.plate} (confidence: {result.confidence:.2f})")
                if result.plate_type:
                    print(f"     Type: {result.plate_type}")
            
            if args.output:
                cv2.imwrite(args.output, debug_img)
                print(f"\nSaved result to: {args.output}")
            else:
                output_path = Path(args.image).stem + "_result.jpg"
                cv2.imwrite(output_path, debug_img)
                print(f"\nSaved result to: {output_path}")
            
            if args.save_crops:
                for i, result in enumerate(results, 1):
                    x1, y1, x2, y2 = [int(v) for v in result.bbox]
                    img = cv2.imread(args.image)
                    crop = img[y1:y2, x1:x2]
                    crop_path = f"plate_{i}_{result.plate}.jpg"
                    cv2.imwrite(crop_path, crop)
                    print(f"Saved crop: {crop_path}")
        
        elif args.video:
            print(f"\nProcessing video: {args.video}")
            
            output_path = args.output or str(Path(args.video).stem) + "_output.mp4"
            
            frame_results = pipeline.process_video(
                args.video,
                output_path=output_path,
                show_progress=True
            )
            
            total_plates = sum(len(r.results) for r in frame_results)
            avg_time = sum(r.total_processing_time_ms for r in frame_results) / len(frame_results)
            
            print(f"\nProcessed {len(frame_results)} frames")
            print(f"Total plates detected: {total_plates}")
            print(f"Average processing time: {avg_time:.2f} ms/frame")
            print(f"Output saved to: {output_path}")
        
        elif args.camera is not None:
            print(f"\nStarting camera {args.camera}...")
            print("Press 'q' to quit, 's' to save frame")
            
            pipeline.process_camera(
                camera_id=args.camera,
                window_name="Vietnamese LPR"
            )
    
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
