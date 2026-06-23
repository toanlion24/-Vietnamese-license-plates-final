import cv2
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from src.modules.advanced_ocr import AdvancedLPROCRProcessor

def process_video(video_path, max_frames=30):
    print(f"Processing: {video_path}")
    print("=" * 60)
    
    # Load ONNX model directly
    model = YOLO("weights/best.onnx")
    ocr = AdvancedLPROCRProcessor(use_gpu=False)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {total_frames} frames, {fps:.1f} FPS")
    print("=" * 60)
    
    results = []
    frame_idx = 0
    sampled = 0
    
    while sampled < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Sample every 5 frames
        if frame_idx % 5 != 0:
            continue
        
        # Detect plates
        detections = model(frame, verbose=False, conf=0.25)
        
        if len(detections) > 0 and len(detections[0].boxes) > 0:
            for box in detections[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                
                # OCR
                result = ocr.process_ensemble(plate_crop)
                text = result['best_text']
                
                results.append({
                    'frame': frame_idx,
                    'plate': text,
                    'confidence': conf,
                })
                print(f"Frame {frame_idx}: {text} (conf: {conf:.2f})")
        
        sampled += 1
    
    cap.release()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    unique_plates = {}
    for r in results:
        plate = r['plate']
        if plate not in unique_plates:
            unique_plates[plate] = []
        unique_plates[plate].append(r['frame'])
    
    for i, (plate, frames) in enumerate(unique_plates.items(), 1):
        print(f"{i}. {plate} (seen at frames: {frames[0]}-{frames[-1]})")
    
    print(f"\nTotal detections: {len(results)}")
    print(f"Unique plates: {len(unique_plates)}")

if __name__ == "__main__":
    videos = [
        "C:/Users/ToanLion/Downloads/testvid/vid1.mp4",
        "C:/Users/ToanLion/Downloads/testvid/vid2.mp4",
        "C:/Users/ToanLion/Downloads/testvid/vid3.mp4",
        "C:/Users/ToanLion/Downloads/testvid/vid4.mp4",
    ]
    
    video = random.choice(videos)
    process_video(video)
