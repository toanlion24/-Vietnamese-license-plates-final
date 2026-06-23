import cv2
import sys
import random
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from src.modules.advanced_ocr import AdvancedLPROCRProcessor

def process_video_visual(video_path, output_dir="output_frames", max_frames=20):
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"Processing: {video_path}")
    print("=" * 60)
    
    model = YOLO("weights/best.onnx")
    ocr = AdvancedLPROCRProcessor(use_gpu=False)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {total_frames} frames")
    print("=" * 60)
    
    results = []
    frame_idx = 0
    saved = 0
    
    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        if frame_idx % 10 != 0:
            continue
        
        detections = model(frame, verbose=False, conf=0.25)
        
        if len(detections) > 0 and len(detections[0].boxes) > 0:
            frame_copy = frame.copy()
            
            for i, box in enumerate(detections[0].boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                
                if plate_crop.size == 0:
                    continue
                
                result = ocr.process_ensemble(plate_crop)
                text = result['best_text']
                
                results.append({
                    'frame': frame_idx,
                    'plate': text,
                    'confidence': conf,
                })
                
                # Draw box
                cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Put text
                label = f"{text} ({conf:.2f})"
                cv2.putText(frame_copy, label, (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Save cropped plate
                plate_resized = cv2.resize(plate_crop, (300, 100))
                cv2.imwrite(f"{output_dir}/plate_f{frame_idx}_{i}.jpg", plate_resized)
            
            # Save frame with annotations
            cv2.imwrite(f"{output_dir}/frame_{frame_idx}.jpg", frame_copy)
            print(f"Frame {frame_idx}: {len(detections[0].boxes)} plates detected")
            saved += 1
    
    cap.release()
    
    print("\n" + "=" * 60)
    print(f"Saved {saved} frames to {output_dir}/")
    print("=" * 60)
    
    # Create montage
    create_montage(output_dir, results)

def create_montage(output_dir, results):
    frames = sorted(Path(output_dir).glob("frame_*.jpg"))
    if not frames:
        return
    
    rows = min(4, (len(frames) + 3) // 4)
    cols = min(5, len(frames))
    
    frame_h, frame_w = 360, 640
    montage = np.zeros((frame_h * rows, frame_w * cols, 3), dtype=np.uint8)
    
    for idx, frame_path in enumerate(frames[:rows*cols]):
        img = cv2.imread(str(frame_path))
        img = cv2.resize(img, (frame_w, frame_h))
        r, c = idx // cols, idx % cols
        montage[r*frame_h:(r+1)*frame_h, c*frame_w:(c+1)*frame_w] = img
    
    cv2.imwrite(f"{output_dir}/montage.jpg", montage)
    print(f"Montage saved: {output_dir}/montage.jpg")

if __name__ == "__main__":
    videos = [
        "C:/Users/ToanLion/Downloads/testvid/vid1.mp4",
        "C:/Users/ToanLion/Downloads/testvid/vid2.mp4",
        "C:/Users/ToanLion/Downloads/testvid/vid3.mp4",
        "C:/Users/ToanLion/Downloads/testvid/vid4.mp4",
    ]
    
    video = random.choice(videos)
    process_video_visual(video)
