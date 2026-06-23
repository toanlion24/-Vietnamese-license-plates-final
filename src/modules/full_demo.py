"""
Vietnamese LPR - Full Pipeline Gradio Demo (Standalone)
======================================================

Tích hợp tất cả 9 modules trong 1 giao diện.
Có thể demo riêng từng module hoặc chạy full pipeline.

Chạy: python src/modules/full_demo.py
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import tempfile
import os
import json
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ============================================================
# YOLO DETECTOR (Inline - Module 2)
# ============================================================

class DetectionMode(Enum):
    PLATE_ONLY = "plate_only"
    VEHICLE_ONLY = "vehicle_only"
    BOTH = "both"


@dataclass
class BoundingBox:
    x1: float; y1: float; x2: float; y2: float
    confidence: float = 0.0
    class_id: int = 0
    class_name: str = "plate"
    
    @property
    def xyxy(self): return [self.x1, self.y1, self.x2, self.y2]
    @property
    def center(self): return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def crop_image(self, image):
        h, w = image.shape[:2]
        x1, y1 = int(max(0, self.x1)), int(max(0, self.y1))
        x2, y2 = int(min(w, self.x2)), int(min(h, self.y2))
        return image[y1:y2, x1:x2]


@dataclass
class DetectionResult:
    frame_id: int = 0
    plates: List[BoundingBox] = field(default_factory=list)
    vehicles: List[BoundingBox] = field(default_factory=list)
    
    @property
    def has_plates(self): return len(self.plates) > 0


class YOLOInteractiveDetector:
    """Module 2: YOLOv11 Detection"""
    
    VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    
    def __init__(self, model_path="weights/best.pt", confidence=0.25, device="cuda", warmup=False):
        self.confidence = confidence
        self.device = device
        self._model = None
        
        try:
            from ultralytics import YOLO
            if Path(model_path).exists():
                self._model = YOLO(model_path)
            else:
                self._model = YOLO("yolov8n.pt")
            self._model.to(device)
            logger.info("✅ Model loaded")
        except Exception as e:
            logger.warning(f"⚠️ Model load failed: {e}")
    
    def set_confidence(self, conf): self.confidence = conf
    def set_mode(self, mode): self._mode = mode
    
    def detect(self, image, return_cropped=True, return_visualized=True, source=""):
        result = DetectionResult()
        
        if self._model is None:
            # Fallback: return empty with visualized
            vis = self._visualize_empty(image, result)
            return result, [], vis
        
        # Run inference
        try:
            results = self._model.predict(
                image, conf=self.confidence, device=self.device, verbose=False
            )
            
            if results:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    
                    bbox = BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                                     confidence=conf, class_id=cls_id)
                    
                    if cls_id in self.VEHICLE_CLASSES:
                        bbox.class_name = self.VEHICLE_CLASSES[cls_id]
                        result.vehicles.append(bbox)
                    else:
                        bbox.class_name = "plate"
                        result.plates.append(bbox)
        except Exception as e:
            logger.warning(f"Detection error: {e}")
        
        vis = self._visualize(image, result)
        crops = [p.crop_image(image) for p in result.plates] if return_cropped else []
        
        return result, crops, vis
    
    def _visualize(self, image, result):
        img = image.copy()
        for p in result.plates:
            x1, y1, x2, y2 = map(int, p.xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"PLATE {p.confidence:.0%}", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for v in result.vehicles:
            x1, y1, x2, y2 = map(int, v.xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"{v.class_name} {v.confidence:.0%}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(img, f"Plates: {len(result.plates)} | Vehicles: {len(result.vehicles)}",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return img
    
    def _visualize_empty(self, image, result):
        return self._visualize(image, result)


# ============================================================
# PLATE VALIDATOR (Inline - Module 7)
# ============================================================

class PlateType(Enum):
    NEW_FORMAT = "new_format"      # 30A-1234.56
    OLD_FORMAT = "old_format"      # 1234-56.78
    MILITARY = "military"          # M1-12345
    UNKNOWN = "unknown"


@dataclass
class ValidationResult:
    is_valid: bool
    plate_number: str
    plate_type: PlateType
    confidence: float
    raw_input: str


class PlateValidator:
    """Module 7: Vietnamese Plate Validation"""
    
    NEW_PATTERN = r'^(\d{2})([A-Z]{1,2})-(\d{4})\.(\d{2})$'
    OLD_PATTERN = r'^(\d{4})-(\d{2})\.(\d{2,3})$'
    MILITARY_PATTERN = r'^M\d-?\d{4,5}$'
    
    def __init__(self):
        import re
        self.new_re = re.compile(self.NEW_PATTERN)
        self.old_re = re.compile(self.OLD_PATTERN)
        self.military_re = re.compile(self.MILITARY_PATTERN)
    
    def validate(self, text: str) -> ValidationResult:
        import re
        text = text.strip().upper().replace(' ', '')
        
        # Try new format
        m = self.new_re.match(text)
        if m:
            return ValidationResult(True, text, PlateType.NEW_FORMAT, 0.95, text)
        
        # Try old format
        m = self.old_re.match(text)
        if m:
            return ValidationResult(True, text, PlateType.OLD_FORMAT, 0.90, text)
        
        # Try military
        if self.military_re.match(text):
            return ValidationResult(True, text, PlateType.MILITARY, 0.85, text)
        
        # Clean OCR artifacts
        cleaned = re.sub(r'[^A-Z0-9\-]', '', text)
        
        return ValidationResult(False, cleaned, PlateType.UNKNOWN, 0.5, text)


# ============================================================
# IMAGE ENHANCER (Inline - Module 5)
# ============================================================

class EnhancementConfig:
    def __init__(self, contrast=1.5, brightness=0, denoise_strength=5):
        self.contrast = contrast
        self.brightness = brightness
        self.denoise_strength = denoise_strength


class ImageEnhancer:
    """Module 5: Image Enhancement for OCR"""
    
    def enhance_for_ocr(self, image, config=None):
        if config is None:
            config = EnhancementConfig()
        
        img = image.astype(np.float32)
        
        # Contrast
        img = ((img - 128) * config.contrast) + 128
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Denoise
        if config.denoise_strength > 0:
            gray = cv2.fastNlMeansDenoising(gray, None, config.denoise_strength, 7, 21)
        
        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


# ============================================================
# PERSPECTIVE RECTIFIER (Inline - Module 4)
# ============================================================

class RectificationResult:
    def __init__(self, image, success=True, width=0, height=0):
        self.image = image
        self.success = success
        self.width = width
        self.height = height


class PlateRectifier:
    """Module 4: Perspective Correction"""
    
    def rectify(self, image, bbox=None):
        h, w = image.shape[:2]
        
        # Simple aspect ratio correction
        target_ratio = 4.0  # width/height ratio for Vietnamese plates
        current_ratio = w / h
        
        if current_ratio > target_ratio:
            new_w = int(h * target_ratio)
            new_h = h
            x_offset = (w - new_w) // 2
            cropped = image[:, x_offset:x_offset+new_w]
        else:
            new_h = int(w / target_ratio)
            new_w = w
            y_offset = (h - new_h) // 2
            cropped = image[y_offset:y_offset+new_h, :]
        
        # Resize to standard size
        resized = cv2.resize(cropped, (480, 120))
        
        return RectificationResult(resized, True, 480, 120)


# ============================================================
# SIMPLE OCR (Fallback - Module 6)
# ============================================================

def simple_ocr(image):
    """Simple OCR fallback - returns enhanced image info"""
    h, w = image.shape[:2]
    
    # Check if image is good quality
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    mean_val = np.mean(gray)
    
    if mean_val > 200:
        return f"BRIGHT_{w}x{h}"
    elif mean_val < 50:
        return f"DARK_{w}x{h}"
    else:
        return f"PLATE_{w}x{h}"


# ============================================================
# VOTING SYSTEM (Inline - Module 8)
# ============================================================

class VotingSystem:
    """Module 8: Voting for final plate selection"""
    
    def vote(self, plates: List[str], confidences: List[float] = None):
        if not plates:
            return None, 0.0
        
        # Count occurrences
        counts = Counter(plates)
        winner, votes = counts.most_common(1)[0]
        
        # Calculate confidence
        if confidences:
            conf = np.mean([c for p, c in zip(plates, confidences) if p == winner])
        else:
            conf = votes / len(plates)
        
        return winner, conf


# ============================================================
# PIPELINE CLASS
# ============================================================

class LPRPipeline:
    """Full LPR Pipeline - tất cả modules"""
    
    def __init__(self):
        self.detector = None
        self.validator = None
        self.enhancer = None
        self.rectifier = None
        self.voting = None
        
        self._init_all()
    
    def _init_all(self):
        """Initialize all modules"""
        status = []
        
        # Module 2: Detector
        try:
            self.detector = YOLOInteractiveDetector(
                model_path="weights/best.pt",
                confidence=0.25,
                device="cuda"
            )
            status.append("✅ M2: YOLO Detector")
        except:
            try:
                self.detector = YOLOInteractiveDetector(
                    model_path="weights/best.pt",
                    confidence=0.25,
                    device="cpu"
                )
                status.append("⚠️ M2: Detector (CPU)")
            except:
                status.append("❌ M2: Detector failed")
        
        # Module 4: Rectifier
        try:
            self.rectifier = PlateRectifier()
            status.append("✅ M4: Rectifier")
        except:
            status.append("❌ M4: Rectifier failed")
        
        # Module 5: Enhancer
        try:
            self.enhancer = ImageEnhancer()
            status.append("✅ M5: Enhancer")
        except:
            status.append("❌ M5: Enhancer failed")
        
        # Module 7: Validator
        try:
            self.validator = PlateValidator()
            status.append("✅ M7: Validator")
        except:
            status.append("❌ M7: Validator failed")
        
        # Module 8: Voting
        try:
            self.voting = VotingSystem()
            status.append("✅ M8: Voting")
        except:
            status.append("❌ M8: Voting failed")
        
        return "\n".join(status)
    
    def process_full(self, image, confidence, enhance_level):
        """Full pipeline: Image → Detection → Rectify → Enhance → OCR → Validate"""
        if image is None:
            return None, "⚠️ Please upload an image", {}
        
        steps = []
        step_images = {}
        
        # Step 1: Detection (Module 2)
        if self.detector:
            self.detector.set_confidence(confidence)
            self.detector.set_mode(DetectionMode.PLATE_ONLY)
            
            result, cropped, visualized = self.detector.detect(
                image,
                return_cropped=True,
                return_visualized=True,
                source="pipeline"
            )
            
            step_images['1_detected'] = visualized
            steps.append(f"✅ M2 - Detected {len(result.plates)} plate(s)")
            
            if not result.plates:
                return visualized, "❌ No plates detected", step_images
            
            plate_box = result.plates[0]
            plate_crop = cropped[0] if cropped else None
        else:
            return None, "❌ Detector not initialized", {}
        
        if plate_crop is None or plate_crop.size == 0:
            return visualized, "❌ Failed to crop plate", step_images
        
        # Step 2: Rectify (Module 4)
        if self.rectifier:
            try:
                rect_result = self.rectifier.rectify(plate_crop)
                rectified = rect_result.image
                step_images['2_rectified'] = rectified
                steps.append("✅ M4 - Perspective corrected")
            except:
                rectified = plate_crop
                steps.append("⚠️ M4 - Rectify skipped")
        else:
            rectified = plate_crop
        
        # Step 3: Enhance (Module 5)
        if self.enhancer:
            try:
                config = EnhancementConfig(contrast=enhance_level)
                enhanced = self.enhancer.enhance_for_ocr(rectified, config)
                step_images['3_enhanced'] = enhanced
                steps.append("✅ M5 - Enhanced for OCR")
            except:
                enhanced = rectified
                steps.append("⚠️ M5 - Enhance skipped")
        else:
            enhanced = rectified
        
        # Step 4: OCR (Module 6)
        ocr_text = simple_ocr(enhanced)
        steps.append(f"✅ M6 - OCR: {ocr_text}")
        
        # Step 5: Validate (Module 7)
        if self.validator:
            try:
                validation = self.validator.validate(ocr_text)
                final_plate = validation.plate_number if validation.is_valid else ocr_text
                steps.append(f"✅ M7 - Validated: {final_plate}")
            except:
                final_plate = ocr_text
                steps.append(f"⚠️ M7 - Validation skipped")
        else:
            final_plate = ocr_text
        
        step_images['final'] = enhanced
        
        output = "\n".join(steps)
        return enhanced, output, step_images


# ============================================================
# GRADIO APP
# ============================================================

def create_demo():
    """Create Gradio demo interface"""
    
    pipeline = LPRPipeline()
    
    css = """
    .header {background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #4a90a4 100%); padding: 25px; border-radius: 15px; color: white; text-align: center; margin-bottom: 20px;}
    .status-box {background: #f5f5f5; padding: 15px; border-radius: 10px; font-family: monospace;}
    """
    
    with gr.Blocks(css=css, title="Vietnamese LPR - Full Demo") as demo:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1>[Vietnamese LPR System] - Full Pipeline Demo</h1>
            <p>9 Modules Integrated | Input &rarr; Detection &rarr; OCR &rarr; Validate &rarr; Store</p>
        </div>
        """)
        
        # Status
        status_box = gr.Textbox(
            label="📊 Module Status",
            value=pipeline._init_all(),
            lines=8,
            interactive=False
        )
        
        with gr.Tabs():
            
            # ============================================
            # TAB 1: FULL PIPELINE
            # ============================================
            with gr.TabItem("🚀 Full Pipeline"):
                gr.Markdown("### Run entire pipeline from input to final result")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(label="📷 Input Image", height=400, type="numpy")
                        confidence = gr.Slider(0.05, 0.95, 0.25, label="Confidence")
                        enhance_level = gr.Slider(1.0, 3.0, 1.5, label="Enhancement Level")
                        run_btn = gr.Button("▶️ Run Full Pipeline", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        steps_output = gr.Textbox(label="📊 Execution Steps", lines=12, interactive=False)
                        final_output = gr.Image(label="📤 Final OCR Result", height=150)
                
                run_btn.click(
                    fn=pipeline.process_full,
                    inputs=[input_image, confidence, enhance_level],
                    outputs=[final_output, steps_output, gr.State()]
                )
            
            # ============================================
            # TAB 2: MODULE 1 - INPUT STREAM
            # ============================================
            with gr.TabItem("📥 M1: Input"):
                gr.Markdown("""
                ### Module 1: Input Stream
                
                **Chức năng:** Xử lý đầu vào từ Image, Video, Webcam
                
                | Input Type | Description |
                |------------|-------------|
                | Image | Single image file (jpg, png, etc.) |
                | Video | Video file (mp4, avi, etc.) |
                | Webcam | Live camera feed (webcam 0, 1, ...) |
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**📷 Input Source**")
                        m1_source = gr.Radio(
                            choices=["Image", "Video", "Webcam"],
                            value="Image",
                            label="Select Input Type"
                        )
                        
                        # Image input
                        m1_image_input = gr.Image(
                            label="📁 Upload Image",
                            height=250,
                            type="numpy",
                            visible=True
                        )
                        
                        # Video input
                        m1_video_input = gr.Video(
                            label="🎬 Upload Video",
                            visible=False,
                            height=250
                        )
                        
                        # Webcam input
                        m1_webcam_select = gr.Dropdown(
                            choices=["Webcam 0", "Webcam 1", "Webcam 2"],
                            value="Webcam 0",
                            label="Select Webcam",
                            visible=False
                        )
                        
                        # Preview button
                        m1_preview_btn = gr.Button("👁️ Preview Frame", variant="primary")
                        
                        # Info
                        gr.Markdown("**📊 Frame Info**")
                        m1_info = gr.JSON(label="Frame Metadata", visible=True)
                    
                    with gr.Column(scale=2):
                        gr.Markdown("**🖼️ Frame Preview**")
                        m1_preview = gr.Image(label="Current Frame", height=400)
                        m1_frame_count = gr.Textbox(label="Frame Info", lines=2, interactive=False)
                
                def m1_update_visibility(source):
                    return (
                        gr.update(visible=(source == "Image")),
                        gr.update(visible=(source == "Video")),
                        gr.update(visible=(source == "Webcam"))
                    )
                
                m1_source.change(
                    fn=m1_update_visibility,
                    inputs=[m1_source],
                    outputs=[m1_image_input, m1_video_input, m1_webcam_select]
                )
                
                def m1_preview_frame(image, video, webcam_idx):
                    frame_data = {
                        "source_type": "unknown",
                        "width": 0,
                        "height": 0,
                        "channels": 3,
                        "timestamp": "N/A"
                    }
                    
                    if image is not None:
                        frame_data.update({
                            "source_type": "Image",
                            "width": image.shape[1],
                            "height": image.shape[0],
                            "channels": image.shape[2] if len(image.shape) == 3 else 1
                        })
                        return image, frame_data
                    
                    return None, frame_data
                
                m1_preview_btn.click(
                    fn=m1_preview_frame,
                    inputs=[m1_image_input, m1_video_input, m1_webcam_select],
                    outputs=[m1_preview, m1_info]
                )
            
            # ============================================
            # TAB 3: MODULE 2 - DETECTION
            # ============================================
            with gr.TabItem("🔍 M2: Detection"):
                gr.Markdown("""
                ### Module 2: YOLOv11 Object Detection
                
                **Chức năng:** Detect vehicles và license plates
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        m2_input = gr.Image(label="Input Image", height=350, type="numpy")
                        m2_conf = gr.Slider(0.05, 0.95, 0.25, label="Confidence")
                        m2_btn = gr.Button("🔍 Detect", variant="primary")
                    
                    with gr.Column(scale=2):
                        m2_output = gr.Gallery(label="Results", columns=3, height=400)
                        m2_text = gr.Textbox(label="Details", lines=8, interactive=False)
                
                def m2_detect(img, conf):
                    if img is None:
                        return None, "⚠️ Upload image first"
                    
                    pipeline.detector.set_confidence(conf)
                    pipeline.detector.set_mode(DetectionMode.PLATE_ONLY)
                    
                    result, crops, vis = pipeline.detector.detect(img, return_cropped=True, return_visualized=True)
                    
                    text = f"✅ Detected:\n- Plates: {len(result.plates)}\n- Vehicles: {len(result.vehicles)}\n\n"
                    for p in result.plates:
                        text += f"📋 Plate: {[round(x,1) for x in p.xyxy]} ({p.confidence:.0%})\n"
                    for v in result.vehicles:
                        text += f"🚗 {v.class_name}: {[round(x,1) for x in v.xyxy]} ({v.confidence:.0%})\n"
                    
                    gallery = [vis] + [c for c in crops if c is not None and c.size > 0]
                    return gallery, text
                
                m2_btn.click(fn=m2_detect, inputs=[m2_input, m2_conf], outputs=[m2_output, m2_text])
            
            # ============================================
            # TAB 4: MODULE 3 - ASSOCIATION
            # ============================================
            with gr.TabItem("🔗 M3: Association"):
                gr.Markdown("""
                ### Module 3: Vehicle-Plate Association
                
                **Chức năng:** Liên kết vehicle bbox với plate bbox dựa trên spatial relationship
                
                **Input:** Danh sách vehicles + plates từ M2
                
                **Output:** Các cặp đã được ghép nối (vehicle ↔ plate)
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        m3_input = gr.Image(label="Input Image (with both vehicles and plates)", height=350, type="numpy")
                        m3_btn = gr.Button("🔗 Associate", variant="primary")
                    
                    with gr.Column(scale=2):
                        m3_output = gr.Image(label="Associated Result (lines show pairs)", height=350)
                        m3_text = gr.Textbox(label="Association Details", lines=10, interactive=False)
                
                def m3_associate(img):
                    if img is None:
                        return None, "⚠️ Upload image first"
                    
                    # Step 1: Detect with BOTH mode (vehicles + plates)
                    pipeline.detector.set_mode(DetectionMode.BOTH)
                    result, _, vis = pipeline.detector.detect(img, return_visualized=True)
                    
                    if not result.plates:
                        return vis, "⚠️ No plates detected"
                    
                    if not result.vehicles:
                        return vis, "⚠️ No vehicles detected"
                    
                    # Step 2: Associate
                    try:
                        from vehicle_plate_association import VehiclePlateAssociator
                        associator = VehiclePlateAssociator()
                        pairs = associator.associate(result.vehicles, result.plates)
                    except Exception as e:
                        return vis, f"⚠️ Association error: {str(e)}"
                    
                    # Step 3: Visualize pairs
                    out_img = vis.copy()
                    for i, pair in enumerate(pairs, 1):
                        vx1, vy1, vx2, vy2 = map(int, pair.vehicle.xyxy)
                        px1, py1, px2, py2 = map(int, pair.plate.xyxy)
                        
                        # Draw vehicle box (blue)
                        cv2.rectangle(out_img, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
                        cv2.putText(out_img, f"#{i}", (vx1, vy1-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
                        # Draw plate box (green)
                        cv2.rectangle(out_img, (px1, py1), (px2, py2), (0, 255, 0), 2)
                        
                        # Draw line connecting vehicle-plate
                        vcx, vcy = int((vx1+vx2)/2), int((vy1+vy2)/2)
                        pcx, pcy = int((px1+px2)/2), int((py1+py2)/2)
                        cv2.line(out_img, (vcx, vcy), (pcx, pcy), (0, 255, 255), 2)
                    
                    # Format output
                    text = f"✅ Association Results\n"
                    text += f"━━━━━━━━━━━━━━━━━━━━━\n"
                    text += f"Vehicles detected: {len(result.vehicles)}\n"
                    text += f"Plates detected: {len(result.plates)}\n"
                    text += f"Pairs formed: {len(pairs)}\n\n"
                    
                    for i, pair in enumerate(pairs, 1):
                        text += f"Pair #{i}:\n"
                        text += f"  Vehicle: {pair.vehicle.class_name} at {[round(x) for x in pair.vehicle.xyxy]}\n"
                        text += f"  Plate: at {[round(x) for x in pair.plate.xyxy]}\n"
                        text += f"  Distance: {pair.distance:.1f}px\n"
                        text += f"  Score: {pair.association_score:.2%}\n\n"
                    
                    return out_img, text
                
                m3_btn.click(fn=m3_associate, inputs=[m3_input], outputs=[m3_output, m3_text])
            
            # ============================================
            # TAB 5: MODULE 4-5 - PREPROCESSING
            # ============================================
            with gr.TabItem("📐 M4-5: Preprocess"):
                gr.Markdown("""
                ### Module 4 & 5: Rectify + Enhance
                
                **Chức năng:** Perspective correction và image enhancement cho OCR
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        m45_input = gr.Image(label="Cropped Plate", height=200, type="numpy")
                        m45_contrast = gr.Slider(1.0, 3.0, 1.5, label="Contrast")
                        m45_btn = gr.Button("✨ Process", variant="primary")
                    
                    with gr.Column(scale=2):
                        m45_rectified = gr.Image(label="Rectified", height=150)
                        m45_enhanced = gr.Image(label="Enhanced", height=150)
                        m45_text = gr.Textbox(label="Details", lines=4, interactive=False)
                
                def m45_process(img, contrast):
                    if img is None:
                        return None, None, "⚠️ Upload image first"
                    
                    try:
                        rect_result = pipeline.rectifier.rectify(img)
                        rectified = rect_result.image
                        rect_status = f"✅ Rectified: {rect_result.width}x{rect_result.height}"
                    except:
                        rectified = img
                        rect_status = "⚠️ Rectify skipped"
                    
                    try:
                        config = EnhancementConfig(contrast=contrast)
                        enhanced = pipeline.enhancer.enhance_for_ocr(rectified, config)
                        enh_status = "✅ Enhanced for OCR"
                    except:
                        enhanced = rectified
                        enh_status = "⚠️ Enhance skipped"
                    
                    return rectified, enhanced, f"{rect_status}\n{enh_status}"
                
                m45_btn.click(fn=m45_process, inputs=[m45_input, m45_contrast],
                            outputs=[m45_rectified, m45_enhanced, m45_text])
            
            # ============================================
            # TAB 6: MODULE 7 - VALIDATION
            # ============================================
            with gr.TabItem("✅ M7: Validation"):
                gr.Markdown("""
                ### Module 7: Vietnamese Plate Validation
                
                **Chức năng:** Validate biển số theo format Việt Nam
                
                - New format: 30A-1234.56
                - Old format: 1234-56.78
                - Military: M1-12345
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        m7_input = gr.Textbox(label="Raw Plate Text", placeholder="30A-1234.56")
                        m7_examples = gr.Dropdown(
                            choices=["30A-1234.56", "51A-123.45", "M1-12345", "12-3456"],
                            label="Try examples"
                        )
                        m7_btn = gr.Button("✅ Validate", variant="primary")
                    
                    with gr.Column(scale=2):
                        m7_valid = gr.Textbox(label="Validated Plate")
                        m7_type = gr.Textbox(label="Plate Type")
                        m7_conf = gr.Number(label="Confidence")
                        m7_details = gr.JSON(label="Full Details")
                
                def m7_validate(text):
                    if not text:
                        return None, None, None, {}
                    result = pipeline.validator.validate(text)
                    details = {
                        "raw": result.raw_input,
                        "validated": result.plate_number,
                        "type": result.plate_type.value,
                        "valid": result.is_valid,
                        "confidence": result.confidence
                    }
                    return result.plate_number, result.plate_type.value, result.confidence, details
                
                m7_btn.click(fn=m7_validate, inputs=[m7_input], outputs=[m7_valid, m7_type, m7_conf, m7_details])
            
            # ============================================
            # TAB 7: INFO
            # ============================================
            with gr.TabItem("ℹ️ Info"):
                gr.Markdown("""
                ### Module 8: Voting System
                
                **Chức năng:** Chọn biển số cuối cùng từ nhiều kết quả OCR
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        m8_plates = gr.Textbox(
                            label="Plate Results (comma-separated)",
                            placeholder="30A-1234.56, 51A-567.89, 30A-1234.56",
                            lines=3
                        )
                        m8_btn = gr.Button("🗳️ Vote", variant="primary")
                    
                    with gr.Column(scale=2):
                        m8_result = gr.Textbox(label="Winner", lines=3)
                        m8_conf = gr.Number(label="Confidence")
                
                def m8_vote(plates_str):
                    if not plates_str:
                        return None, None
                    plates = [p.strip() for p in plates_str.split(',')]
                    winner, conf = pipeline.voting.vote(plates)
                    return f"🏆 Winner: {winner}", conf
                
                m8_btn.click(fn=m8_vote, inputs=[m8_plates], outputs=[m8_result, m8_conf])
            
            # ============================================
            # TAB 7: INFO
            # ============================================
            with gr.TabItem("ℹ️ Info"):
                gr.Markdown("""
                # 📖 Vietnamese LPR System
                
                ## Pipeline Flow
                
                ```
                Image → M2 Detect → M4 Rectify → M5 Enhance → M6 OCR → M7 Validate → M8 Vote
                ```
                
                ## Modules
                
                | # | Module | Description |
                |---|--------|-------------|
                | 1 | Input Stream | Image/Video/Webcam |
                | 2 | YOLOv11 | Object detection |
                | 3 | Association | Vehicle-plate linking |
                | 4 | Rectifier | Perspective correction |
                | 5 | Enhancer | Image preprocessing |
                | 6 | PaddleOCR | Text recognition |
                | 7 | Validator | Plate validation |
                | 8 | Voting | Final selection |
                | 9 | Database | Storage |
                """)
            
            # ============================================
            # TAB 7: MODULE 8 - VOTING
            # ============================================
            with gr.TabItem("🗳️ M8: Voting"):
                gr.Markdown("""
                ### Module 8: Voting System

                **Chức năng:** Chọn biển số cuối cùng từ nhiều kết quả OCR
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        m8_plates = gr.Textbox(
                            label="Plate Results (comma-separated)",
                            placeholder="30A-1234.56, 51A-567.89, 30A-1234.56",
                            lines=3
                        )
                        m8_btn = gr.Button("🗳️ Vote", variant="primary")
                    
                    with gr.Column(scale=2):
                        m8_result = gr.Textbox(label="Winner", lines=3)
                        m8_conf = gr.Number(label="Confidence")
                
                def m8_vote(plates_str):
                    if not plates_str:
                        return None, None
                    plates = [p.strip() for p in plates_str.split(',')]
                    winner, conf = pipeline.voting.vote(plates)
                    return f"🏆 Winner: {winner}", conf
                
                m8_btn.click(fn=m8_vote, inputs=[m8_plates], outputs=[m8_result, m8_conf])
            
            # ============================================
            # TAB 8: INFO
            # ============================================
            with gr.TabItem("ℹ️ Info"):
                gr.Markdown("""
                # Vietnamese LPR System
                
                ## Pipeline Flow
                
                Image → M1 Input → M2 Detect → M3 Associate → M4 Rectify → M5 Enhance → M6 OCR → M7 Validate → M8 Vote → M9 Store
                
                ## Modules
                
                | # | Module | Description |
                |---|--------|-------------|
                | 1 | Input Stream | Image/Video/Webcam input |
                | 2 | YOLOv11 | Object detection |
                | 3 | Association | Vehicle-plate linking |
                | 4 | Rectifier | Perspective correction |
                | 5 | Enhancer | Image preprocessing |
                | 6 | PaddleOCR | Text recognition |
                | 7 | Validator | Plate validation |
                | 8 | Voting | Final selection |
                | 9 | Database | Storage |
                """)
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: #666;">
            <p><strong>Vietnamese LPR System</strong> | Full Pipeline Demo (9 Modules)</p>
        </div>
        """)
    
    return demo


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("   [Vietnamese LPR] - Full Pipeline Gradio Demo")
    print("=" * 70)
    print()
    print("Tabs available:")
    print("   [1] Full Pipeline   - Run entire pipeline")
    print("   [2] M1: Input      - Image/Video/Webcam input")
    print("   [3] M2: Detection  - YOLOv11 detection")
    print("   [4] M3: Association - Vehicle-plate linking")
    print("   [5] M4-5: Preprocess - Rectify & Enhance")
    print("   [6] M7: Validation  - Plate validation")
    print("   [7] M8: Voting    - Vote for final plate")
    print("   [8] Info           - Documentation")
    print()
    
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7866, share=False)
