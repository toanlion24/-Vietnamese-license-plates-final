"""
Module 2: YOLOv11 Detection - Gradio Demo
=========================================

Chạy: python src/modules/gradio_demo.py
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import tempfile
import os

# Direct import from module file (bypass src/__init__.py)
import sys
import importlib.util
spec = importlib.util.spec_from_file_location('yolo_interactive', 'src/modules/yolo_interactive.py')
yolo_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(yolo_module)

YOLOInteractiveDetector = yolo_module.YOLOInteractiveDetector
DetectionMode = yolo_module.DetectionMode


# ============================================================
# GLOBAL DETECTOR
# ============================================================

class DetectorApp:
    def __init__(self):
        self.detector = None
        self._init_detector()
    
    def _init_detector(self):
        """Initialize detector"""
        try:
            self.detector = YOLOInteractiveDetector(
                model_path="weights/best.pt",
                confidence=0.25,
                device="cuda",
                warmup=True
            )
            return "✅ Detector initialized (CUDA)"
        except Exception as e:
            try:
                self.detector = YOLOInteractiveDetector(
                    model_path="weights/best.pt",
                    confidence=0.25,
                    device="cpu",
                    warmup=True
                )
                return "⚠️ Detector initialized (CPU)"
            except:
                return f"❌ Error: {str(e)}"
    
    def detect_image(self, image, confidence, show_vehicles):
        """Detect from uploaded image"""
        if image is None:
            return None, "⚠️ Please upload an image"
        
        if self.detector is None:
            return None, "❌ Detector not initialized"
        
        # Update confidence
        self.detector.set_confidence(confidence)
        
        # Set mode
        if show_vehicles:
            self.detector.set_mode(DetectionMode.BOTH)
        else:
            self.detector.set_mode(DetectionMode.PLATE_ONLY)
        
        # Detect
        result, cropped_images, visualized = self.detector.detect(
            image,
            return_cropped=True,
            return_visualized=True,
            source="upload"
        )
        
        # Format output text
        output_text = f"✅ Found {len(result.plates)} plate(s), {len(result.vehicles)} vehicle(s)\n\n"
        
        for i, plate in enumerate(result.plates, 1):
            output_text += f"📋 Plate #{i}\n"
            output_text += f"   Bbox: {[round(x, 1) for x in plate.xyxy]}\n"
            output_text += f"   Conf: {plate.confidence:.2%}\n\n"
        
        for i, vehicle in enumerate(result.vehicles, 1):
            output_text += f"🚗 Vehicle #{i}\n"
            output_text += f"   Type: {vehicle.class_name}\n"
            output_text += f"   Bbox: {[round(x, 1) for x in vehicle.xyxy]}\n"
            output_text += f"   Conf: {vehicle.confidence:.2%}\n\n"
        
        # Return visualized image and cropped plates
        cropped_gallery = []
        if cropped_images and len(cropped_images) > 0:
            for i, crop in enumerate(cropped_images):
                if crop is not None and crop.size > 0:
                    # Resize for gallery
                    crop_resized = cv2.resize(crop, (200, 80))
                    cropped_gallery.append(crop_resized)
        
        if cropped_gallery:
            return [visualized, *cropped_gallery], output_text
        return visualized, output_text
    
    def detect_video(self, video_file, confidence, max_frames):
        """Detect from video file"""
        if video_file is None:
            return "⚠️ Please upload a video"
        
        if self.detector is None:
            return "❌ Detector not initialized"
        
        self.detector.set_confidence(confidence)
        self.detector.set_mode(DetectionMode.PLATE_ONLY)
        
        # Get video path
        if hasattr(video_file, 'name'):
            video_path = video_file.name
        else:
            video_path = str(video_file)
        
        try:
            # Process video
            results = self.detector.detect_from_video(
                video_path,
                max_frames=int(max_frames)
            )
            
            # Count detections
            total_plates = sum(len(r.plates) for r in results)
            total_vehicles = sum(len(r.vehicles) for r in results)
            
            output_text = f"✅ Processed {len(results)} frames\n"
            output_text += f"📋 Total plates detected: {total_plates}\n"
            output_text += f"🚗 Total vehicles detected: {total_vehicles}\n\n"
            
            # Show frame-by-frame results
            for i, r in enumerate(results[:20]):  # First 20 frames
                if r.has_plates:
                    output_text += f"Frame {r.frame_id}: {len(r.plates)} plate(s)\n"
            
            if len(results) > 20:
                output_text += f"... and {len(results) - 20} more frames"
            
            return output_text
            
        except Exception as e:
            return f"❌ Error: {str(e)}"


# ============================================================
# GRADIO APP
# ============================================================

def create_demo():
    """Create Gradio demo interface"""
    
    app = DetectorApp()
    
    # Custom CSS
    css = """
    .gradio-container {max-width: 1100px !important; margin: auto !important;}
    .header-box {background: linear-gradient(90deg, #1e3a5f 0%, #2d5a87 100%); padding: 25px; border-radius: 12px; color: white; text-align: center; margin-bottom: 20px;}
    .header-box h1 {margin: 0; font-size: 28px;}
    .header-box p {margin: 10px 0 0 0; opacity: 0.9;}
    .info-section {background: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 20px;}
    .result-box {border: 2px solid #dee2e6; border-radius: 8px; padding: 15px; background: white;}
    """
    
    with gr.Blocks(css=css) as demo:
        
        # Header
        gr.HTML("""
        <div class="header-box">
            <h1>🚗 YOLOv11 License Plate Detection</h1>
            <p>Module 2: Object Detection | Input: RGB Frame → Output: BoundingBox</p>
        </div>
        """)
        
        # Status
        status_text = gr.Textbox(
            label="Detector Status", 
            value=app._init_detector(), 
            interactive=False,
            info="Current model status"
        )
        
        with gr.Tabs():
            
            # ============================================
            # TAB 1: Image Detection
            # ============================================
            with gr.TabItem("📷 Image Detection"):
                gr.Markdown("### Upload an image for plate detection")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="Upload Image",
                            height=350,
                            type="numpy"
                        )
                        
                        gr.Markdown("**Settings**")
                        confidence_slider = gr.Slider(
                            minimum=0.05, maximum=0.95, value=0.25, step=0.05,
                            label="Confidence",
                            info="Lower = more detections"
                        )
                        show_vehicles = gr.Checkbox(
                            label="Show Vehicles",
                            info="Also detect vehicles (car, truck, etc.)",
                            value=False
                        )
                        
                        detect_btn = gr.Button("🔍 Detect Plates", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        output_gallery = gr.Gallery(
                            label="Detection Results",
                            columns=3,
                            object_fit="contain",
                            height=400,
                            allow_preview=True
                        )
                        output_text = gr.Textbox(
                            label="Detection Details",
                            lines=8,
                            interactive=False,
                            show_label=True
                        )
                
                detect_btn.click(
                    fn=app.detect_image,
                    inputs=[image_input, confidence_slider, show_vehicles],
                    outputs=[output_gallery, output_text]
                )
            
            # ============================================
            # TAB 2: Video
            # ============================================
            with gr.TabItem("🎬 Video Detection"):
                gr.Markdown("### Process video file for plate detection")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(
                            label="Upload Video",
                            height=300
                        )
                        
                        gr.Markdown("**Settings**")
                        video_confidence = gr.Slider(
                            minimum=0.05, maximum=0.95, value=0.25, step=0.05,
                            label="Confidence"
                        )
                        max_frames = gr.Slider(
                            minimum=10, maximum=500, value=100, step=10,
                            label="Max Frames",
                            info="More frames = longer processing"
                        )
                        
                        video_btn = gr.Button("🔍 Process Video", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        video_output = gr.Textbox(
                            label="Processing Results",
                            lines=20,
                            interactive=False
                        )
                
                video_btn.click(
                    fn=app.detect_video,
                    inputs=[video_input, video_confidence, max_frames],
                    outputs=[video_output]
                )
            
            # ============================================
            # TAB 3: Info
            # ============================================
            with gr.TabItem("ℹ️ Info"):
                gr.Markdown("""
                ## 📖 Hướng dẫn sử dụng
                
                ### Input Sources
                1. **Image**: Upload ảnh (.jpg, .png) để detect biển số
                2. **Video**: Upload video (.mp4) để process nhiều frame
                
                ### Parameters
                - **Confidence Threshold**: Ngưỡng confidence (0.05-0.95)
                  - Giảm giá trị để detect nhiều hơn
                  - Tăng giá trị để chỉ giữ detections cao
                
                ### Output
                - **Bounding Box**: Khung xanh cho biển số, đỏ cho xe
                - **Cropped Images**: Ảnh biển số đã cắt
                - **Details**: Tọa độ bbox và confidence score
                """)
                
                gr.Markdown("""
                ## 🔧 Technical Details
                
                | Component | Value |
                |-----------|-------|
                | Model | YOLOv11 |
                | Input Size | 640x640 |
                | Device | CUDA/CPU auto |
                | NMS IOU | 0.45 |
                """)
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: #666; border-top: 1px solid #eee; margin-top: 30px;">
            <p><strong>Vietnamese LPR System</strong> | Module 2: YOLOv11 Detection</p>
            <p><a href="MODULES_INDEX.md">View Documentation</a></p>
        </div>
        """)
    
    return demo


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Gradio Demo for Module 2: YOLOv11 Detection")
    print("=" * 60)
    print()
    print("📍 URL: http://localhost:7865")
    print("📍 Or use the public URL shown below")
    print()
    
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7865,
        share=False
    )
