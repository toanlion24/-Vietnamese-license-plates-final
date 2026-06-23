"""
Simple Demo Web Interface for Vietnamese LPR
"""

import gradio as gr
import cv2
import numpy as np
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import VietnameseLPRPipeline


class LPRDemo:
    """Gradio-based demo interface"""
    
    def __init__(self):
        self.pipeline = None
        self._init_pipeline()
    
    def _init_pipeline(self):
        """Initialize the LPR pipeline"""
        try:
            self.pipeline = VietnameseLPRPipeline(
                detector_weights="models/yolov11/best.pt",
                device="cuda",
                min_confidence=0.5,
            )
            print("[OK] Pipeline initialized")
        except Exception as e:
            print(f"[WARN] Pipeline init issue: {e}")
            print("Will use fallback mode...")
            self.pipeline = None
    
    def process_image(self, image):
        """Process uploaded image"""
        if image is None:
            return None, "No image provided"
        
        try:
            if isinstance(image, str):
                img = cv2.imread(image)
            else:
                img = image
            
            if img is None:
                return None, "Could not read image"
            
            results, debug_img = self.pipeline.process_image(
                img,
                return_debug_image=True
            )
            
            output_text = f"Found {len(results)} plate(s):\n\n"
            for i, result in enumerate(results, 1):
                output_text += f"{i}. {result.plate}\n"
                output_text += f"   Confidence: {result.confidence:.2f}\n"
                if result.plate_type:
                    output_text += f"   Type: {result.plate_type}\n"
                output_text += "\n"
            
            debug_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            
            return debug_rgb, output_text
            
        except Exception as e:
            return None, f"Error: {str(e)}"


def create_demo():
    """Create Gradio interface"""
    
    demo = LPRDemo()
    
    with gr.Blocks(
        title="Vietnamese LPR Demo",
        theme=gr.themes.Soft()
    ) as app:
        
        gr.Markdown("""
        # 🚗 Vietnamese License Plate Recognition
        
        **YOLOv11 + PaddleOCR** for Vietnamese license plate detection and recognition
        
        ### Supported Plate Types:
        - **Private Car**: 30A-1234.56
        - **Motorcycle**: 43-12345
        - **Police**: 60-1234-56
        - **Army**: 123456-78
        """)
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="Upload Image",
                    type="filepath"
                )
                submit_btn = gr.Button("🔍 Recognize Plates", variant="primary")
            
            with gr.Column():
                image_output = gr.Image(
                    label="Detection Result",
                    interactive=False
                )
        
        results_output = gr.Textbox(
            label="Recognition Results",
            lines=10,
            interactive=False
        )
        
        gr.Examples(
            examples=[
                ["examples/car1.jpg"],
                ["examples/car2.jpg"],
            ],
            inputs=image_input,
        )
        
        submit_btn.click(
            fn=demo.process_image,
            inputs=image_input,
            outputs=[image_output, results_output]
        )
        
        gr.Markdown("""
        ---
        ### How It Works
        
        1. **Upload** an image containing Vietnamese license plates
        2. Click **Recognize Plates** to process
        3. View **detection boxes** and **recognized text**
        
        ### Pipeline Architecture
        
        ```
        Image → Preprocessing → YOLOv11 Detection → Plate Crop → PaddleOCR → Results
        ```
        """)
    
    return app


def main():
    print("=" * 60)
    print("Vietnamese LPR Demo Server")
    print("=" * 60)
    print()
    print("Starting Gradio server...")
    print("Open http://localhost:7860 in your browser")
    print()
    
    app = create_demo()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
