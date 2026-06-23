"""
Vietnamese License Plate Detection Demo - Interactive Version
Gradio Interface with Auto-Processing Features
"""

import gradio as gr
import cv2
import numpy as np
import sys
import json
import os
import time
import threading
from pathlib import Path
from datetime import datetime
import importlib.util
import glob
import shutil
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

# Import detector
spec = importlib.util.spec_from_file_location('detector', 'src/detection/detector.py')
detector_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detector_module)
PlateDetector = detector_module.PlateDetector
visualize_detections = detector_module.visualize_detections


class InteractiveLPRDemo:
    """Interactive Gradio demo with auto-processing features"""
    
    def __init__(self, model_path="weights/best.pt", conf_threshold=0.25):
        print(f"[INFO] Loading model: {model_path}")
        self.detector = PlateDetector(
            model_path=model_path,
            confidence_threshold=conf_threshold,
            device="cuda"
        )
        self.watch_folder = None
        self.watch_thread = None
        self.latest_results = []
        print("[OK] Model loaded successfully!")
    
    def process_single_image(self, image, conf_threshold):
        """Process single uploaded image"""
        if image is None:
            return None, "Khong co anh duoc upload"
        
        try:
            img = image if isinstance(image, np.ndarray) else cv2.imread(image)
            if img is None:
                return None, "Khong the doc anh"
            
            self.detector.confidence_threshold = conf_threshold
            detections = self.detector.detect(img, return_cropped=True)
            debug_img = visualize_detections(img.copy(), detections)
            debug_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            
            return debug_rgb, self._format_results(detections)
            
        except Exception as e:
            return None, f"Loi: {str(e)}"
    
    def process_batch(self, folder_path: str, conf_threshold: float):
        """Process all images in a folder"""
        if not folder_path or not os.path.exists(folder_path):
            return "Thu muc khong ton tai", None, None
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        if not image_files:
            return f"Khong tim thay anh trong: {folder_path}", None, None
        
        results_text = f"TIM THAY {len(image_files)} ANH\n\n"
        processed_count = 0
        
        for img_path in image_files[:10]:  # Limit to 10 images
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            self.detector.confidence_threshold = conf_threshold
            detections = self.detector.detect(img, return_cropped=True)
            
            if detections:
                results_text += f"[OK] {os.path.basename(img_path)}: {len(detections)} bien so\n"
                processed_count += 1
            else:
                results_text += f"[--] {os.path.basename(img_path)}: khong tim thay\n"
        
        results_text += f"\nDa xu ly: {processed_count}/{len(image_files)} anh co bien so"
        
        # Return first detected image as preview
        preview_img = visualize_detections(img.copy(), detections) if 'img' in locals() else None
        preview_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB) if preview_img is not None else None
        
        return results_text, preview_rgb, len(image_files)
    
    def _format_results(self, detections):
        """Format detection results"""
        if not detections:
            return "Khong tim thay bien so nao!"
        
        text = f"**TIM THAY {len(detections)} BIEN SO:**\n\n"
        text += "| # | Vi tri | Confidence | Kich thuoc |\n"
        text += "|:--:|:------:|:----------:|:----------:|\n"
        
        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det.bbox
            text += f"| {i} | `[{int(x1)}, {int(y1)}]` | **{det.confidence:.1%}** | {int(x2-x1)}x{int(y2-y1)}px |\n"
        
        text += f"\n**Trung binh:** {np.mean([d.confidence for d in detections]):.1%}"
        return text


class FolderWatcher:
    """Watch folder for new images and auto-process"""
    
    def __init__(self, folder_path: str, detector, processed_folder: str):
        self.folder_path = Path(folder_path)
        self.processed_folder = Path(processed_folder)
        self.detector = detector
        self.processed_files = set()
        self.running = False
        self.results = []
        
        # Mark already processed files
        for f in self.folder_path.glob("*"):
            if f.suffix.lower() in ['.jpg', '.png', '.bmp']:
                self.processed_files.add(f.name)
        
        self.processed_folder.mkdir(exist_ok=True)
    
    def start(self, callback=None):
        """Start watching folder"""
        self.running = True
        self.thread = threading.Thread(target=self._watch_loop, args=(callback,))
        self.thread.daemon = True
        self.thread.start()
        return f"Da bat dau theo doi: {self.folder_path}"
    
    def stop(self):
        """Stop watching"""
        self.running = False
        return "Da dung theo doi"
    
    def _watch_loop(self, callback):
        """Watch loop"""
        while self.running:
            for f in self.folder_path.glob("*"):
                if f.suffix.lower() in ['.jpg', '.png', '.bmp'] and f.name not in self.processed_files:
                    self._process_file(f, callback)
            
            time.sleep(1)  # Check every second
    
    def _process_file(self, f, callback):
        """Process new file"""
        try:
            img = cv2.imread(str(f))
            if img is None:
                return
            
            detections = self.detector.detect(img)
            
            result = {
                'file': f.name,
                'detections': len(detections),
                'confidences': [d.confidence for d in detections],
                'timestamp': datetime.now().isoformat()
            }
            self.results.append(result)
            
            # Move to processed folder
            shutil.move(str(f), str(self.processed_folder / f.name))
            self.processed_files.add(f.name)
            
            # Call callback if provided
            if callback:
                callback(result)
                
        except Exception as e:
            print(f"Error processing {f}: {e}")
    
    def get_results(self):
        """Get all results"""
        return self.results


def create_demo():
    """Create interactive Gradio interface"""
    
    demo = InteractiveLPRDemo()
    
    with gr.Blocks(
        title="Vietnamese LPR - Interactive Demo",
        theme=gr.themes.Soft(),
    ) as app:
        
        gr.Markdown("""
        # Vietnamese LPR - Interactive Demo
        
        **YOLOv11** License Plate Detection
        
        ---
        
        Chon che do:
        - **Mode 1**: Upload / Drag & Drop anh
        - **Mode 2**: Theo doi thu muc tu dong
        - **Mode 3**: Webcam
        """)
        
        # ===== TAB 1: Upload Image =====
        with gr.TabItem("Upload / Drag & Drop"):
            gr.Markdown("### Kéo thả hoặc upload anh")
            
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        label="Upload anh",
                        type="numpy",
                        height=350,
                        sources=["upload", "clipboard"]  # Allow paste from clipboard
                    )
                    
                    conf_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.25,
                        step=0.05,
                        label="Confidence Threshold"
                    )
                    
                    submit_btn = gr.Button("Nhan dien", variant="primary", size="lg")
                    
                    gr.Examples(
                        examples=[
                            ["outputs/boderngoaigiao1_20260621_220652/detected.jpg"],
                        ],
                        inputs=image_input,
                    )
                
                with gr.Column():
                    image_output = gr.Image(
                        label="Ket qua",
                        interactive=False,
                        height=350
                    )
            
            results_output = gr.Markdown("*Chua co ket qua*")
        
        # ===== TAB 2: Folder Watch =====
        with gr.TabItem("Theo doi Thu muc"):
            gr.Markdown("""
            ### Tu dong xu ly anh moi
            
            Dat anh vao thu muc - he thong se tu dong nhan dien!
            """)
            
            folder_input = gr.Textbox(
                label="Duong dan thu muc",
                placeholder="VD: D:/images/input",
                value="outputs/watch_folder"
            )
            
            with gr.Row():
                start_watch_btn = gr.Button("Bat dau theo doi", variant="primary")
                stop_watch_btn = gr.Button("Dung theo doi", variant="secondary")
                process_folder_btn = gr.Button("Xu ly ngay", variant="secondary")
            
            watch_status = gr.Textbox(label="Trang thai", interactive=False)
            batch_results = gr.Textbox(label="Ket qua", lines=10, interactive=False)
            batch_preview = gr.Image(label="Preview", height=300)
            
            # Batch processing
            process_folder_btn.click(
                fn=demo.process_batch,
                inputs=[folder_input, conf_slider],
                outputs=[batch_results, batch_preview]
            )
        
        # ===== TAB 3: Webcam =====
        with gr.TabItem("Webcam"):
            gr.Markdown("### Chup anh tu webcam")
            
            webcam_input = gr.Image(
                label="Webcam",
                type="numpy",
                sources=["webcam"],
                height=350
            )
            
            webcam_conf = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.25,
                step=0.05,
                label="Confidence Threshold"
            )
            
            webcam_btn = gr.Button("Nhan dien tu Webcam", variant="primary", size="lg")
            
            webcam_result = gr.Image(label="Ket qua", interactive=False, height=350)
            webcam_text = gr.Markdown("*Chua co ket qua*")
        
        # ===== Connect Events =====
        # Tab 1: Upload
        submit_btn.click(
            fn=demo.process_single_image,
            inputs=[image_input, conf_slider],
            outputs=[image_output, results_output]
        )
        
        # Tab 1: Auto process on upload
        image_input.change(
            fn=demo.process_single_image,
            inputs=[image_input, conf_slider],
            outputs=[image_output, results_output]
        )
        
        # Tab 3: Webcam
        webcam_btn.click(
            fn=demo.process_single_image,
            inputs=[webcam_input, webcam_conf],
            outputs=[webcam_result, webcam_text]
        )
        
        # Info section
        gr.Markdown("""
        ---
        
        ### Huong dan su dung
        
        **Mode 1 - Upload:**
        1. Keo tha anh vao khung upload
        2. Hoac paste tu clipboard (Ctrl+V)
        3. Dieu chinh confidence neu can
        4. Nhan "Nhan dien" hoac tu dong xu ly
        
        **Mode 2 - Thu muc:**
        1. Tao thu muc chua anh (VD: D:/images/input)
        2. Nhap duong dan vao o "Duong dan thu muc"
        3. Nhan "Bat dau theo doi" de theo doi tu dong
        4. Copy anh vao thu muc - he thong tu dong xu ly
        
        **Mode 3 - Webcam:**
        1. Nhan nut "Capture" tren khung webcam
        2. Nhan "Nhan dien tu Webcam"
        
        ---
        
        *Model: weights/best.pt | Gradio Demo v2*
        """)
    
    return app


def main():
    print("=" * 60)
    print("Vietnamese LPR - Interactive Demo")
    print("=" * 60)
    print()
    print("Open: http://localhost:7860")
    print()
    
    app = create_demo()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
