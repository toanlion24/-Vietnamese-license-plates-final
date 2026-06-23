"""
Vietnamese License Plate Recognition (LPR) Demo - Full Pipeline
=============================================================
Integrates: YOLO11s (Detection) + Image Enhancement + PaddleOCR

Model trained on Google Colab:
- Precision: 99.76%
- Recall: 99.83%
- mAP@50: 99.48%
- mAP@50-95: 99.43%

Chay: python gradio_demo.py
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
from typing import List, Optional, Tuple, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

# Fix torch import order on Windows
import torch  # noqa: F401

# Import detector
spec = importlib.util.spec_from_file_location('detector', 'src/detection/detector.py')
detector_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detector_module)
PlateDetector = detector_module.PlateDetector
visualize_detections = detector_module.visualize_detections

# Import image enhancement
enhance_spec = importlib.util.spec_from_file_location('enhancer', 'src/modules/image_enhancement.py')
enhance_module = importlib.util.module_from_spec(enhance_spec)
enhance_spec.loader.exec_module(enhance_module)
ImageEnhancer = enhance_module.ImageEnhancer

# Import rule engine
rule_spec = importlib.util.spec_from_file_location('rule_engine', 'src/modules/rule_engine.py')
rule_module = importlib.util.module_from_spec(rule_spec)
rule_spec.loader.exec_module(rule_module)
PlateValidator = rule_module.PlateValidator

# Import PaddleOCR
from paddleocr import PaddleOCR

# Import Advanced OCR
import importlib.util
advanced_spec = importlib.util.spec_from_file_location(
    'advanced_ocr', 'src/modules/advanced_ocr.py'
)
advanced_module = importlib.util.module_from_spec(advanced_spec)
advanced_spec.loader.exec_module(advanced_module)
AdvancedLPROCRProcessor = advanced_module.AdvancedLPROCRProcessor


# Training metrics from Colab
TRAINING_METRICS = {
    "precision": "99.76%",
    "recall": "99.83%", 
    "mAP50": "99.48%",
    "mAP50_95": "99.43%",
    "epochs": 100,
    "batch_size": 16,
    "model": "YOLO11s"
}

MODEL_PATH = "weights/best.pt"


# ============================================================
# OCR PROCESSOR CLASS
# ============================================================

class LPROCRProcessor:
    """OCR Processor for License Plate Recognition using Advanced Ensemble OCR."""
    
    def __init__(self, use_advanced: bool = True):
        print("[INFO] Initializing OCR Processor...")
        self.use_advanced = use_advanced
        
        if use_advanced:
            self.advanced_ocr = AdvancedLPROCRProcessor(use_gpu=False)
            print("[OK] Advanced OCR Processor (Ensemble) initialized!")
        else:
            # Standard OCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=False,
                show_log=False,
                det_db_thresh=0.1,
                det_db_box_thresh=0.3,
            )
            
            # Import modules
            enhance_spec = importlib.util.spec_from_file_location(
                'enhancer', 'src/modules/image_enhancement.py'
            )
            enhance_module = importlib.util.module_from_spec(enhance_spec)
            enhance_spec.loader.exec_module(enhance_module)
            self.enhancer = enhance_module.ImageEnhancer()
            
            rule_spec = importlib.util.spec_from_file_location(
                'rule_engine', 'src/modules/rule_engine.py'
            )
            rule_module = importlib.util.module_from_spec(rule_spec)
            rule_spec.loader.exec_module(rule_module)
            self.validator = rule_module.PlateValidator()
            print("[OK] Standard OCR Processor initialized!")
    
    def process_plate(
        self,
        plate_image: np.ndarray,
        scale_factor: int = 4
    ) -> Dict[str, Any]:
        """
        Process a plate image and return OCR results.
        
        Args:
            plate_image: Cropped plate image
            scale_factor: Scale factor for upscaling (standard mode)
            
        Returns:
            Dictionary with OCR results
        """
        if self.use_advanced:
            # Use advanced ensemble OCR
            return self.advanced_ocr.process_ensemble(plate_image)
        
        # Standard OCR processing
        result = {
            'raw_text': '',
            'confidence': 0.0,
            'normalized_text': '',
            'is_valid': False,
            'plate_type': 'unknown',
            'province': None,
            'errors': []
        }
        
        if plate_image is None or plate_image.size == 0:
            result['errors'].append('Empty plate image')
            return result
        
        # Enhance image
        enhanced = self.enhancer.enhance_for_ocr(plate_image)
        
        # Resize for better OCR
        if len(enhanced.shape) == 2:
            h, w = enhanced.shape
        else:
            h, w = enhanced.shape[:2]
        
        new_w = w * scale_factor
        new_h = h * scale_factor
        
        resized = cv2.resize(
            enhanced,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )
        
        # Convert to BGR if grayscale
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        
        # Run OCR
        try:
            ocr_result = self.ocr.ocr(resized)
            
            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    if line:
                        text = line[1][0]
                        conf = line[1][1]
                        
                        # Validate plate
                        validation = self.validator.validate(text)
                        
                        result['raw_text'] = text
                        result['confidence'] = conf
                        result['normalized_text'] = validation.normalized_text
                        result['is_valid'] = validation.is_valid
                        result['plate_type'] = validation.plate_type.value
                        
                        if validation.is_valid:
                            province = self.validator.get_province(validation.normalized_text)
                            if province:
                                result['province'] = province
                        
                        if validation.errors:
                            result['errors'] = validation.errors
                        
                        break  # Take first result
            else:
                result['errors'].append('No text detected')
                
        except Exception as e:
            result['errors'].append(f'OCR error: {str(e)}')
        
        return result


# ============================================================
# MAIN LPR DEMO CLASS
# ============================================================

class InteractiveLPRDemo:
    """Interactive Gradio demo with full LPR pipeline"""
    
    def __init__(self, model_path: str = MODEL_PATH, conf_threshold: float = 0.25):
        self.model_path = model_path
        self.ocr_processor = LPROCRProcessor()
        self._check_and_load_model(conf_threshold)
        self.watch_folder = None
        self.watch_thread = None
        self.latest_results = []
    
    def _check_and_load_model(self, conf_threshold: float):
        """Check and load model with fallback options"""
        if Path(self.model_path).exists():
            print(f"[INFO] Loading Colab model: {self.model_path}")
            
            # Try CUDA first, fallback to CPU if not available
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                    print(f"[INFO] Using CUDA device")
                else:
                    device = "cpu"
                    print(f"[INFO] CUDA not available, using CPU")
            except:
                device = "cpu"
                print(f"[INFO] Using CPU device")
            
            self.detector = PlateDetector(
                model_path=self.model_path,
                confidence_threshold=conf_threshold,
                device=device
            )
            self.device = device
            print(f"[OK] Colab model loaded successfully!")
            print(f"     Model metrics: mAP50={TRAINING_METRICS['mAP50']}")
        else:
            print(f"[WARNING] Colab model not found at: {self.model_path}")
            # Try alternative paths
            alt_paths = [
                "weights/best.onnx",
                "models/yolov11/best.pt",
            ]
            found = False
            for alt_path in alt_paths:
                if Path(alt_path).exists():
                    print(f"[INFO] Using alternative model: {alt_path}")
                    self.model_path = alt_path
                    self.detector = PlateDetector(
                        model_path=alt_path,
                        confidence_threshold=conf_threshold,
                        device="cpu"
                    )
                    self.device = "cpu"
                    found = True
                    break
            if not found:
                print("[ERROR] No model found. Using detection-only mode.")
                self.detector = None
                self.device = "N/A"
    
    def _annotate_plate(
        self,
        img: np.ndarray,
        bbox: List[float],
        plate_text: str,
        plate_conf: float,
        ocr_conf: float,
        is_valid: bool,
    ) -> np.ndarray:
        """Draw bounding box + label for a detected plate."""
        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 255, 0) if is_valid else (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label = plate_text or "(no text)"
        status = "OK" if is_valid else "?"
        caption = f"{label} [{plate_conf:.0%}|{ocr_conf:.0%}|{status}]"
        
        (tw, th), baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        ty1 = max(0, y1 - th - baseline - 4)
        ty2 = y1
        cv2.rectangle(img, (x1, ty1), (x1 + tw + 4, ty2), color, -1)
        cv2.putText(
            img,
            caption,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        
        return img
    
    def process_full_lpr(
        self,
        image,
        conf_threshold: float = 0.25,
        scale_factor: int = 4,
        min_ocr_conf: float = 0.3
    ) -> Tuple:
        """
        Full LPR pipeline: YOLO Detection -> Image Enhancement -> PaddleOCR -> Validation
        
        Args:
            image: Input image (numpy array or path)
            conf_threshold: YOLO detection confidence
            scale_factor: Scale factor for OCR
            min_ocr_conf: Minimum OCR confidence threshold
            
        Returns:
            Tuple of (annotated_image, plates_gallery, results_text)
        """
        if image is None:
            return None, None, "Khong co anh duoc upload"
        
        try:
            img = image if isinstance(image, np.ndarray) else cv2.imread(image)
            if img is None:
                return None, None, "Khong the doc anh"
            
            # Result containers
            results_text = "=" * 55 + "\n"
            results_text += "KET QUA NHAN DIEN BIEN SO - YOLO + OCR PIPELINE\n"
            results_text += "=" * 55 + "\n\n"
            
            results_text += f"Model: YOLO11s (Colab) | Device: {self.device.upper()}\n"
            results_text += f"Training Metrics: P={TRAINING_METRICS['precision']} R={TRAINING_METRICS['recall']} mAP={TRAINING_METRICS['mAP50']}\n\n"
            
            plates_gallery = []
            annotated = img.copy()
            
            # Add info panel at top
            h, w = annotated.shape[:2]
            panel_height = 60
            panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
            panel[:] = (40, 40, 40)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"YOLO11s Colab | P={TRAINING_METRICS['precision']} | mAP={TRAINING_METRICS['mAP50']}"
            cv2.putText(panel, text, (10, 25), font, 0.6, (0, 255, 0), 2)
            cv2.putText(panel, f"OCR Scale: {scale_factor}x | Min Conf: {min_ocr_conf:.0%}", (10, 50), font, 0.4, (200, 200, 200), 1)
            
            # ============ STEP 1: YOLO DETECTION ============
            self.detector.confidence_threshold = conf_threshold
            detections = self.detector.detect(img, return_cropped=True)
            
            n_plates = len(detections)
            results_text += f"[YOLO] Phat hien: {n_plates} bien so\n"
            results_text += f"[YOLO] Confidence: {conf_threshold:.0%}\n\n"
            
            if n_plates == 0:
                results_text += "Khong tim thay bien so nao!\n"
                results_text += "Goi y:\n"
                results_text += "  - Giam confidence threshold\n"
                results_text += "  - Su dung anh ro net, bien so nhin thay ro\n"
                annotated = np.vstack([panel, annotated])
                return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), None, results_text
            
            results_text += "-" * 55 + "\n"
            results_text += "CHI TIET TUNG BIEN SO:\n"
            results_text += "-" * 55 + "\n\n"
            
            # ============ STEP 2: OCR EACH PLATE ============
            for idx, det in enumerate(detections, 1):
                x1, y1, x2, y2 = det.bbox
                
                # Get cropped plate
                if hasattr(det, 'cropped_image') and det.cropped_image is not None:
                    plate_crop = det.cropped_image
                else:
                    plate_crop = img[int(y1):int(y2), int(x1):int(x2)]
                
                if plate_crop is None or plate_crop.size == 0:
                    continue
                
                plates_gallery.append(plate_crop)
                
                # OCR the plate
                ocr_result = self.ocr_processor.process_plate(plate_crop, scale_factor=scale_factor)
                
                # Format results
                text = ocr_result['normalized_text'] or ocr_result.get('best_text', '') or ocr_result['raw_text'] or "(khong doc duoc)"
                ocr_conf = ocr_result['confidence']
                plate_conf = det.confidence
                ocr_method = ocr_result.get('method', 'standard')
                
                results_text += f"[{idx}] " + "=" * 45 + "\n"
                results_text += f"  [YOLO] Confidence:   {plate_conf:.1%}\n"
                results_text += f"  [OCR] Method:       {ocr_method}\n"
                results_text += f"  [OCR] Confidence:    {ocr_conf:.1%}\n"
                results_text += f"  [OCR] Text goc:     {ocr_result['raw_text']}\n"
                results_text += f"  [OCR] Text chuan:   {text}\n"
                results_text += f"  [OCR] Loai bien:    {ocr_result['plate_type']}\n"
                
                status = "HOP LE" if ocr_result['is_valid'] else "KHONG HOP LE"
                results_text += f"  [OCR] Trang thai:   {status}\n"
                
                if ocr_result['province']:
                    results_text += f"  [OCR] Tinh/TP:      {ocr_result['province']}\n"
                
                # Show top candidates if available
                if ocr_result.get('all_candidates'):
                    results_text += f"  [OCR] All candidates:\n"
                    for c in ocr_result['all_candidates'][:3]:
                        results_text += f"      - {c['method']}: {c['text']} ({c['conf']:.1%})\n"
                
                if ocr_result['errors']:
                    results_text += f"  [OCR] Loi:          {', '.join(ocr_result['errors'])}\n"
                
                results_text += "\n"
                
                # Annotate image
                self._annotate_plate(
                    annotated,
                    det.bbox,
                    text,
                    plate_conf,
                    ocr_conf,
                    ocr_result['is_valid']
                )
            
            # Add panel to image
            annotated = np.vstack([panel, annotated])
            
            # Summary
            results_text += "-" * 55 + "\n"
            results_text += "TOM TAT:\n"
            valid_count = sum(1 for det in detections if hasattr(det, 'cropped_image'))
            results_text += f"  Tong bien so:  {n_plates}\n"
            results_text += f"  Hop le:        {valid_count}\n"
            results_text += "-" * 55 + "\n"
            
            return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), plates_gallery, results_text
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, None, f"Loi: {str(e)}"
    
    def process_single_image(self, image, conf_threshold) -> Tuple:
        """Process single uploaded image (detection only mode)"""
        if image is None:
            return None, "Khong co anh duoc upload"
        
        try:
            img = image if isinstance(image, np.ndarray) else cv2.imread(image)
            if img is None:
                return None, "Khong the doc anh"
            
            self.detector.confidence_threshold = conf_threshold
            detections = self.detector.detect(img, return_cropped=True)
            
            # Enhanced visualization with info panel
            vis_img = visualize_detections(img.copy(), detections)
            
            h, w = vis_img.shape[:2]
            panel_height = 60
            panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
            panel[:] = (30, 30, 30)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"YOLO11s Colab | Detected: {len(detections)} | mAP: {TRAINING_METRICS['mAP50']}"
            cv2.putText(panel, text, (15, 30), font, 0.6, (0, 255, 0), 2)
            if detections:
                avg_conf = np.mean([d.confidence for d in detections])
                conf_text = f"Avg Confidence: {avg_conf:.1%}"
                cv2.putText(panel, conf_text, (15, 52), font, 0.4, (255, 255, 255), 1)
            
            result = np.vstack([panel, vis_img])
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            
            return result_rgb, self._format_results(detections)
            
        except Exception as e:
            return None, f"Loi: {str(e)}"
    
    def process_batch(self, folder_path: str, conf_threshold: float) -> Tuple:
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
        
        preview_img = visualize_detections(img.copy(), detections) if 'img' in locals() else None
        preview_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB) if preview_img is not None else None
        
        return results_text, preview_rgb, len(image_files)
    
    def _format_results(self, detections) -> str:
        """Format detection results with Colab metrics"""
        if not detections:
            return "**Ket qua:** Khong tim thay bien so nao!"
        
        text = f"### Ket qua nhan dien\n\n"
        text += f"| # | Toa do | Confidence | Kich thuoc |\n"
        text += f"|:--:|:------:|:----------:|:----------:|\n"
        
        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det.bbox
            text += f"| {i} | `[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]` | **{det.confidence:.1%}** | {int(x2-x1)}x{int(y2-y1)}px |\n"
        
        text += f"\n**Tong cong:** {len(detections)} bien so duoc phat hien\n"
        text += f"**Do chinh xac trung binh:** {np.mean([d.confidence for d in detections]):.1%}\n"
        
        return text


# ============================================================
# FOLDER WATCHER CLASS
# ============================================================

class FolderWatcher:
    """Watch folder for new images and auto-process"""
    
    def __init__(self, folder_path: str, detector, processed_folder: str):
        self.folder_path = Path(folder_path)
        self.processed_folder = Path(processed_folder)
        self.detector = detector
        self.processed_files = set()
        self.running = False
        self.results = []
        
        for f in self.folder_path.glob("*"):
            if f.suffix.lower() in ['.jpg', '.png', '.bmp']:
                self.processed_files.add(f.name)
        
        self.processed_folder.mkdir(exist_ok=True)
    
    def start(self, callback=None):
        self.running = True
        self.thread = threading.Thread(target=self._watch_loop, args=(callback,))
        self.thread.daemon = True
        self.thread.start()
        return f"Da bat dau theo doi: {self.folder_path}"
    
    def stop(self):
        self.running = False
        return "Da dung theo doi"
    
    def _watch_loop(self, callback):
        while self.running:
            for f in self.folder_path.glob("*"):
                if f.suffix.lower() in ['.jpg', '.png', '.bmp'] and f.name not in self.processed_files:
                    self._process_file(f, callback)
            time.sleep(1)
    
    def _process_file(self, f, callback):
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
            
            shutil.move(str(f), str(self.processed_folder / f.name))
            self.processed_files.add(f.name)
            
            if callback:
                callback(result)
                
        except Exception as e:
            print(f"Error processing {f}: {e}")
    
    def get_results(self):
        return self.results


# ============================================================
# CREATE GRADIO INTERFACE
# ============================================================

def create_demo():
    """Create interactive Gradio interface with full LPR pipeline"""
    
    demo = InteractiveLPRDemo()
    
    # Custom CSS
    css = """
    .gradio-container {max-width: 1400px !important; margin: auto !important;}
    .header-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 25px 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .header-box h1 {margin: 0; font-size: 28px; font-weight: bold;}
    .metrics-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 15px 20px;
        border-radius: 10px;
        color: white;
        margin: 15px 0;
    }
    .pipeline-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 15px 0;
    }
    """
    
    with gr.Blocks(css=css, title="Vietnamese LPR - Full Pipeline") as app:
        
        # Header
        gr.HTML("""
        <div class="header-box">
            <h1>Vietnamese LPR - Nhan dien Bien so xe</h1>
            <p>YOLO11s Detection + Advanced PaddleOCR (Ensemble) Recognition</p>
        </div>
        """)
        
        # Pipeline info
        gr.HTML("""
        <div class="pipeline-box">
            <h4 style="margin: 0 0 10px 0;">Pipeline xu ly (Ensemble OCR)</h4>
            <ol style="margin: 0; padding-left: 20px;">
                <li><strong>YOLO11s (Colab)</strong>: Phat hien va cat bien so tu anh</li>
                <li><strong>Advanced Preprocessing</strong>: 7 methods (2.5x-4x scale, gamma correction, grayscale)</li>
                <li><strong>Ensemble OCR</strong>: 2 OCR engines x 7 preprocessing = 14 combinations</li>
                <li><strong>Best Selection</strong>: Chon ket qua co confidence cao nhat</li>
                <li><strong>Rule Engine</strong>: Chuan hoa + kiem tra format bien so VN</li>
            </ol>
        </div>
        """)
        
        # Metrics box
        gr.HTML(f"""
        <div class="metrics-box">
            <h4 style="margin: 0 0 10px 0;">Model Performance (Trained on Colab)</h4>
            <div style="display: flex; justify-content: space-around; text-align: center;">
                <div>
                    <div style="font-size: 24px; font-weight: bold;">{TRAINING_METRICS['precision']}</div>
                    <div>Precision</div>
                </div>
                <div>
                    <div style="font-size: 24px; font-weight: bold;">{TRAINING_METRICS['recall']}</div>
                    <div>Recall</div>
                </div>
                <div>
                    <div style="font-size: 24px; font-weight: bold;">{TRAINING_METRICS['mAP50']}</div>
                    <div>mAP@50</div>
                </div>
                <div>
                    <div style="font-size: 24px; font-weight: bold;">{TRAINING_METRICS['mAP50_95']}</div>
                    <div>mAP@50-95</div>
                </div>
            </div>
        </div>
        """)
        
        with gr.Tabs():
            # ============ TAB 1: FULL LPR PIPELINE ============
            with gr.TabItem("Full LPR Pipeline (YOLO + OCR)"):
                gr.Markdown("### Nhan dien day du: Phat hien + Doc bien so")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="Upload anh",
                            type="numpy",
                            height=350,
                            sources=["upload", "clipboard"]
                        )
                        
                        with gr.Row():
                            conf_slider = gr.Slider(
                                minimum=0.1,
                                maximum=0.9,
                                value=0.25,
                                step=0.05,
                                label="YOLO Confidence"
                            )
                            scale_slider = gr.Slider(
                                minimum=1,
                                maximum=8,
                                value=4,
                                step=1,
                                label="OCR Scale Factor"
                            )
                        
                        min_conf_slider = gr.Slider(
                            minimum=0.1,
                            maximum=0.95,
                            value=0.3,
                            step=0.05,
                            label="Min OCR Confidence"
                        )
                        
                        submit_btn = gr.Button("Nhan dien (YOLO + OCR)", variant="primary", size="lg")
                        
                        gr.Examples(
                            examples=[
                                ["outputs/boderngoaigiao1_20260621_220652/detected.jpg"],
                            ],
                            inputs=image_input,
                        )
                    
                    with gr.Column(scale=1):
                        image_output = gr.Image(
                            label="Ket qua (bbox + text)",
                            interactive=False,
                            height=350
                        )
                
                with gr.Row():
                    plates_gallery = gr.Gallery(
                        label="Bien so da cat",
                        columns=3,
                        height=120,
                        object_fit="contain"
                    )
                    results_output = gr.Textbox(
                        label="Chi tiet ket qua",
                        lines=10,
                        interactive=False
                    )
            
            # ============ TAB 2: DETECTION ONLY ============
            with gr.TabItem("Detection Only"):
                gr.Markdown("### Chi phat hien bien so (khong doc text)")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        det_image_input = gr.Image(
                            label="Upload anh",
                            type="numpy",
                            height=350,
                            sources=["upload", "clipboard"]
                        )
                        
                        det_conf_slider = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.25,
                            step=0.05,
                            label="Confidence Threshold"
                        )
                        
                        det_submit_btn = gr.Button("Phat hien bien so", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        det_image_output = gr.Image(
                            label="Ket qua",
                            interactive=False,
                            height=350
                        )
                
                det_results_output = gr.Markdown("*Chua co ket qua*")
            
            # ============ TAB 3: FOLDER WATCH ============
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
        
        # Connect events for Full LPR Pipeline
        submit_btn.click(
            fn=demo.process_full_lpr,
            inputs=[image_input, conf_slider, scale_slider, min_conf_slider],
            outputs=[image_output, plates_gallery, results_output]
        )
        
        # Connect events for Detection Only
        det_submit_btn.click(
            fn=demo.process_single_image,
            inputs=[det_image_input, det_conf_slider],
            outputs=[det_image_output, det_results_output]
        )
        
        # Connect events for Folder Watch
        process_folder_btn.click(
            fn=demo.process_batch,
            inputs=[folder_input, conf_slider],
            outputs=[batch_results, batch_preview]
        )
        
        # Footer
        gr.HTML(f"""
        <div style="text-align: center; padding: 20px; color: #666; border-top: 1px solid #eee; margin-top: 30px;">
            <p><strong>Vietnamese LPR System</strong> | YOLO11s + PaddleOCR</p>
            <p>Model: {MODEL_PATH} | Device: {demo.device.upper()}</p>
            <p>Training Metrics: P={TRAINING_METRICS['precision']} | R={TRAINING_METRICS['recall']} | mAP@50={TRAINING_METRICS['mAP50']}</p>
        </div>
        """)
    
    return app


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Vietnamese LPR - Full Pipeline Demo")
    print("YOLO11s (Detection) + PaddleOCR (OCR)")
    print("=" * 60)
    print()
    print(f"Model: {MODEL_PATH}")
    print(f"Training Metrics:")
    print(f"  - Precision: {TRAINING_METRICS['precision']}")
    print(f"  - Recall: {TRAINING_METRICS['recall']}")
    print(f"  - mAP@50: {TRAINING_METRICS['mAP50']}")
    print(f"  - mAP@50-95: {TRAINING_METRICS['mAP50_95']}")
    print()
    print("Open: http://localhost:7860")
    print()
    
    app = create_demo()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
