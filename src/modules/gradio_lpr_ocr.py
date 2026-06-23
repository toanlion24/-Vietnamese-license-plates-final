# -*- coding: utf-8 -*-
"""
Gradio LPR Demo with YOLO Detection + OCR
==========================================
Full License Plate Recognition pipeline:
  YOLOv11 (Detection)  ->  Image Enhancement  ->  PaddleOCR (Recognition)

Chay: python src/modules/gradio_lpr_ocr.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import os

# Fix torch import order on Windows
import torch  # noqa: F401

# PaddleOCR
from paddleocr import PaddleOCR

# Local modules (import directly to avoid torch conflicts)
import importlib.util


def load_module(name, path):
    """Load module directly to avoid torch conflicts."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load enhancement module
enhance_module = load_module('image_enhancement', 'src/modules/image_enhancement.py')
ImageEnhancer = enhance_module.ImageEnhancer

# Load rule engine
rule_module = load_module('rule_engine', 'src/modules/rule_engine.py')
PlateValidator = rule_module.PlateValidator

# Load YOLO interactive detector
yolo_spec = importlib.util.spec_from_file_location(
    'yolo_interactive', 'src/modules/yolo_interactive.py'
)
yolo_module = importlib.util.module_from_spec(yolo_spec)
yolo_spec.loader.exec_module(yolo_module)
YOLOInteractiveDetector = yolo_module.YOLOInteractiveDetector
DetectionMode = yolo_module.DetectionMode

# Load YOLO detector (use weights/best.pt if exists, else fallback)
YOLO_MODEL_PATH = "weights/best.pt"
if not Path(YOLO_MODEL_PATH).exists():
    YOLO_MODEL_PATH = "yolov8n.pt"  # Fallback to pretrained


# ============================================================
# OCR PROCESSOR
# ============================================================

class LPROCRProcessor:
    """OCR Processor for License Plate Recognition."""

    def __init__(self):
        print("Initializing OCR Processor...")
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='vi',
            use_gpu=False,
            show_log=False
        )
        self.enhancer = ImageEnhancer()
        self.validator = PlateValidator()
        print("OCR Processor initialized!")

    def process_plate(
        self,
        plate_image: np.ndarray,
        scale_factor: int = 4
    ) -> dict:
        """
        Process a plate image and return OCR results.

        Args:
            plate_image: Cropped plate image
            scale_factor: Scale factor for upscaling small images

        Returns:
            Dictionary with OCR results
        """
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
# GRADIO APP
# ============================================================

class LPRGradioApp:
    """Gradio App for LPR with YOLO Detection + OCR."""

    def __init__(self):
        self.ocr_processor = LPROCRProcessor()
        self.yolo_detector = None
        self.yolo_available = False
        self._init_yolo()

    def _init_yolo(self):
        """Initialize YOLO detector with CUDA/CPU fallback."""
        for device in ("cuda", "cpu"):
            try:
                self.yolo_detector = YOLOInteractiveDetector(
                    model_path=YOLO_MODEL_PATH,
                    confidence=0.25,
                    device=device,
                    warmup=False,
                    detection_mode=DetectionMode.PLATE_ONLY,
                )
                if self.yolo_detector._model is not None:
                    self.yolo_available = True
                    self.yolo_device = device
                    print(f"YOLO detector ready ({device.upper()})")
                    return
            except Exception as e:
                print(f"YOLO init failed on {device}: {e}")

        print("YOLO detector unavailable - falling back to OCR-only mode")
        self.yolo_available = False

    def _ensure_bgr(self, image):
        """Normalize image to BGR ndarray."""
        if isinstance(image, dict):
            image = image.get('composite') or image.get('image')

        if not isinstance(image, np.ndarray):
            return None

        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image.copy()

    def _annotate_plate(
        self,
        img: np.ndarray,
        bbox,
        plate_text: str,
        plate_conf: float,
        ocr_conf: float,
        is_valid: bool,
    ) -> None:
        """Draw bounding box + label for a detected plate."""
        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 255, 0) if is_valid else (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = plate_text or "(no text)"
        status = "OK" if is_valid else "?"
        caption = f"{label} [{plate_conf:.0%}|{ocr_conf:.0%}|{status}]"

        (tw, th), baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        ty1 = max(0, y1 - th - baseline - 4)
        ty2 = y1
        cv2.rectangle(img, (x1, ty1), (x1 + tw + 4, ty2), color, -1)
        cv2.putText(
            img,
            caption,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    def process_image(
        self,
        image,
        scale_factor: int,
        min_confidence: float,
        yolo_confidence: float,
        enable_yolo: bool,
    ):
        """
        Process uploaded image with full LPR pipeline.

        Pipeline:
            YOLO detection -> crop plates -> enhancement -> PaddleOCR -> validation

        Returns:
            Tuple of (annotated_image, plates_gallery, results_text)
        """
        img = self._ensure_bgr(image)
        if img is None:
            return None, None, "Vui long tai len mot hinh anh"

        # Result containers
        results_text = "=" * 50 + "\n"
        results_text += "KET QUA NHAN DIEN BIEN SO\n"
        results_text += "=" * 50 + "\n\n"

        plates_gallery = []
        annotated = img.copy()

        use_yolo = enable_yolo and self.yolo_available

        if use_yolo:
            results_text += (
                f"Che do: YOLOv11 + PaddleOCR "
                f"(device={self.yolo_device.upper()})\n\n"
            )

            # Step 1: Detect plates with YOLO
            self.yolo_detector.set_confidence(yolo_confidence)
            self.yolo_detector.set_mode(DetectionMode.PLATE_ONLY)

            det = self.yolo_detector.detect(
                img,
                return_cropped=True,
                return_visualized=False,
                source="gradio_upload",
            )
            detection_result, cropped_plates = det

            n_plates = len(detection_result.plates)
            results_text += f"YOLO detect: {n_plates} bien so\n\n"

            if n_plates == 0:
                results_text += (
                    "Khong tim thay bien so nao. Goi y:\n"
                    "   - Giam YOLO confidence (hien tai "
                    f"{yolo_confidence:.2f})\n"
                    "   - Su dung anh ro net, bien so nhin thay ro\n"
                )
                return annotated, None, results_text

            # Step 2: OCR each cropped plate
            for idx, (plate_bbox, crop) in enumerate(
                zip(detection_result.plates, cropped_plates), start=1
            ):
                if crop is None or crop.size == 0:
                    continue

                plates_gallery.append(crop)

                ocr = self.ocr_processor.process_plate(
                    crop, scale_factor=scale_factor
                )

                if ocr['confidence'] < min_confidence and not ocr['is_valid']:
                    results_text += (
                        f"[{idx}] YOLO={plate_bbox.confidence:.0%} "
                        f"OCR={ocr['confidence']:.0%} "
                        f"(bo qua, OCR conf < {min_confidence:.2f})\n"
                    )
                    self._annotate_plate(
                        annotated,
                        plate_bbox.xyxy,
                        ocr.get('normalized_text') or ocr.get('raw_text') or "",
                        plate_bbox.confidence,
                        ocr['confidence'],
                        ocr['is_valid'],
                    )
                    continue

                text = ocr['normalized_text'] or ocr['raw_text']
                self._annotate_plate(
                    annotated,
                    plate_bbox.xyxy,
                    text,
                    plate_bbox.confidence,
                    ocr['confidence'],
                    ocr['is_valid'],
                )

                results_text += f"[{idx}] " + "=" * 40 + "\n"
                results_text += f"  YOLO conf:    {plate_bbox.confidence:.1%}\n"
                results_text += f"  OCR conf:     {ocr['confidence']:.1%}\n"
                results_text += (
                    f"  Bbox:         "
                    f"[{plate_bbox.x1:.0f}, {plate_bbox.y1:.0f}, "
                    f"{plate_bbox.x2:.0f}, {plate_bbox.y2:.0f}]\n"
                )
                results_text += f"  Text goc:     {ocr['raw_text']}\n"
                results_text += f"  Text chuan:   {ocr['normalized_text']}\n"
                results_text += f"  Loai:         {ocr['plate_type']}\n"
                status = "Hop le" if ocr['is_valid'] else "Khong hop le"
                results_text += f"  Trang thai:   {status}\n"
                if ocr['province']:
                    results_text += f"  Tinh/TP:      {ocr['province']}\n"
                if ocr['errors']:
                    results_text += (
                        f"  Canh bao:     {', '.join(ocr['errors'])}\n"
                    )
                results_text += "\n"

            return annotated, plates_gallery, results_text

        # Fallback: OCR-only mode (no YOLO)
        results_text += "Che do: OCR truc tiep (YOLO khong kha dung)\n"
        results_text += (
            "De su dung day du, hay upload anh da cat bien so "
            "hoac khoi dong lai YOLO detector.\n\n"
        )

        ocr = self.ocr_processor.process_plate(img, scale_factor=scale_factor)

        if ocr['raw_text']:
            text = ocr['normalized_text']
            conf = ocr['confidence']
            conf_pct = f"{conf*100:.1f}%"

            label = f"{text}"
            if ocr['province']:
                label += f" | {ocr['province']}"
            label += f" ({conf_pct})"

            cv2.putText(
                annotated,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            status = "Hop le" if ocr['is_valid'] else "Khong hop le"
            cv2.putText(
                annotated,
                f"Type: {ocr['plate_type']} | {status}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            results_text += "KET QUA OCR:\n\n"
            results_text += f"  Text goc:     {ocr['raw_text']}\n"
            results_text += f"  Text chuan:   {ocr['normalized_text']}\n"
            results_text += f"  Confidence:   {conf_pct}\n"
            results_text += f"  Loai:         {ocr['plate_type']}\n"

            if ocr['province']:
                results_text += f"  Tinh/TP:      {ocr['province']}\n"

            if ocr['errors']:
                results_text += f"  Canh bao:     {', '.join(ocr['errors'])}\n"

            plates_gallery.append(img)

            return annotated, plates_gallery, results_text

        results_text += "Khong phat hien text trong anh\n"
        results_text += "\nGoi y:\n"
        results_text += "   - Su dung anh co bien so ro rang\n"
        results_text += "   - Dam bao bien so khong bi mo\n"
        results_text += "   - Bat YOLO detector de tu dong cat bien so\n\n"

        return annotated, None, results_text

    def load_demo1(self):
        """Load demo image 1."""
        img = cv2.imread("outputs/boderngoaigiao1_20260621_220652/original.jpg")
        return img

    def load_demo2(self):
        """Load demo image 2."""
        img = cv2.imread("outputs/boderngoaigiao1_20260621_220652/plate_01_crop.jpg")
        return img


# ============================================================
# CREATE GRADIO INTERFACE
# ============================================================

def create_lpr_demo():
    """Create the LPR Gradio demo interface."""

    app = LPRGradioApp()

    yolo_status = (
        f"YOLO: SAN SANG ({app.yolo_device.upper()})"
        if app.yolo_available
        else "YOLO: KHONG KHA DUNG (OCR-only fallback)"
    )

    # Custom CSS
    css = """
    .gradio-container {max-width: 1200px !important; margin: auto !important;}
    .header-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
    }
    .header-box h1 {margin: 0; font-size: 32px; font-weight: bold;}
    .header-box p {margin: 10px 0 0 0; opacity: 0.9; font-size: 16px;}
    """

    with gr.Blocks(css=css, title="Vietnamese LPR - YOLO + OCR") as demo:

        # Header
        gr.HTML("""
        <div class="header-box">
            <h1>Vietnamese LPR - Nhan dien Bien so xe</h1>
            <p>YOLOv11 Detection &rarr; Image Enhancement &rarr; PaddleOCR Recognition</p>
        </div>
        """)

        # Info box
        gr.HTML(f"""
        <div style="background: #e7f3ff; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #2196F3;">
            <h4 style="margin: 0 0 10px 0;">Pipeline tich hop</h4>
            <ul style="margin: 0; padding-left: 20px;">
                <li><strong>YOLOv11:</strong> phat hien va crop bien so tu anh dau vao</li>
                <li><strong>Image Enhancement:</strong> CLAHE + denoise truoc khi OCR</li>
                <li><strong>PaddleOCR (vi):</strong> nhan dien ky tu tieng Viet</li>
                <li><strong>Rule Engine:</strong> chuan hoa + validate format bien so</li>
            </ul>
            <p style="margin: 10px 0 0 0;"><strong>{yolo_status}</strong></p>
        </div>
        """)

        with gr.Row():
            # Left Panel - Input
            with gr.Column(scale=1):
                gr.Markdown("### Tai len hinh anh")

                # Image input
                image_input = gr.Image(
                    label="Chon anh bien so",
                    height=350,
                    type="numpy",
                    sources=["upload", "clipboard"],
                )

                gr.Markdown("**Cai dat YOLO Detection**")
                with gr.Row():
                    enable_yolo = gr.Checkbox(
                        label="Bat YOLO Detection",
                        value=app.yolo_available,
                        interactive=app.yolo_available,
                        info="Tu dong cat bien so truoc khi OCR",
                    )
                    yolo_conf_slider = gr.Slider(
                        minimum=0.05,
                        maximum=0.95,
                        value=0.25,
                        step=0.05,
                        label="YOLO Confidence",
                        info="Nguong confidence YOLO (0.05-0.95)",
                    )

                gr.Markdown("**Cai dat OCR**")
                with gr.Row():
                    scale_slider = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=4,
                        step=1,
                        label="Scale Factor",
                        info="Phong to anh truoc khi OCR (1-8)",
                    )
                    min_conf_slider = gr.Slider(
                        minimum=0.3,
                        maximum=0.95,
                        value=0.5,
                        step=0.05,
                        label="Min OCR Confidence",
                        info="Nguong OCR confidence toi thieu",
                    )

                # Process button
                process_btn = gr.Button(
                    "Nhan dien bien so",
                    variant="primary",
                    size="lg",
                )

                # Demo buttons
                gr.Markdown("### Demo Images")
                with gr.Row():
                    demo_btn1 = gr.Button("Anh Demo 1 (Full)", size="sm")
                    demo_btn2 = gr.Button("Anh Demo 2 (Crop)", size="sm")

                # Status
                status_box = gr.Textbox(
                    label="Trang thai",
                    value=yolo_status,
                    interactive=False,
                )

            # Right Panel - Output
            with gr.Column(scale=1):
                gr.Markdown("### Ket qua nhan dien")

                # Annotated image
                output_image = gr.Image(
                    label="Anh da xu ly (bbox + OCR text)",
                    height=300,
                    type="numpy",
                )

                # Plates gallery
                plates_gallery = gr.Gallery(
                    label="Bien so da cat (YOLO crops)",
                    columns=3,
                    height=150,
                    object_fit="contain",
                )

                # Results text
                results_text = gr.Textbox(
                    label="Chi tiet ket qua",
                    lines=12,
                    interactive=False,
                    show_label=True,
                )

        # Connect events
        process_btn.click(
            fn=app.process_image,
            inputs=[
                image_input,
                scale_slider,
                min_conf_slider,
                yolo_conf_slider,
                enable_yolo,
            ],
            outputs=[output_image, plates_gallery, results_text],
        )

        # Demo button handlers
        demo_btn1.click(
            fn=app.load_demo1,
            outputs=[image_input],
        )

        demo_btn2.click(
            fn=app.load_demo2,
            outputs=[image_input],
        )

        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: #666; border-top: 1px solid #eee; margin-top: 30px;">
            <p><strong>Vietnamese LPR System</strong> | Module 2 (YOLOv11) + Module 6 (PaddleOCR)</p>
            <p>YOLOv11 detection &rarr; CLAHE enhancement &rarr; PaddleOCR &rarr; Rule validation</p>
        </div>
        """)

    return demo


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Gradio LPR Demo - YOLO + OCR Pipeline")
    print("=" * 60)
    print()
    print("URL: http://localhost:7870")
    print("Or use the public URL shown below")
    print()

    demo = create_lpr_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7870,
        share=False,
    )
