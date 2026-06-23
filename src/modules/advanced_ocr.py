"""
Module 8: Advanced OCR Processor for Vietnamese License Plates
============================================================
Tối ưu OCR với multiple preprocessing techniques để đạt độ chính xác cao nhất.

Techniques:
- Multi-scale interpolation (2.5x, 3x, 4x)
- CLAHE enhancement
- Gamma correction
- Sharpen filtering
- Auto-gamma optimization
- Ensemble OCR with multiple configs

Reference: Trained on Google Colab with YOLO11s (mAP@50: 99.48%)
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class AdvancedOCRPreprocessor:
    """
    Advanced image preprocessing for OCR to maximize accuracy.
    
    Based on techniques from Colab notebook:
    - Upscaling với INTER_CUBIC
    - CLAHE enhancement
    - Sharpening
    - Gamma correction
    """
    
    def __init__(self):
        self.supported_scales = [2.5, 3.0, 3.5, 4.0]
        self.supported_gammas = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    def enhance_colab_style(self, img: np.ndarray, scale: float = 2.5) -> np.ndarray:
        """
        Enhance image the same way as Colab notebook.
        
        Args:
            img: Input plate image
            scale: Scale factor (default 2.5 for Colab style)
            
        Returns:
            Enhanced image
        """
        # 1. Upscaling với Bicubic interpolation
        h, w = img.shape[:2]
        img = cv2.resize(
            img, 
            (int(w * scale), int(h * scale)), 
            interpolation=cv2.INTER_CUBIC
        )
        
        # 2. CLAHE trên kênh Value (HSV)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 3. Sharpen
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        img = cv2.filter2D(img, -1, kernel)
        
        return img
    
    def apply_gamma(self, img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Apply gamma correction.
        
        Args:
            img: Input image
            gamma: Gamma value (>1 brightens, <1 darkens)
            
        Returns:
            Gamma-corrected image
        """
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 
            for i in np.arange(0, 256)
        ]).astype("uint8")
        return cv2.LUT(img, table)
    
    def enhance_for_dark(self, img: np.ndarray, gamma: float = 3.0) -> np.ndarray:
        """
        Enhance image for dark/low-light conditions.
        
        Args:
            img: Input plate image
            gamma: Gamma value (higher = brighter)
            
        Returns:
            Enhanced image
        """
        # Upscale first
        img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        
        # Apply gamma correction
        img = self.apply_gamma(img, gamma)
        
        # Strong CLAHE
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
        v = clahe.apply(v)
        hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Sharpen
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)
        
        return img
    
    def enhance_for_blur(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance blurry images with aggressive methods.
        
        Args:
            img: Input plate image
            
        Returns:
            Enhanced image
        """
        # Upscale 4x for better detail
        h, w = img.shape[:2]
        img = cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Denoise (use grayscale version for gray input)
        denoised = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
        
        # 2. Deconvolution-inspired sharpening (Wiener-like)
        blur = cv2.GaussianBlur(denoised, (0, 0), 3)
        deconvolved = cv2.addWeighted(denoised, 1.5, blur, -0.5, 0)
        
        # 3. Unsharp masking for edge enhancement
        gaussian = cv2.GaussianBlur(deconvolved, (0, 0), 2.5)
        unsharp = cv2.addWeighted(deconvolved, 1.5, gaussian, -0.5, 0)
        
        # 4. Morphological operations to make characters thicker
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(unsharp, cv2.MORPH_CLOSE, kernel)
        
        # 5. Strong CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(morph)
        
        # 6. Binarize for OCR
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to BGR
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def enhance_for_low_light(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance image for low-light conditions.
        
        Args:
            img: Input plate image
            
        Returns:
            Enhanced image
        """
        # Upscale
        img = cv2.resize(img, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        
        # Apply gamma correction for brightness
        gamma = 2.5
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(img, table)
        
        # Convert to LAB color space for better contrast
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE on L channel
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
        l = clahe.apply(l)
        
        # Merge and convert back
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel)
        
        return img
    
    def enhance_adaptive(self, img: np.ndarray) -> np.ndarray:
        """
        Adaptive enhancement based on image characteristics.
        
        Args:
            img: Input plate image
            
        Returns:
            Enhanced image
        """
        # Upscale 4x
        h, w = img.shape[:2]
        img = cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate blur amount
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = laplacian_var < 100
        
        if is_blurry:
            # For blurry images: heavy denoising + sharpening
            denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
            
            # Blind deconvolution approximation
            blur = cv2.GaussianBlur(denoised, (0, 0), 4)
            sharp = cv2.addWeighted(denoised, 2.0, blur, -1.0, 0)
            
            # Morphological to thicken text
            kernel = np.ones((2, 2), np.uint8)
            morph = cv2.morphologyEx(sharp, cv2.MORPH_CLOSE, kernel)
            
            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(3, 3))
            enhanced = clahe.apply(morph)
        else:
            # For clear images: standard processing
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
            enhanced = clahe.apply(gray)
        
        # Adaptive binarization
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Invert if background is dark
        if np.mean(binary) > 127:
            binary = 255 - binary
        
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    def enhance_deblur(self, img: np.ndarray) -> np.ndarray:
        """
        Enhanced deblurring for motion-blurred images.
        Uses multiple techniques: Wiener deconvolution, Richardson-Lucy, etc.
        
        Args:
            img: Input plate image
            
        Returns:
            Deblurred image
        """
        # Upscale 4x
        h, w = img.shape[:2]
        img = cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Heavy denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 25, 7, 21)
        
        # 2. Bilateral filter to preserve edges while smoothing
        bilateral = cv2.bilateralFilter(denoised, 9, 75, 75)
        
        # 3. Sharpening with unsharp mask (strong)
        gaussian = cv2.GaussianBlur(bilateral, (0, 0), 3)
        sharpened = cv2.addWeighted(bilateral, 2.5, gaussian, -1.5, 0)
        
        # 4. Edge-enhanced sharpening
        edges = cv2.Canny(sharpened, 50, 150)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        sharpened_color = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.addWeighted(sharpened_color, 1.0, edges_color, 0.3, 0)
        
        # 5. Morphological to thicken characters
        kernel = np.ones((2, 3), np.uint8)
        morph = cv2.morphologyEx(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY), cv2.MORPH_CLOSE, kernel)
        
        # 6. Strong CLAHE
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(3, 3))
        enhanced = clahe.apply(morph)
        
        # 7. Otsu binarization
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    def enhance_wiener(self, img: np.ndarray, noise_var: float = 10.0) -> np.ndarray:
        """
        Wiener deconvolution for motion blur removal.
        
        Args:
            img: Input plate image
            noise_var: Estimated noise variance
            
        Returns:
            Deblurred image
        """
        # Upscale
        h, w = img.shape[:2]
        img = cv2.resize(img, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Estimate motion blur kernel (assume horizontal motion blur)
        size = 15
        kernel = np.zeros((size, size))
        kernel[size//2, :] = np.ones(size) / size
        
        # Simple deconvolution (Wiener approximation)
        # Using OpenCV's deconvolution
        deblur = cv2.divide(gray, cv2.GaussianBlur(gray, (size, size), 0), scale=255)
        
        # Clip values
        deblur = np.clip(deblur, 0, 255).astype(np.uint8)
        
        # Denoise result
        denoised = cv2.fastNlMeansDenoising(deblur, None, 15, 7, 21)
        
        # Sharpen
        kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharp = cv2.filter2D(denoised, -1, kernel_sharp)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(sharp)
        
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    def enhance_grayscale(self, img: np.ndarray) -> np.ndarray:
        """
        Enhanced grayscale preprocessing.
        
        Args:
            img: Input plate image
            
        Returns:
            Enhanced grayscale image
        """
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Upscale
        h, w = gray.shape
        gray = cv2.resize(gray, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)
        
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def find_best_gamma(self, img: np.ndarray, ocr_func, 
                        gammas: List[float] = None) -> Tuple[np.ndarray, float, float]:
        """
        Find optimal gamma value for best OCR confidence.
        
        Args:
            img: Input plate image
            ocr_func: OCR function that takes image and returns (text, confidence)
            gammas: List of gamma values to try
            
        Returns:
            Tuple of (best_enhanced_image, best_gamma, best_confidence)
        """
        if gammas is None:
            gammas = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        
        best_conf = 0.0
        best_gamma = 1.0
        best_img = img
        
        for gamma in gammas:
            # Apply gamma
            enhanced = self.apply_gamma(img, gamma)
            
            # Try OCR
            text, conf = ocr_func(enhanced)
            
            if conf > best_conf:
                best_conf = conf
                best_gamma = gamma
                best_img = enhanced
        
        return best_img, best_gamma, best_conf


class AdvancedLPROCRProcessor:
    """
    Advanced OCR Processor for License Plate Recognition.
    
    Uses multiple preprocessing techniques and ensemble OCR
    to achieve maximum accuracy.
    """
    
    def __init__(self, use_gpu: bool = False):
        print("[INFO] Initializing Advanced OCR Processor...")
        
        # Import PaddleOCR
        from paddleocr import PaddleOCR
        
        # Primary OCR with optimized settings
        self.ocr_primary = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=use_gpu,
            show_log=False,
            det_db_thresh=0.1,      # Low threshold for better detection
            det_db_box_thresh=0.3,  # Low threshold for better detection
            rec_algorithm='SVTR_LCNet',
        )
        
        # Secondary OCR with different settings
        self.ocr_secondary = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=use_gpu,
            show_log=False,
            det_db_thresh=0.15,
            det_db_box_thresh=0.4,
            rec_algorithm='CRNN',
        )
        
        # Initialize preprocessor
        self.preprocessor = AdvancedOCRPreprocessor()
        
        # Import modules
        import importlib.util
        rule_spec = importlib.util.spec_from_file_location(
            'rule_engine', 'src/modules/rule_engine.py'
        )
        rule_module = importlib.util.module_from_spec(rule_spec)
        rule_spec.loader.exec_module(rule_module)
        self.validator = rule_module.PlateValidator()
        
        print("[OK] Advanced OCR Processor initialized!")
    
    def _ocr_single(self, img: np.ndarray, ocr_engine) -> Tuple[str, float]:
        """
        Run OCR on single image.
        
        Args:
            img: Input image
            ocr_engine: PaddleOCR engine
            
        Returns:
            Tuple of (detected_text, confidence)
        """
        try:
            result = ocr_engine.ocr(img, cls=True)
            
            if result and result[0]:
                texts = [line[1][0] for line in result[0]]
                confs = [line[1][1] for line in result[0]]
                
                text = " ".join(texts)
                conf = sum(confs) / len(confs) if confs else 0.0
                
                return text.strip(), conf
            
            return "", 0.0
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return "", 0.0
    
    def process_ensemble(self, plate_img: np.ndarray) -> Dict[str, Any]:
        """
        Process plate using ensemble of preprocessing + OCR methods.
        
        Args:
            plate_img: Cropped plate image
            
        Returns:
            Dictionary with OCR results
        """
        result = {
            'raw_text': '',
            'best_text': '',
            'confidence': 0.0,
            'normalized_text': '',
            'is_valid': False,
            'plate_type': 'unknown',
            'province': None,
            'method': '',
            'all_candidates': [],
            'errors': []
        }
        
        if plate_img is None or plate_img.size == 0:
            result['errors'].append('Empty plate image')
            return result
        
        all_results = []
        
        # Method 1: Colab-style enhancement (2.5x)
        img1 = self.preprocessor.enhance_colab_style(plate_img, scale=2.5)
        text1, conf1 = self._ocr_single(img1, self.ocr_primary)
        if text1:
            all_results.append(('colab_2.5x', text1, conf1, img1))
        
        # Method 2: Colab-style (3x)
        img2 = self.preprocessor.enhance_colab_style(plate_img, scale=3.0)
        text2, conf2 = self._ocr_single(img2, self.ocr_primary)
        if text2:
            all_results.append(('colab_3.0x', text2, conf2, img2))
        
        # Method 3: Colab-style (4x)
        img3 = self.preprocessor.enhance_colab_style(plate_img, scale=4.0)
        text3, conf3 = self._ocr_single(img3, self.ocr_primary)
        if text3:
            all_results.append(('colab_4.0x', text3, conf3, img3))
        
        # Method 4: Dark condition (gamma 3.0)
        img4 = self.preprocessor.enhance_for_dark(plate_img, gamma=3.0)
        text4, conf4 = self._ocr_single(img4, self.ocr_primary)
        if text4:
            all_results.append(('dark_3.0', text4, conf4, img4))
        
        # Method 5: Dark condition (gamma 2.5)
        img5 = self.preprocessor.enhance_for_dark(plate_img, gamma=2.5)
        text5, conf5 = self._ocr_single(img5, self.ocr_primary)
        if text5:
            all_results.append(('dark_2.5', text5, conf5, img5))
        
        # Method 6: Grayscale
        img6 = self.preprocessor.enhance_grayscale(plate_img)
        text6, conf6 = self._ocr_single(img6, self.ocr_primary)
        if text6:
            all_results.append(('grayscale', text6, conf6, img6))
        
        # Method 7: Blur enhancement (NEW - for blurry images)
        img7 = self.preprocessor.enhance_for_blur(plate_img)
        text7, conf7 = self._ocr_single(img7, self.ocr_primary)
        if text7:
            all_results.append(('blur_fix', text7, conf7, img7))
        
        # Method 8: Low-light enhancement (NEW)
        img8 = self.preprocessor.enhance_for_low_light(plate_img)
        text8, conf8 = self._ocr_single(img8, self.ocr_primary)
        if text8:
            all_results.append(('low_light', text8, conf8, img8))
        
        # Method 9: Adaptive enhancement (NEW - auto-detect blur)
        img9 = self.preprocessor.enhance_adaptive(plate_img)
        text9, conf9 = self._ocr_single(img9, self.ocr_primary)
        if text9:
            all_results.append(('adaptive', text9, conf9, img9))
        
        # Method 10-11: Secondary OCR engine with different preps
        text10, conf10 = self._ocr_single(img2, self.ocr_secondary)
        if text10:
            all_results.append(('sec_colab_3x', text10, conf10, img2))
        
        text11, conf11 = self._ocr_single(img7, self.ocr_secondary)
        if text11:
            all_results.append(('sec_blur_fix', text11, conf11, img7))
        
        # Method 12: Deblur enhancement (NEW)
        img12 = self.preprocessor.enhance_deblur(plate_img)
        text12, conf12 = self._ocr_single(img12, self.ocr_primary)
        if text12:
            all_results.append(('deblur', text12, conf12, img12))
        
        # Method 13: Wiener deconvolution (NEW)
        img13 = self.preprocessor.enhance_wiener(plate_img)
        text13, conf13 = self._ocr_single(img13, self.ocr_primary)
        if text13:
            all_results.append(('wiener', text13, conf13, img13))
        
        # Select best result
        if all_results:
            # Sort by confidence
            all_results.sort(key=lambda x: x[2], reverse=True)
            best = all_results[0]
            
            result['method'] = best[0]
            result['best_text'] = best[1]
            result['confidence'] = best[2]
            result['raw_text'] = best[1]
            result['all_candidates'] = [
                {'method': r[0], 'text': r[1], 'conf': r[2]} 
                for r in all_results
            ]
            
            # Validate best result
            validation = self.validator.validate(best[1])
            result['normalized_text'] = validation.normalized_text
            result['is_valid'] = validation.is_valid
            result['plate_type'] = validation.plate_type.value
            
            if validation.is_valid:
                province = self.validator.get_province(validation.normalized_text)
                if province:
                    result['province'] = province
            
            if validation.errors:
                result['errors'] = validation.errors
            
            # If best is not valid, try other candidates
            if not validation.is_valid:
                for method, text, conf, _ in all_results[1:]:
                    val = self.validator.validate(text)
                    if val.is_valid:
                        result['best_text'] = text
                        result['confidence'] = conf
                        result['method'] = method
                        result['normalized_text'] = val.normalized_text
                        result['is_valid'] = True
                        result['plate_type'] = val.plate_type.value
                        if val.get_province:
                            result['province'] = val.get_province(val.normalized_text)
                        break
        
        return result
    
    def process_standard(self, plate_img: np.ndarray, scale: int = 4) -> Dict[str, Any]:
        """
        Standard processing with single preprocessing method.
        
        Args:
            plate_img: Cropped plate image
            scale: Scale factor
            
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
        
        if plate_img is None or plate_img.size == 0:
            result['errors'].append('Empty plate image')
            return result
        
        # Use Colab-style enhancement
        enhanced = self.preprocessor.enhance_colab_style(plate_img, scale=float(scale))
        
        # Run OCR
        text, conf = self._ocr_single(enhanced, self.ocr_primary)
        
        result['raw_text'] = text
        result['confidence'] = conf
        
        # Validate
        validation = self.validator.validate(text)
        result['normalized_text'] = validation.normalized_text
        result['is_valid'] = validation.is_valid
        result['plate_type'] = validation.plate_type.value
        
        if validation.is_valid:
            province = self.validator.get_province(validation.normalized_text)
            if province:
                result['province'] = province
        
        if validation.errors:
            result['errors'] = validation.errors
        
        return result


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_advanced_ocr_processor(use_gpu: bool = False) -> AdvancedLPROCRProcessor:
    """
    Create advanced OCR processor instance.
    
    Args:
        use_gpu: Whether to use GPU for OCR
        
    Returns:
        AdvancedLPROCRProcessor instance
    """
    return AdvancedLPROCRProcessor(use_gpu=use_gpu)


def process_plate_advanced(plate_img: np.ndarray, 
                           use_gpu: bool = False) -> Dict[str, Any]:
    """
    Convenience function for advanced plate OCR.
    
    Args:
        plate_img: Cropped plate image
        use_gpu: Whether to use GPU
        
    Returns:
        OCR results dictionary
    """
    processor = AdvancedLPROCRProcessor(use_gpu=use_gpu)
    return processor.process_ensemble(plate_img)


def process_plate_standard(plate_img: np.ndarray, 
                           scale: int = 4,
                           use_gpu: bool = False) -> Dict[str, Any]:
    """
    Convenience function for standard plate OCR.
    
    Args:
        plate_img: Cropped plate image
        scale: Scale factor
        use_gpu: Whether to use GPU
        
    Returns:
        OCR results dictionary
    """
    processor = AdvancedLPROCRProcessor(use_gpu=use_gpu)
    return processor.process_standard(plate_img, scale)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Test with sample image if available
    test_img_path = "outputs/boderngoaigiao1_20260621_220652/detected.jpg"
    
    if Path(test_img_path).exists():
        print(f"Testing with: {test_img_path}")
        
        img = cv2.imread(test_img_path)
        if img is not None:
            # Crop a region for testing
            h, w = img.shape[:2]
            crop = img[h//4:3*h//4, w//4:3*w//4]
            
            # Standard processing
            print("\n=== Standard Processing ===")
            result = process_plate_standard(crop, scale=4)
            print(f"Text: {result['raw_text']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Valid: {result['is_valid']}")
            print(f"Normalized: {result['normalized_text']}")
            
            # Advanced processing
            print("\n=== Advanced Processing (Ensemble) ===")
            result = process_plate_advanced(crop)
            print(f"Method: {result['method']}")
            print(f"Best Text: {result['best_text']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Valid: {result['is_valid']}")
            print(f"All Candidates:")
            for c in result['all_candidates'][:5]:
                print(f"  {c['method']}: {c['text']} ({c['conf']:.2%})")
    else:
        print(f"Test image not found: {test_img_path}")
        print("Run with: python src/modules/advanced_ocr.py")
