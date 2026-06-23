"""
Module 5: Image Enhancement
Enhances license plate images for better OCR recognition
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnhancementConfig:
    """Configuration for image enhancement"""
    # CLAHE settings
    clahe_enabled: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    
    # Denoising
    denoise_enabled: bool = True
    denoise_strength: int = 10
    
    # Contrast enhancement
    contrast_enabled: bool = True
    contrast_alpha: float = 1.3  # 1.0 = no change
    contrast_beta: float = 10    # 0 = no change
    
    # Sharpening
    sharpen_enabled: bool = True
    sharpen_amount: float = 1.0
    
    # Histogram equalization
    hist_eq_enabled: bool = False
    
    # Adaptive thresholding
    adaptive_threshold: bool = False
    adaptive_blocksize: int = 11
    adaptive_c: int = 2


class ImageEnhancer:
    """
    Image enhancement for license plate OCR.
    
    Techniques:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Denoising (Non-local means)
    - Contrast enhancement
    - Sharpening
    - Grayscale conversion
    - Adaptive thresholding
    """
    
    def __init__(self, config: Optional[EnhancementConfig] = None):
        """
        Initialize enhancer.
        
        Args:
            config: Enhancement configuration
        """
        self.config = config or EnhancementConfig()
        self._clahe = None
        self._init_clahe()
    
    def _init_clahe(self):
        """Initialize CLAHE object"""
        if self.config.clahe_enabled:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_grid_size
            )
    
    def enhance(
        self,
        image: np.ndarray,
        to_grayscale: bool = True,
    ) -> np.ndarray:
        """
        Apply all enhancement techniques.
        
        Args:
            image: Input image
            to_grayscale: Convert to grayscale
            
        Returns:
            Enhanced image
        """
        if len(image.shape) == 2:
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img = image.copy()
        
        if self.config.denoise_enabled:
            img = self._denoise(img)
        
        if self.config.contrast_enabled:
            img = self._adjust_contrast(img)
        
        if self.config.sharpen_enabled:
            img = self._sharpen(img)
        
        if to_grayscale:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        if self.config.clahe_enabled and self._clahe is not None:
            gray = self._clahe.apply(gray)
        
        if self.config.adaptive_threshold:
            gray = self._adaptive_threshold(gray)
        
        if not to_grayscale:
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        return gray
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply non-local means denoising"""
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            self.config.denoise_strength,
            self.config.denoise_strength,
            7,
            21
        )
    
    def _adjust_contrast(self, image: np.ndarray) -> np.ndarray:
        """Adjust contrast using alpha-beta scaling"""
        return cv2.convertScaleAbs(
            image,
            alpha=self.config.contrast_alpha,
            beta=self.config.contrast_beta
        )
    
    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """Apply sharpening filter"""
        if self.config.sharpen_amount <= 0:
            return image
        
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ], dtype=np.float32) * self.config.sharpen_amount
        
        kernel[1, 1] = 1 + 8 * self.config.sharpen_amount
        
        return cv2.filter2D(image, -1, kernel)
    
    def _adaptive_threshold(self, gray: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding"""
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config.adaptive_blocksize,
            self.config.adaptive_c
        )
    
    def enhance_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Optimize image specifically for OCR.
        
        Args:
            image: Input plate image
            
        Returns:
            OCR-optimized image (grayscale)
        """
        # Simple, non-destructive enhancement for OCR
        # Too much processing can distort small text
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for contrast enhancement (gentle)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def enhance_low_light(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image for low-light conditions.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        config = EnhancementConfig(
            clahe_enabled=True,
            clahe_clip_limit=4.0,
            clahe_grid_size=(6, 6),
            denoise_enabled=True,
            denoise_strength=15,
            contrast_enabled=True,
            contrast_alpha=1.5,
            contrast_beta=20,
            sharpen_enabled=True,
            sharpen_amount=0.3,
        )
        
        enhancer = ImageEnhancer(config)
        return enhancer.enhance(image, to_grayscale=True)
    
    def enhance_night(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image for night-time conditions.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        config = EnhancementConfig(
            clahe_enabled=True,
            clahe_clip_limit=5.0,
            clahe_grid_size=(8, 8),
            denoise_enabled=True,
            denoise_strength=12,
            contrast_enabled=True,
            contrast_alpha=1.6,
            contrast_beta=25,
            sharpen_enabled=True,
            sharpen_amount=0.2,
        )
        
        enhancer = ImageEnhancer(config)
        return enhancer.enhance(image, to_grayscale=True)
    
    def enhance_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance blurry images.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    def remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """
        Remove shadows from image.
        
        Args:
            image: Input image
            
        Returns:
            Shadow-removed image
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
        
        if len(rgb.shape) == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        
        rgb = rgb.astype(np.float32)
        
        luminosity = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
        
        shadow_free = np.zeros_like(rgb)
        for i in range(3):
            channel = rgb[:,:,i].astype(np.float32)
            corrected = channel / (luminosity + 1e-6) * np.mean(luminosity)
            shadow_free[:,:,i] = np.clip(corrected, 0, 255)
        
        return shadow_free.astype(np.uint8)
    
    def remove_glare(self, image: np.ndarray, threshold: int = 200) -> np.ndarray:
        """
        Detect and reduce glare/gloss.
        
        Args:
            image: Input image
            threshold: Brightness threshold for glare detection
            
        Returns:
            Image with reduced glare
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        
        result = image.copy()
        blur = cv2.GaussianBlur(image, (21, 21), 0)
        result[mask > 0] = blur[mask > 0]
        
        return result
    
    def normalize_brightness(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image brightness.
        
        Args:
            image: Input image
            
        Returns:
            Brightness-normalized image
        """
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            return cv2.equalizeHist(image)


def auto_enhance(image: np.ndarray, quality: str = "balanced") -> np.ndarray:
    """
    Automatically enhance image based on quality preset.
    
    Args:
        image: Input image
        quality: 'fast', 'balanced', 'quality'
        
    Returns:
        Enhanced image
    """
    if quality == "fast":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    
    elif quality == "balanced":
        enhancer = ImageEnhancer(EnhancementConfig(
            clahe_enabled=True,
            clahe_clip_limit=2.5,
            clahe_grid_size=(6, 6),
            denoise_enabled=True,
            denoise_strength=8,
            contrast_enabled=True,
            contrast_alpha=1.3,
            contrast_beta=10,
            sharpen_enabled=True,
            sharpen_amount=0.3,
        ))
        return enhancer.enhance(image)
    
    else:  # quality
        enhancer = ImageEnhancer(EnhancementConfig(
            clahe_enabled=True,
            clahe_clip_limit=3.0,
            clahe_grid_size=(4, 4),
            denoise_enabled=True,
            denoise_strength=10,
            contrast_enabled=True,
            contrast_alpha=1.4,
            contrast_beta=15,
            sharpen_enabled=True,
            sharpen_amount=0.5,
        ))
        return enhancer.enhance(image)


def analyze_image_quality(image: np.ndarray) -> dict:
    """
    Analyze image quality metrics.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary of quality metrics
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    metrics = {}
    
    # Brightness
    metrics['brightness'] = float(np.mean(gray))
    
    # Contrast (std deviation)
    metrics['contrast'] = float(np.std(gray))
    
    # Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    metrics['sharpness'] = float(laplacian.var())
    
    # Blur detection
    metrics['is_blurry'] = metrics['sharpness'] < 100
    
    # Dynamic range
    metrics['dynamic_range'] = float(np.max(gray) - np.min(gray))
    
    # Histogram entropy
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    metrics['entropy'] = float(entropy)
    
    return metrics


def visualize_enhancement(
    original: np.ndarray,
    enhanced: np.ndarray,
    metrics_before: dict = None,
    metrics_after: dict = None,
) -> np.ndarray:
    """
    Create side-by-side comparison visualization.
    
    Args:
        original: Original image
        enhanced: Enhanced image
        metrics_before: Quality metrics before enhancement
        metrics_after: Quality metrics after enhancement
        
    Returns:
        Comparison visualization
    """
    h = max(original.shape[0], enhanced.shape[0])
    w = original.shape[1] + enhanced.shape[1] + 20
    
    canvas = np.ones((h + 60, w, 3), dtype=np.uint8) * 240
    
    canvas[:original.shape[0], :original.shape[1]] = original
    canvas[:enhanced.shape[0], original.shape[1]+20:] = enhanced
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(canvas, "Original", (10, h + 30), font, 0.6, (0, 0, 0), 2)
    cv2.putText(canvas, "Enhanced", (original.shape[1] + 30, h + 30), font, 0.6, (0, 0, 0), 2)
    
    if metrics_before and metrics_after:
        text_y = h + 50
        info = f"Sharpness: {metrics_before.get('sharpness', 0):.1f} -> {metrics_after.get('sharpness', 0):.1f}"
        cv2.putText(canvas, info, (10, text_y), font, 0.4, (100, 100, 100), 1)
    
    return canvas
