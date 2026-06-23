"""
Module 4: Rectify & Perspective Correction
Crops and straightens license plates from skewed images
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RectificationResult:
    """Result of plate rectification"""
    image: np.ndarray
    corners: np.ndarray  # Detected corners
    source_corners: Optional[np.ndarray] = None
    transform_matrix: Optional[np.ndarray] = None
    quality_score: float = 1.0
    width: int = 0
    height: int = 0
    
    def __post_init__(self):
        if self.width == 0 and self.image is not None:
            self.height, self.width = self.image.shape[:2]


@dataclass
class PlateTemplate:
    """Standard plate template dimensions"""
    # Vietnamese license plate dimensions (width:height ratio)
    CAR_WIDTH = 480
    CAR_HEIGHT = 140
    CAR_RATIO = CAR_WIDTH / CAR_HEIGHT  # ~3.43
    
    MOTORCYCLE_WIDTH = 280
    MOTORCYCLE_HEIGHT = 110
    MOTORCYCLE_RATIO = MOTORCYCLE_WIDTH / MOTORCYCLE_HEIGHT  # ~2.55
    
    @classmethod
    def get_dimensions(cls, plate_type: str = "car") -> Tuple[int, int]:
        if plate_type == "motorcycle":
            return cls.MOTORCYCLE_WIDTH, cls.MOTORCYCLE_HEIGHT
        return cls.CAR_WIDTH, cls.CAR_HEIGHT


class PlateRectifier:
    """
    Rectifies and normalizes license plate images.
    
    Handles:
    - Perspective correction
    - Skew angle detection and correction
    - Cropping with padding
    - Size normalization
    """
    
    def __init__(
        self,
        target_width: int = 480,
        target_height: int = 140,
        padding_percent: float = 0.1,
        auto_detect_corners: bool = True,
    ):
        """
        Initialize rectifier.
        
        Args:
            target_width: Target output width
            target_height: Target output height
            padding_percent: Padding around plate (0-1)
            auto_detect_corners: Automatically detect plate corners
        """
        self.target_width = target_width
        self.target_height = target_height
        self.padding_percent = padding_percent
        self.auto_detect_corners = auto_detect_corners
    
    def rectify(
        self,
        image: np.ndarray,
        bbox: List[float],
        corners: Optional[np.ndarray] = None,
        plate_type: str = "car",
    ) -> RectificationResult:
        """
        Rectify plate image.
        
        Args:
            image: Source image
            bbox: Bounding box [x1, y1, x2, y2]
            corners: Optional corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            plate_type: Type of plate ('car' or 'motorcycle')
            
        Returns:
            RectificationResult with rectified image
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding
        pad_x = int((x2 - x1) * self.padding_percent)
        pad_y = int((y2 - y1) * self.padding_percent)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(image.shape[1], x2 + pad_x)
        y2 = min(image.shape[0], y2 + pad_y)
        
        cropped = image[y1:y2, x1:x2]
        
        if corners is not None and len(corners) == 4:
            adjusted_corners = corners.copy()
            adjusted_corners[:, 0] -= x1
            adjusted_corners[:, 1] -= y1
            
            rectified, matrix = self._perspective_transform(
                cropped, adjusted_corners
            )
            
            return RectificationResult(
                image=rectified,
                corners=adjusted_corners,
                source_corners=corners,
                transform_matrix=matrix,
                width=rectified.shape[1],
                height=rectified.shape[0],
            )
        
        return RectificationResult(
            image=cropped,
            corners=np.array([[0, 0], [cropped.shape[1], 0], 
                             [cropped.shape[1], cropped.shape[0]], [0, cropped.shape[0]]]),
            width=cropped.shape[1],
            height=cropped.shape[0],
        )
    
    def _perspective_transform(
        self,
        image: np.ndarray,
        corners: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply perspective transform to straighten plate.
        
        Args:
            image: Input plate image
            corners: 4 corner points
            
        Returns:
            Tuple of (transformed image, transform matrix)
        """
        corners = np.float32(corners)
        
        # Sort corners: top-left, top-right, bottom-right, bottom-left
        corners_sorted = self._sort_corners(corners)
        
        # Calculate output dimensions
        width = self.target_width
        height = self.target_height
        
        # Define destination points
        dst = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])
        
        # Get perspective transform matrix
        matrix = cv2.getPerspectiveTransform(corners_sorted, dst)
        
        # Apply transform
        rectified = cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        
        return rectified, matrix
    
    def _sort_corners(self, corners: np.ndarray) -> np.ndarray:
        """Sort corners in order: top-left, top-right, bottom-right, bottom-left"""
        corners = corners.copy()
        
        # Sort by sum (x + y) - smallest is top-left
        sorted_indices = np.argsort(corners.sum(axis=1))
        top_left = corners[sorted_indices[0]]
        bottom_right = corners[sorted_indices[3]]
        
        # Sort by difference (x - y) - smallest is top-right
        remaining = np.delete(corners, [sorted_indices[0], sorted_indices[3]], axis=0)
        top_right = remaining[np.argmin(remaining[:, 0] - remaining[:, 1])]
        bottom_left = remaining[np.argmax(remaining[:, 0] - remaining[:, 1])]
        
        return np.float32([top_left, top_right, bottom_right, bottom_left])
    
    def detect_corners(
        self,
        image: np.ndarray,
        method: str = "contour",
    ) -> Optional[np.ndarray]:
        """
        Detect plate corners in image.
        
        Args:
            image: Plate image
            method: Detection method ('contour', 'hough', 'edge')
            
        Returns:
            Corner points or None if not detected
        """
        if method == "contour":
            return self._detect_corners_contour(image)
        elif method == "hough":
            return self._detect_corners_hough(image)
        elif method == "edge":
            return self._detect_corners_edge(image)
        return None
    
    def _detect_corners_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect corners using contour approximation"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) >= 4:
            corners = approx[:4].reshape(-1, 2).astype(np.float32)
            return self._sort_corners(corners)
        
        return None
    
    def _detect_corners_hough(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect corners using Hough line transform"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=50,
            maxLineGap=10
        )
        
        if lines is None or len(lines) < 4:
            return None
        
        # Find intersection points
        intersections = []
        line_segments = [(l[0][0], l[0][1], l[0][2], l[0][3]) for l in lines]
        
        for i, (x1, y1, x2, y2) in enumerate(line_segments):
            for x3, y3, x4, y4 in line_segments[i+1:]:
                point = self._line_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
                if point is not None:
                    if 0 <= point[0] <= image.shape[1] and 0 <= point[1] <= image.shape[0]:
                        intersections.append(point)
        
        if len(intersections) >= 4:
            intersections = np.array(intersections, dtype=np.float32)
            
            # Keep 4 corner points (extreme positions)
            hull = cv2.convexHull(intersections.astype(np.float32))
            return self._sort_corners(hull.reshape(-1, 2))
        
        return None
    
    def _detect_corners_edge(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect corners using edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        edges = cv2.Canny(gray, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        corners = cv2.boxPoints(rect)
        
        return self._sort_corners(corners.astype(np.float32))
    
    def _line_intersection(
        self,
        x1: float, y1: float, x2: float, y2: float,
        x3: float, y3: float, x4: float, y4: float,
    ) -> Optional[Tuple[float, float]]:
        """Find intersection point of two line segments"""
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-6:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return (x, y)
    
    def correct_skew(
        self,
        image: np.ndarray,
        angle_threshold: float = 5.0,
    ) -> Tuple[np.ndarray, float]:
        """
        Correct image skew rotation.
        
        Args:
            image: Input image
            angle_threshold: Minimum angle to correct
            
        Returns:
            Tuple of (corrected image, detected angle)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        coords = np.column_stack(np.where(binary > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = 90 + angle
        
        if abs(angle) < angle_threshold:
            return image, angle
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        corrected = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return corrected, angle
    
    def normalize_size(
        self,
        image: np.ndarray,
        maintain_aspect: bool = True,
    ) -> np.ndarray:
        """
        Normalize image to standard size.
        
        Args:
            image: Input image
            maintain_aspect: Keep aspect ratio
            
        Returns:
            Normalized image
        """
        if maintain_aspect:
            h, w = image.shape[:2]
            aspect = w / h
            target_aspect = self.target_width / self.target_height
            
            if aspect > target_aspect:
                new_w = self.target_width
                new_h = int(self.target_width / aspect)
            else:
                new_h = self.target_height
                new_w = int(self.target_height * aspect)
            
            normalized = cv2.resize(image, (new_w, new_h))
            
            canvas = np.ones((self.target_height, self.target_width, 3), dtype=np.uint8) * 255
            y_offset = (self.target_height - new_h) // 2
            x_offset = (self.target_width - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = normalized
            
            return canvas
        else:
            return cv2.resize(image, (self.target_width, self.target_height))


def visualize_rectification(
    original: np.ndarray,
    rectified: np.ndarray,
    corners: np.ndarray,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Visualize rectification results.
    
    Args:
        original: Original plate crop
        rectified: Rectified plate
        corners: Detected corners
        save_path: Optional path to save visualization
        
    Returns:
        Combined visualization image
    """
    h = max(original.shape[0], rectified.shape[0])
    w = original.shape[1] + rectified.shape[1] + 10
    
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 200
    
    canvas[:original.shape[0], :original.shape[1]] = original
    
    y_offset = (h - rectified.shape[0]) // 2
    canvas[y_offset:y_offset+rectified.shape[0], original.shape[1]+10:] = rectified
    
    for i, corner in enumerate(corners):
        cv2.circle(canvas, (int(corner[0]), int(corner[1])), 5, (0, 0, 255), -1)
        cv2.putText(canvas, str(i+1), (int(corner[0])+5, int(corner[1])+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    if corners is not None and len(corners) == 4:
        pts = corners.astype(np.int32)
        for i in range(4):
            pt1 = tuple(pts[i])
            pt2 = tuple(pts[(i+1) % 4])
            cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)
    
    cv2.putText(canvas, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(canvas, "Rectified", (original.shape[1]+20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    if save_path:
        cv2.imwrite(save_path, canvas)
    
    return canvas
