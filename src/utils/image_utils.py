"""
Utility functions for Vietnamese LPR
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Args:
        image: Input image (BGR or grayscale)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Enhanced image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    
    if len(image.shape) == 3:
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced


def perspective_transform(
    image: np.ndarray,
    corners: np.ndarray,
    target_size: Tuple[int, int] = (320, 64)
) -> np.ndarray:
    """
    Apply perspective transform to straighten a plate.
    
    Args:
        image: Input image
        corners: 4 corners of the plate [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        target_size: Output size (width, height)
        
    Returns:
        Transformed plate image
    """
    if len(corners) != 4:
        raise ValueError("Exactly 4 corners required")
    
    corners = np.array(corners, dtype=np.float32)
    
    sorted_corners = corners.copy()
    sorted_corners = sorted_corners[np.argsort(sorted_corners[:, 1])]
    
    top = sorted_corners[:2]
    bottom = sorted_corners[2:]
    
    top_left = top[np.argmin(top[:, 0])]
    top_right = top[np.argmax(top[:, 0])]
    bottom_left = bottom[np.argmin(bottom[:, 0])]
    bottom_right = bottom[np.argmax(bottom[:, 0])]
    
    src_points = np.array([
        top_left, top_right, bottom_right, bottom_left
    ], dtype=np.float32)
    
    w = target_size[0]
    h = target_size[1]
    
    dst_points = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    transformed = cv2.warpPerspective(
        image, 
        matrix, 
        target_size,
        flags=cv2.INTER_CUBIC
    )
    
    return transformed


def enhance_plate_image(
    image: np.ndarray,
    denoise: bool = True,
    sharpen: bool = True,
    contrast_boost: bool = True
) -> np.ndarray:
    """
    Enhance plate image for better recognition.
    
    Args:
        image: Input plate image
        denoise: Apply denoising
        sharpen: Apply sharpening
        contrast_boost: Boost contrast
        
    Returns:
        Enhanced image
    """
    img = image.copy()
    
    if denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    if contrast_boost:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    if sharpen:
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
    
    return img


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union between two boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU score
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def non_max_suppression(
    boxes: List[List[float]],
    scores: List[float],
    iou_threshold: float = 0.45
) -> Tuple[List[int], List[float]]:
    """
    Apply Non-Maximum Suppression.
    
    Args:
        boxes: List of boxes [x1, y1, x2, y2]
        scores: Confidence scores
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Indices of kept boxes and their scores
    """
    if len(boxes) == 0:
        return [], []
    
    indices = np.argsort(scores)[::-1]
    kept_indices = []
    
    while len(indices) > 0:
        current = indices[0]
        kept_indices.append(current)
        
        if len(indices) == 1:
            break
        
        current_box = boxes[current]
        rest_boxes = [boxes[i] for i in indices[1:]]
        
        ious = [calculate_iou(current_box, box) for box in rest_boxes]
        
        indices = indices[1:][np.array(ious) < iou_threshold]
    
    return kept_indices, [scores[i] for i in kept_indices]


def visualize_comparison(
    images: List[np.ndarray],
    titles: List[str],
    figsize: Tuple[int, int] = (15, 5)
) -> np.ndarray:
    """
    Visualize multiple images side by side.
    
    Args:
        images: List of images
        titles: List of titles
        figsize: Figure size
        
    Returns:
        Combined visualization image
    """
    n = len(images)
    
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images, titles):
        if len(img.shape) == 2:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return data


def save_detection_results(
    image: np.ndarray,
    results: List[dict],
    output_path: str,
    show_confidence: bool = True
):
    """
    Save detection results to image file.
    
    Args:
        image: Input image
        results: Detection results
        output_path: Output file path
        show_confidence: Show confidence scores
    """
    img = image.copy()
    
    for result in results:
        bbox = result.get('bbox', [])
        plate = result.get('plate', '')
        confidence = result.get('confidence', 0)
        
        if len(bbox) >= 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if show_confidence:
                label = f"{plate} ({confidence:.2f})"
                cv2.putText(
                    img, label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )
    
    cv2.imwrite(output_path, img)
