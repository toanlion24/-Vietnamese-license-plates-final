"""
Module 3: Vehicle-Plate Association
Associates detected plates with their corresponding vehicles
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from scipy.spatial.distance import cdist
import logging

from .yolo_detection import BoundingBox, DetectionResult

logger = logging.getLogger(__name__)


@dataclass
class VehiclePlatePair:
    """Association between a vehicle and its license plate"""
    vehicle: BoundingBox
    plate: BoundingBox
    vehicle_id: Optional[int] = None
    association_score: float = 0.0
    association_method: str = ""
    
    @property
    def plate_crop(self, image: np.ndarray) -> np.ndarray:
        """Crop plate region from image"""
        x1, y1, x2, y2 = map(int, self.plate.xyxy)
        return image[y1:y2, x1:x2]
    
    @property
    def vehicle_crop(self, image: np.ndarray) -> np.ndarray:
        """Crop vehicle region from image"""
        x1, y1, x2, y2 = map(int, self.vehicle.xyxy)
        return image[y1:y2, x1:x2]
    
    @property
    def vertical_distance(self) -> float:
        """Distance from plate to vehicle center (vertical)"""
        plate_center = self.plate.center
        vehicle_center = self.vehicle.center
        return abs(plate_center[1] - vehicle_center[1])
    
    @property
    def distance(self) -> float:
        """Euclidean distance between vehicle and plate centers"""
        from math import sqrt
        px, py = self.plate.center
        vx, vy = self.vehicle.center
        return sqrt((px - vx)**2 + (py - vy)**2)


@dataclass
class AssociationResult:
    """Result of vehicle-plate association"""
    pairs: List[VehiclePlatePair] = field(default_factory=list)
    unmatched_vehicles: List[BoundingBox] = field(default_factory=list)
    unmatched_plates: List[BoundingBox] = field(default_factory=list)
    frame_id: int = 0
    timestamp: float = 0.0
    
    @property
    def total_pairs(self) -> int:
        return len(self.pairs)
    
    @property
    def association_rate(self) -> float:
        """Percentage of plates successfully associated"""
        total_plates = len(self.pairs) + len(self.unmatched_plates)
        if total_plates == 0:
            return 1.0
        return len(self.pairs) / total_plates


class VehiclePlateAssociator:
    """
    Associates detected license plates with their corresponding vehicles.
    
    Methods:
    - Spatial: Based on vertical position (plate below vehicle)
    - Distance: Based on center distance
    - Size: Based on size relationship
    - Hungarian: Optimal matching using Hungarian algorithm
    """
    
    def __init__(
        self,
        method: str = "spatial",
        max_distance_ratio: float = 0.3,
        size_ratio_min: float = 0.02,
        size_ratio_max: float = 0.5,
        iou_threshold: float = 0.3,
    ):
        """
        Initialize associator.
        
        Args:
            method: Association method ('spatial', 'distance', 'hungarian', 'hybrid')
            max_distance_ratio: Max distance as ratio of image height
            size_ratio_min: Min plate/vehicle size ratio
            size_ratio_max: Max plate/vehicle size ratio
            iou_threshold: IoU threshold for matching
        """
        self.method = method
        self.max_distance_ratio = max_distance_ratio
        self.size_ratio_min = size_ratio_min
        self.size_ratio_max = size_ratio_max
        self.iou_threshold = iou_threshold
    
    def associate(
        self,
        detections: DetectionResult,
        image_shape: Optional[Tuple[int, int]] = None,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> AssociationResult:
        """
        Associate vehicles with plates.
        
        Args:
            detections: DetectionResult from YOLO detector
            image_shape: (height, width) of image
            frame_id: Current frame ID
            timestamp: Current timestamp
            
        Returns:
            AssociationResult with pairs
        """
        vehicles = detections.vehicles
        plates = detections.plates
        
        if image_shape is None:
            image_shape = detections.image_shape
        
        h, w = image_shape[:2]
        
        result = AssociationResult(
            frame_id=frame_id,
            timestamp=timestamp
        )
        
        if not vehicles:
            result.unmatched_plates = plates
            return result
        
        if not plates:
            result.unmatched_vehicles = vehicles
            return result
        
        if self.method == "spatial":
            pairs = self._spatial_association(vehicles, plates, h, w)
        elif self.method == "distance":
            pairs = self._distance_association(vehicles, plates)
        elif self.method == "hungarian":
            pairs = self._hungarian_association(vehicles, plates)
        elif self.method == "hybrid":
            pairs = self._hybrid_association(vehicles, plates, h, w)
        else:
            pairs = self._spatial_association(vehicles, plates, h, w)
        
        result.pairs = pairs
        
        paired_vehicle_ids = {id(p.vehicle) for p in pairs}
        paired_plate_ids = {id(p.plate) for p in pairs}
        
        result.unmatched_vehicles = [v for v in vehicles if id(v) not in paired_vehicle_ids]
        result.unmatched_plates = [p for p in plates if id(p) not in paired_plate_ids]
        
        return result
    
    def _spatial_association(
        self,
        vehicles: List[BoundingBox],
        plates: List[BoundingBox],
        img_height: int,
        img_width: int,
    ) -> List[VehiclePlatePair]:
        """
        Spatial association based on vertical position.
        
        For Vietnamese vehicles, license plates are typically:
        - Front plates: in front of vehicle (upper area)
        - Rear plates: behind vehicle (lower area)
        - Most common: on the vehicle body
        """
        pairs = []
        max_dist = img_height * self.max_distance_ratio
        
        for plate in plates:
            best_pair = None
            best_score = 0.0
            
            for vehicle in vehicles:
                score = self._calculate_spatial_score(plate, vehicle, img_height, max_dist)
                
                if score > best_score:
                    best_score = score
                    best_pair = VehiclePlatePair(
                        vehicle=vehicle,
                        plate=plate,
                        association_score=score,
                        association_method="spatial"
                    )
            
            if best_pair and best_score > 0:
                pairs.append(best_pair)
        
        return pairs
    
    def _calculate_spatial_score(
        self,
        plate: BoundingBox,
        vehicle: BoundingBox,
        img_height: int,
        max_dist: float,
    ) -> float:
        """Calculate spatial association score"""
        score = 0.0
        
        # Vertical distance score (plate should be near vehicle)
        v_dist = abs(plate.center[1] - vehicle.center[1])
        if v_dist < max_dist:
            score += (1 - v_dist / max_dist) * 0.4
        
        # Plate should be inside or near vehicle horizontally
        h_overlap = self._horizontal_overlap(plate, vehicle)
        score += h_overlap * 0.3
        
        # Size ratio check
        plate_area = plate.area
        vehicle_area = vehicle.area
        if vehicle_area > 0:
            size_ratio = plate_area / vehicle_area
            if self.size_ratio_min < size_ratio < self.size_ratio_max:
                ratio_score = 1 - abs(size_ratio - 0.1) / 0.1
                score += max(0, ratio_score) * 0.3
        
        return min(1.0, score)
    
    def _horizontal_overlap(self, plate: BoundingBox, vehicle: BoundingBox) -> float:
        """Calculate horizontal overlap ratio"""
        overlap = min(plate.x2, vehicle.x2) - max(plate.x1, vehicle.x1)
        if overlap <= 0:
            return 0.0
        
        plate_width = plate.x2 - plate.x1
        return overlap / plate_width if plate_width > 0 else 0.0
    
    def _distance_association(
        self,
        vehicles: List[BoundingBox],
        plates: List[BoundingBox],
    ) -> List[VehiclePlatePair]:
        """Association based on center distance"""
        pairs = []
        
        cost_matrix = np.zeros((len(plates), len(vehicles)))
        
        for i, plate in enumerate(plates):
            for j, vehicle in enumerate(vehicles):
                dist = np.sqrt(
                    (plate.center[0] - vehicle.center[0])**2 +
                    (plate.center[1] - vehicle.center[1])**2
                )
                cost_matrix[i, j] = dist
        
        # Simple greedy matching
        for i, plate in enumerate(plates):
            j = np.argmin(cost_matrix[i])
            min_dist = cost_matrix[i, j]
            
            if min_dist < 200:  # Distance threshold
                pairs.append(VehiclePlatePair(
                    vehicle=vehicles[j],
                    plate=plate,
                    association_score=1 - min_dist / 200,
                    association_method="distance"
                ))
        
        return pairs
    
    def _hungarian_association(
        self,
        vehicles: List[BoundingBox],
        plates: List[BoundingBox],
    ) -> List[VehiclePlatePair]:
        """Optimal matching using Hungarian algorithm"""
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            logger.warning("scipy not available, using greedy matching")
            return self._distance_association(vehicles, plates)
        
        pairs = []
        
        if len(plates) == 0 or len(vehicles) == 0:
            return pairs
        
        # Build cost matrix
        cost_matrix = np.zeros((len(plates), len(vehicles)))
        for i, plate in enumerate(plates):
            for j, vehicle in enumerate(vehicles):
                dist = np.sqrt(
                    (plate.center[0] - vehicle.center[0])**2 +
                    (plate.center[1] - vehicle.center[1])**2
                )
                # Add penalty for invalid size ratios
                if vehicle.area > 0:
                    size_ratio = plate.area / vehicle.area
                    if not (self.size_ratio_min < size_ratio < self.size_ratio_max):
                        dist *= 2
                cost_matrix[i, j] = dist
        
        # Hungarian algorithm
        plate_indices, vehicle_indices = linear_sum_assignment(cost_matrix)
        
        for pi, vi in zip(plate_indices, vehicle_indices):
            if cost_matrix[pi, vi] < 200:
                pairs.append(VehiclePlatePair(
                    vehicle=vehicles[vi],
                    plate=plates[pi],
                    association_score=1 - cost_matrix[pi, vi] / 200,
                    association_method="hungarian"
                ))
        
        return pairs
    
    def _hybrid_association(
        self,
        vehicles: List[BoundingBox],
        plates: List[BoundingBox],
        img_height: int,
        img_width: int,
    ) -> List[VehiclePlatePair]:
        """Hybrid association combining multiple methods"""
        pairs = []
        
        # First, try spatial association
        spatial_pairs = self._spatial_association(vehicles, plates, img_height, img_width)
        pairs.extend(spatial_pairs)
        
        # For unmatched plates, try Hungarian
        paired_plates = {id(p.plate) for p in spatial_pairs}
        remaining_plates = [p for p in plates if id(p) not in paired_plates]
        
        if remaining_plates:
            remaining_vehicles = vehicles  # Still use all vehicles
            hungarian_pairs = self._hungarian_association(
                remaining_vehicles, remaining_plates
            )
            
            for pair in hungarian_pairs:
                if id(pair.plate) not in paired_plates:
                    pair.association_method = "hybrid"
                    pairs.append(pair)
                    paired_plates.add(id(pair.plate))
        
        return pairs


def visualize_associations(
    image: np.ndarray,
    associations: AssociationResult,
    show_scores: bool = True,
    show_ids: bool = True,
    vehicle_color: Tuple[int, int, int] = (0, 255, 0),
    plate_color: Tuple[int, int, int] = (255, 0, 0),
    line_color: Tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """
    Visualize vehicle-plate associations.
    
    Args:
        image: Input image
        associations: AssociationResult
        show_scores: Show association scores
        show_ids: Show IDs and labels
        vehicle_color: Vehicle box color
        plate_color: Plate box color
        line_color: Connection line color
        
    Returns:
        Image with visualizations
    """
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Draw connections
    for pair in associations.pairs:
        v_cx, v_cy = map(int, pair.vehicle.center)
        p_cx, p_cy = map(int, pair.plate.center)
        cv2.line(img, (v_cx, v_cy), (p_cx, p_cy), line_color, 2)
    
    # Draw vehicles
    for box in associations.unmatched_vehicles:
        x1, y1, x2, y2 = map(int, box.xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), vehicle_color, 2)
        if show_ids:
            label = f"{box.class_name}: {box.confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), font, 0.5, vehicle_color, 1)
    
    # Draw plates
    for box in associations.unmatched_plates:
        x1, y1, x2, y2 = map(int, box.xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), plate_color, 2)
        if show_ids:
            label = f"PLATE: {box.confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), font, 0.5, plate_color, 1)
    
    # Draw paired vehicles and plates with scores
    for pair in associations.pairs:
        # Vehicle
        vx1, vy1, vx2, vy2 = map(int, pair.vehicle.xyxy)
        cv2.rectangle(img, (vx1, vy1), (vx2, vy2), vehicle_color, 2)
        
        # Plate
        px1, py1, px2, py2 = map(int, pair.plate.xyxy)
        cv2.rectangle(img, (px1, py1), (px2, py2), plate_color, 2)
        
        if show_scores:
            score_text = f"ID:{pair.vehicle_id} Score:{pair.association_score:.2f}"
            mid_x = (px1 + px2) // 2
            cv2.putText(img, score_text, (mid_x, py2 + 20), font, 0.5, (0, 255, 255), 1)
    
    # Summary
    summary = f"Pairs: {associations.total_pairs} | Unmatched V: {len(associations.unmatched_vehicles)} | Unmatched P: {len(associations.unmatched_plates)}"
    cv2.putText(img, summary, (10, 25), font, 0.6, (255, 255, 255), 2)
    
    return img
