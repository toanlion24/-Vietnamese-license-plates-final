"""
Module 8: ByteTrack & Voting
Multi-frame tracking and consensus voting for optimal results
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Tracklet:
    """Object tracklet for tracking across frames"""
    track_id: int
    plate_text: str
    confidence: float
    bbox: List[float]
    vehicle_id: Optional[int] = None
    first_seen: int = 0
    last_seen: int = 0
    frame_count: int = 1
    appearances: List[Dict] = field(default_factory=list)
    
    def add_observation(self, plate: str, conf: float, bbox: List[float], frame_id: int):
        """Add new observation to tracklet"""
        self.last_seen = frame_id
        self.frame_count += 1
        self.appearances.append({
            'plate': plate,
            'confidence': conf,
            'bbox': bbox,
            'frame_id': frame_id,
        })
        
        # Update with higher confidence observation
        if conf > self.confidence:
            self.plate_text = plate
            self.confidence = conf
            self.bbox = bbox
    
    @property
    def avg_confidence(self) -> float:
        if not self.appearances:
            return self.confidence
        return np.mean([a['confidence'] for a in self.appearances])
    
    @property
    def duration(self) -> int:
        return self.last_seen - self.first_seen + 1
    
    @property
    def observations(self) -> int:
        return len(self.appearances)


@dataclass
class VotingResult:
    """Result of voting across tracklet observations"""
    final_plate: str
    confidence: float
    vote_counts: Dict[str, int]
    total_observations: int
    consensus_ratio: float
    
    @property
    def has_consensus(self) -> bool:
        return self.consensus_ratio >= 0.5


class TrackletManager:
    """
    Manages tracklets for multi-frame tracking.
    
    Features:
    - Track creation and deletion
    - Tracklet merging
    - Age-based cleanup
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 2,
        iou_threshold: float = 0.3,
    ):
        """
        Initialize tracklet manager.
        
        Args:
            max_age: Maximum frames without update before track removal
            min_hits: Minimum observations to confirm track
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self._tracklets: Dict[int, Tracklet] = {}
        self._next_id = 1
        self._current_frame = 0
    
    def update(
        self,
        detections: List[Dict],
        frame_id: int,
    ) -> List[Tracklet]:
        """
        Update tracklets with new detections.
        
        Args:
            detections: List of detection dicts with 'plate', 'confidence', 'bbox'
            frame_id: Current frame ID
            
        Returns:
            List of confirmed tracklets
        """
        self._current_frame = frame_id
        
        # Match detections to existing tracklets
        matched, unmatched = self._match_detections(detections)
        
        # Update matched tracklets
        for tracklet, detection in matched:
            tracklet.add_observation(
                plate=detection['plate'],
                conf=detection['confidence'],
                bbox=detection['bbox'],
                frame_id=frame_id,
            )
        
        # Create new tracklets for unmatched detections
        for detection in unmatched:
            tracklet = Tracklet(
                track_id=self._next_id,
                plate_text=detection['plate'],
                confidence=detection['confidence'],
                bbox=detection['bbox'],
                first_seen=frame_id,
                last_seen=frame_id,
            )
            self._tracklets[self._next_id] = tracklet
            self._next_id += 1
        
        # Age out old tracklets
        self._age_tracklets()
        
        # Return confirmed tracklets
        return [t for t in self._tracklets.values() if t.frame_count >= self.min_hits]
    
    def _match_detections(
        self,
        detections: List[Dict],
    ) -> Tuple[List[Tuple[Tracklet, Dict]], List[Dict]]:
        """Match detections to tracklets using IoU"""
        matched = []
        unmatched = []
        
        active_tracks = [t for t in self._tracklets.values()
                        if self._current_frame - t.last_seen <= self.max_age]
        
        if not active_tracks:
            return [], detections
        
        # Build cost matrix
        cost_matrix = np.full((len(detections), len(active_tracks)), 1e6)
        
        for i, det in enumerate(detections):
            for j, track in enumerate(active_tracks):
                iou = self._calculate_iou(det['bbox'], track.bbox)
                if iou > 0:
                    cost_matrix[i, j] = 1 - iou  # Convert to cost
        
        # Greedy matching
        used_det = set()
        used_track = set()
        
        while True:
            min_cost = 1e6
            min_idx = None
            
            for i in range(len(detections)):
                if i in used_det:
                    continue
                for j in range(len(active_tracks)):
                    if j in used_track:
                        continue
                    if cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        min_idx = (i, j)
            
            if min_idx is None or min_cost > (1 - self.iou_threshold):
                break
            
            i, j = min_idx
            matched.append((active_tracks[j], detections[i]))
            used_det.add(i)
            used_track.add(j)
        
        unmatched = [det for i, det in enumerate(detections) if i not in used_det]
        
        return matched, unmatched
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def _age_tracklets(self):
        """Remove old tracklets"""
        to_remove = []
        
        for track_id, track in self._tracklets.items():
            if self._current_frame - track.last_seen > self.max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self._tracklets[track_id]
    
    def get_tracklet(self, track_id: int) -> Optional[Tracklet]:
        """Get tracklet by ID"""
        return self._tracklets.get(track_id)
    
    def get_all_tracklets(self) -> List[Tracklet]:
        """Get all active tracklets"""
        return list(self._tracklets.values())
    
    def get_confirmed_tracklets(self) -> List[Tracklet]:
        """Get confirmed tracklets"""
        return [t for t in self._tracklets.values() if t.frame_count >= self.min_hits]
    
    def reset(self):
        """Reset all tracklets"""
        self._tracklets.clear()
        self._next_id = 1
        self._current_frame = 0


class VotingSystem:
    """
    Voting system for consensus-based plate recognition.
    
    Methods:
    - Majority voting
    - Confidence-weighted voting
    - Time-decay voting
    - Position-aware voting
    """
    
    def __init__(
        self,
        method: str = "majority",
        min_votes: int = 2,
        decay_factor: float = 0.95,
    ):
        """
        Initialize voting system.
        
        Args:
            method: Voting method ('majority', 'confidence', 'decay')
            min_votes: Minimum votes required
            decay_factor: Time decay factor for older observations
        """
        self.method = method
        self.min_votes = min_votes
        self.decay_factor = decay_factor
    
    def vote(self, tracklet: Tracklet) -> VotingResult:
        """
        Perform voting on tracklet observations.
        
        Args:
            tracklet: Tracklet with observations
            
        Returns:
            VotingResult with consensus
        """
        if not tracklet.appearances:
            return VotingResult(
                final_plate=tracklet.plate_text,
                confidence=tracklet.confidence,
                vote_counts={tracklet.plate_text: 1},
                total_observations=1,
                consensus_ratio=1.0,
            )
        
        votes = self._count_votes(tracklet)
        
        if self.method == "majority":
            return self._majority_vote(votes, tracklet)
        elif self.method == "confidence":
            return self._confidence_vote(votes, tracklet)
        elif self.method == "decay":
            return self._decay_vote(tracklet)
        else:
            return self._majority_vote(votes, tracklet)
    
    def _count_votes(self, tracklet: Tracklet) -> Dict[str, List[Tuple[float, int]]]:
        """Count votes for each plate text"""
        votes = defaultdict(list)
        
        for obs in tracklet.appearances:
            plate = obs['plate']
            conf = obs['confidence']
            frame_id = obs['frame_id']
            votes[plate].append((conf, frame_id))
        
        return dict(votes)
    
    def _majority_vote(
        self,
        votes: Dict[str, List[Tuple[float, int]]],
        tracklet: Tracklet,
    ) -> VotingResult:
        """Majority voting"""
        # Count occurrences
        counts = {plate: len(votes_list) for plate, votes_list in votes.items()}
        total = sum(counts.values())
        
        # Find majority
        best_plate = max(counts, key=counts.get)
        best_count = counts[best_plate]
        
        consensus_ratio = best_count / total if total > 0 else 0
        
        # Average confidence for winning plate
        confidences = [v[0] for v in votes[best_plate]]
        avg_conf = np.mean(confidences)
        
        return VotingResult(
            final_plate=best_plate,
            confidence=avg_conf,
            vote_counts=counts,
            total_observations=total,
            consensus_ratio=consensus_ratio,
        )
    
    def _confidence_vote(
        self,
        votes: Dict[str, List[Tuple[float, int]]],
        tracklet: Tracklet,
    ) -> VotingResult:
        """Confidence-weighted voting"""
        # Sum confidences
        weighted_counts = {}
        
        for plate, vote_list in votes.items():
            total_conf = sum(conf for conf, _ in vote_list)
            weighted_counts[plate] = total_conf
        
        total = sum(weighted_counts.values())
        
        best_plate = max(weighted_counts, key=weighted_counts.get)
        best_weight = weighted_counts[best_plate]
        
        consensus_ratio = best_weight / total if total > 0 else 0
        
        # Get confidence for winning plate
        avg_conf = np.mean([v[0] for v in votes[best_plate]])
        
        return VotingResult(
            final_plate=best_plate,
            confidence=avg_conf,
            vote_counts={p: len(v) for p, v in votes.items()},
            total_observations=sum(len(v) for v in votes.values()),
            consensus_ratio=consensus_ratio,
        )
    
    def _decay_vote(self, tracklet: Tracklet) -> VotingResult:
        """Time-decay weighted voting"""
        latest_frame = tracklet.last_seen
        votes = {}
        
        for obs in tracklet.appearances:
            plate = obs['plate']
            conf = obs['confidence']
            frame_id = obs['frame_id']
            
            # Calculate decay
            age = latest_frame - frame_id
            decay = self.decay_factor ** age
            
            weighted_conf = conf * decay
            
            if plate not in votes:
                votes[plate] = []
            votes[plate].append(weighted_conf)
        
        # Sum weighted confidences
        weighted_counts = {plate: sum(v) for plate, v in votes.items()}
        total = sum(weighted_counts.values())
        
        best_plate = max(weighted_counts, key=weighted_counts.get)
        best_weight = weighted_counts[best_plate]
        
        consensus_ratio = best_weight / total if total > 0 else 0
        
        # Original confidence for winning plate
        avg_conf = np.mean([obs['confidence'] for obs in tracklet.appearances
                          if obs['plate'] == best_plate])
        
        return VotingResult(
            final_plate=best_plate,
            confidence=avg_conf,
            vote_counts={p: len(v) for p, v in votes.items()},
            total_observations=len(tracklet.appearances),
            consensus_ratio=consensus_ratio,
        )


class ByteTracker:
    """
    ByteTrack-style tracker for vehicle tracking.
    
    Simple implementation of BYTE algorithm:
    1. High confidence detections -> Strong tracklets
    2. Low confidence detections -> Weak tracklets
    3. Match strong first, then weak
    4. Recover unmatched weak tracks
    """
    
    def __init__(
        self,
        track_thresh: float = 0.5,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        max_time_lost: int = 30,
    ):
        """
        Initialize ByteTracker.
        
        Args:
            track_thresh: Tracking confidence threshold
            high_thresh: High confidence threshold
            low_thresh: Low confidence threshold
            match_thresh: Matching threshold
            max_time_lost: Max frames without detection
        """
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        
        self._tracks = []
        self._track_id_count = 0
        self._frame_id = 0
    
    def update(
        self,
        detections: List[Dict],
        frame_id: Optional[int] = None,
    ) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections with 'bbox', 'confidence', 'plate'
            frame_id: Current frame ID
            
        Returns:
            List of tracked objects
        """
        if frame_id is not None:
            self._frame_id = frame_id
        
        if self._frame_id == 0:
            self._init_tracks(detections)
            self._frame_id += 1
            return self._get_active_tracks()
        
        # Separate high and low confidence detections
        high_dets = [d for d in detections if d['confidence'] >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d['confidence'] < self.high_thresh]
        
        # Match high confidence detections to tracks
        self._match_tracks(high_dets)
        
        # Match low confidence detections
        unmatched = self._get_unmatched_tracks()
        self._match_tracks(low_dets, tracks=unmatched)
        
        # Update unmatched tracks (mark as lost)
        matched_dets = set()
        for track in self._tracks:
            if track.get('matched', False):
                track['matched'] = False
                track['frame_count'] += 1
            else:
                track['time_since_update'] += 1
        
        # Remove lost tracks
        self._tracks = [t for t in self._tracks if t['time_since_update'] < self.max_time_lost]
        
        # Activate new tracks
        for det in high_dets:
            if not det.get('_matched', False):
                self._activate_track(det)
        
        self._frame_id += 1
        
        return self._get_active_tracks()
    
    def _init_tracks(self, detections: List[Dict]):
        """Initialize tracks with first frame detections"""
        for det in detections:
            if det['confidence'] >= self.track_thresh:
                self._activate_track(det)
    
    def _activate_track(self, detection: Dict):
        """Activate a new track"""
        track = {
            'track_id': self._track_id_count,
            'plate': detection.get('plate', ''),
            'confidence': detection['confidence'],
            'bbox': detection['bbox'],
            'frame_count': 1,
            'time_since_update': 0,
            'matched': True,
        }
        self._tracks.append(track)
        self._track_id_count += 1
    
    def _match_tracks(self, detections: List[Dict], tracks: List[Dict] = None):
        """Match detections to tracks"""
        if tracks is None:
            tracks = self._tracks
        
        # Simple IoU matching
        for det in detections:
            best_match = None
            best_iou = 0.3
            
            for track in tracks:
                if track.get('matched', False):
                    continue
                
                iou = self._calculate_iou(det['bbox'], track['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = track
            
            if best_match is not None:
                best_match['matched'] = True
                best_match['bbox'] = det['bbox']
                best_match['confidence'] = det['confidence']
                best_match['plate'] = det.get('plate', best_match['plate'])
                best_match['time_since_update'] = 0
                det['_matched'] = True
    
    def _get_unmatched_tracks(self) -> List[Dict]:
        """Get tracks that were not matched"""
        return [t for t in self._tracks if not t.get('matched', False)]
    
    def _get_active_tracks(self) -> List[Dict]:
        """Get active tracks"""
        return [t for t in self._tracks if t['time_since_update'] <= 2]
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def reset(self):
        """Reset tracker"""
        self._tracks.clear()
        self._track_id_count = 0
        self._frame_id = 0


def consensus_voting(
    observations: List[Dict],
    method: str = "majority",
    min_confidence: float = 0.5,
) -> VotingResult:
    """
    Perform consensus voting on observations.
    
    Args:
        observations: List of observations with 'plate', 'confidence'
        method: Voting method
        min_confidence: Minimum confidence threshold
        
    Returns:
        VotingResult with consensus
    """
    # Filter by confidence
    filtered = [obs for obs in observations if obs['confidence'] >= min_confidence]
    
    if not filtered:
        filtered = observations
    
    # Count votes
    votes = Counter(obs['plate'] for obs in filtered)
    total = len(filtered)
    
    best_plate, best_count = votes.most_common(1)[0]
    consensus_ratio = best_count / total if total > 0 else 0
    
    # Calculate confidence
    confidences = [obs['confidence'] for obs in filtered if obs['plate'] == best_plate]
    avg_conf = np.mean(confidences) if confidences else 0.0
    
    return VotingResult(
        final_plate=best_plate,
        confidence=avg_conf,
        vote_counts=dict(votes),
        total_observations=total,
        consensus_ratio=consensus_ratio,
    )
