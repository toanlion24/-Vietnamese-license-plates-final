"""
Module 1: Input Stream
Handles Image, Video, Webcam, and RTSP stream input
"""

import cv2
import time
import logging
from typing import Optional, Iterator, Union, Tuple, List
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """Container for video/image frame with metadata"""
    image: np.ndarray
    frame_id: int = 0
    timestamp: float = 0.0
    source: str = "unknown"
    width: int = 0
    height: int = 0
    
    def __post_init__(self):
        if len(self.image.shape) >= 2:
            self.height, self.width = self.image.shape[:2]
    
    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)
    
    @property
    def aspect_ratio(self) -> float:
        if self.height == 0:
            return 0
        return self.width / self.height
    
    def copy(self) -> 'Frame':
        """Create a deep copy of this frame"""
        return Frame(
            image=self.image.copy(),
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            source=self.source,
            width=self.width,
            height=self.height
        )


class InputStream:
    """
    Unified input stream for images, videos, and streams.
    
    Usage:
        # From image
        stream = InputStream("test.jpg")
        frame = stream.read()
        
        # From video
        stream = InputStream("video.mp4")
        for frame in stream:
            process(frame)
        
        # From webcam
        stream = InputStream(0)
        frame = stream.read()
    """
    
    def __init__(
        self,
        source: Union[str, int, Path],
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        fps: Optional[float] = None,
        resize: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize input stream.
        
        Args:
            source: File path (str/Path), webcam index (int), or RTSP URL (str)
            start_frame: Start from this frame (video only)
            end_frame: Stop at this frame (video only)
            fps: Target FPS for video playback (None = original)
            resize: Resize frames to (width, height)
        """
        self.source = str(source)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.target_fps = fps
        self.resize = resize
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._total_frames = 0
        self._fps = 0.0
        self._started = False
        
        self._open()
    
    def _open(self):
        """Open the input source"""
        if self._cap is not None:
            self._cap.release()
        
        self._cap = cv2.VideoCapture(self.source)
        
        if not self._cap.isOpened():
            raise ValueError(f"Cannot open source: {self.source}")
        
        # Get video properties
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        
        # Seek to start frame
        if self.start_frame > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self._frame_count = self.start_frame
        
        self._started = True
    
    @property
    def fps(self) -> float:
        """Get video FPS"""
        return self._fps
    
    @property
    def frame_count(self) -> int:
        """Get total frame count (video only)"""
        return self._total_frames
    
    @property
    def duration(self) -> float:
        """Get video duration in seconds"""
        if self._fps == 0:
            return 0
        return self._total_frames / self._fps
    
    @property
    def is_opened(self) -> bool:
        """Check if stream is open"""
        return self._cap is not None and self._cap.isOpened()
    
    @property
    def current_frame(self) -> int:
        """Get current frame number"""
        return self._frame_count
    
    def read(self) -> Optional[Frame]:
        """
        Read next frame from stream.
        
        Returns:
            Frame object or None if end of stream
        """
        if not self.is_opened:
            return None
        
        # Check end frame
        if self.end_frame is not None and self._frame_count >= self.end_frame:
            return None
        
        ret, image = self._cap.read()
        
        if not ret:
            return None
        
        # Apply resize if specified
        if self.resize:
            image = cv2.resize(image, self.resize)
        
        # Control playback speed
        if self.target_fps and self._fps > 0:
            delay = int(1000 / self.target_fps)
            time.sleep(delay / 1000)
        
        frame = Frame(
            image=image,
            frame_id=self._frame_count,
            timestamp=self._frame_count / self._fps if self._fps > 0 else 0,
            source=self.source
        )
        
        self._frame_count += 1
        return frame
    
    def __iter__(self) -> Iterator[Frame]:
        """Iterate over frames"""
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
    
    def release(self):
        """Release resources"""
        if self._cap:
            self._cap.release()
            self._cap = None
            self._started = False
    
    def seek(self, frame_number: int):
        """Seek to specific frame"""
        if self._cap:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self._frame_count = frame_number
    
    def reset(self):
        """Reset to beginning"""
        if self._cap:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self._frame_count = self.start_frame


class WebcamStream:
    """
    Dedicated webcam stream handler with additional features.
    
    Usage:
        with WebcamStream(camera_id=0, width=1280, height=720) as cam:
            frame = cam.read()
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        buffer_size: int = 1,
    ):
        """
        Initialize webcam stream.
        
        Args:
            camera_id: Camera index (0 = default webcam)
            width: Frame width
            height: Frame height
            fps: Target FPS
            buffer_size: Frame buffer size (1 = no buffering)
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = fps
        self.buffer_size = buffer_size
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._last_frame_time = 0.0
        self._started = False
    
    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
    
    def start(self):
        """Start the webcam stream"""
        if self._started:
            return
        
        self._cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
        
        # Set properties
        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.target_fps:
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        
        # Get actual properties
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.target_fps = self._cap.get(cv2.CAP_PROP_FPS)
        
        self._started = True
        self._frame_count = 0
        self._last_frame_time = time.time()
        
        logger.info(f"Webcam started: {self.width}x{self.height} @ {self.target_fps}fps")
    
    def read(self) -> Optional[Frame]:
        """Read next frame from webcam"""
        if not self._started:
            self.start()
        
        ret, image = self._cap.read()
        
        if not ret:
            return None
        
        current_time = time.time()
        timestamp = current_time - self._last_frame_time
        self._last_frame_time = current_time
        
        frame = Frame(
            image=image,
            frame_id=self._frame_count,
            timestamp=timestamp,
            source=f"webcam_{self.camera_id}"
        )
        
        self._frame_count += 1
        return frame
    
    def __iter__(self) -> Iterator[Frame]:
        """Iterate over webcam frames"""
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
    
    def release(self):
        """Release webcam"""
        if self._cap:
            self._cap.release()
            self._cap = None
            self._started = False
            logger.info(f"Webcam {self.camera_id} released")


def load_image(path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load single image from file.
    
    Args:
        path: Image file path
        
    Returns:
        Image as numpy array (BGR) or None if failed
    """
    path = str(path)
    
    if not Path(path).exists():
        logger.error(f"Image not found: {path}")
        return None
    
    image = cv2.imread(path)
    
    if image is None:
        logger.error(f"Cannot read image: {path}")
        return None
    
    return image


def load_video_info(path: Union[str, Path]) -> dict:
    """
    Get video file information.
    
    Args:
        path: Video file path
        
    Returns:
        Dictionary with video properties
    """
    path = str(path)
    cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        return {}
    
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    
    cap.release()
    return info


def list_cameras(max_cameras: int = 5) -> List[int]:
    """
    List available camera indices.
    
    Args:
        max_cameras: Maximum cameras to check
        
    Returns:
        List of available camera indices
    """
    available = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap is not None and cap.isOpened():
            available.append(i)
            cap.release()
    
    return available
