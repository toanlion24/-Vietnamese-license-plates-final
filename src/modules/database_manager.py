"""
Module 9: Database & Streamlit UI
Stores recognition results and provides web interface
"""

import sqlite3
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import base64
import cv2
import logging

logger = logging.getLogger(__name__)


@dataclass
class RecognitionRecord:
    """Record of a single recognition event"""
    id: Optional[int] = None
    timestamp: str = ""
    plate_text: str = ""
    confidence: float = 0.0
    plate_type: str = ""
    province: str = ""
    image_path: str = ""
    thumbnail_data: str = ""  # Base64 encoded
    vehicle_type: str = ""
    camera_id: str = ""
    source: str = ""  # 'image', 'video', 'webcam'
    processing_time_ms: float = 0.0
    bbox_x1: float = 0.0
    bbox_y1: float = 0.0
    bbox_x2: float = 0.0
    bbox_y2: float = 0.0


class DatabaseManager:
    """
    SQLite database manager for storing recognition results.
    """
    
    def __init__(self, db_path: str = "outputs/lpr_database.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_directory()
        self._init_database()
    
    def _ensure_directory(self):
        """Ensure database directory exists"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recognitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                plate_text TEXT NOT NULL,
                confidence REAL,
                plate_type TEXT,
                province TEXT,
                image_path TEXT,
                thumbnail_data TEXT,
                vehicle_type TEXT,
                camera_id TEXT,
                source TEXT,
                processing_time_ms REAL,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_plate_text 
            ON recognitions(plate_text)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON recognitions(timestamp)
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT UNIQUE,
                name TEXT,
                location TEXT,
                status TEXT DEFAULT 'active'
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized: {self.db_path}")
    
    def insert_recognition(self, record: RecognitionRecord) -> int:
        """
        Insert a recognition record.
        
        Args:
            record: RecognitionRecord to insert
            
        Returns:
            Inserted record ID
        """
        if record.timestamp == "":
            record.timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO recognitions (
                timestamp, plate_text, confidence, plate_type, province,
                image_path, thumbnail_data, vehicle_type, camera_id,
                source, processing_time_ms, bbox_x1, bbox_y1, bbox_x2, bbox_y2
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.timestamp,
            record.plate_text,
            record.confidence,
            record.plate_type,
            record.province,
            record.image_path,
            record.thumbnail_data,
            record.vehicle_type,
            record.camera_id,
            record.source,
            record.processing_time_ms,
            record.bbox_x1,
            record.bbox_y1,
            record.bbox_x2,
            record.bbox_y2,
        ))
        
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return record_id
    
    def get_recent_recognitions(
        self,
        limit: int = 100,
        camera_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get recent recognitions.
        
        Args:
            limit: Maximum number of records
            camera_id: Filter by camera ID
            
        Returns:
            List of recognition records
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if camera_id:
            cursor.execute("""
                SELECT * FROM recognitions 
                WHERE camera_id = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (camera_id, limit))
        else:
            cursor.execute("""
                SELECT * FROM recognitions 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def search_by_plate(
        self,
        plate_text: str,
        limit: int = 50,
    ) -> List[Dict]:
        """
        Search recognitions by plate text.
        
        Args:
            plate_text: Plate text to search
            limit: Maximum results
            
        Returns:
            List of matching records
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM recognitions 
            WHERE plate_text LIKE ?
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (f"%{plate_text}%", limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_statistics(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary of statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total recognitions
        cursor.execute("SELECT COUNT(*) FROM recognitions")
        stats['total_recognitions'] = cursor.fetchone()[0]
        
        # Unique plates
        cursor.execute("SELECT COUNT(DISTINCT plate_text) FROM recognitions")
        stats['unique_plates'] = cursor.fetchone()[0]
        
        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM recognitions")
        stats['avg_confidence'] = cursor.fetchone()[0] or 0
        
        # By plate type
        cursor.execute("""
            SELECT plate_type, COUNT(*) as count 
            FROM recognitions 
            GROUP BY plate_type
        """)
        stats['by_plate_type'] = dict(cursor.fetchall())
        
        # By camera
        cursor.execute("""
            SELECT camera_id, COUNT(*) as count 
            FROM recognitions 
            GROUP BY camera_id
        """)
        stats['by_camera'] = dict(cursor.fetchall())
        
        # Recent 24h
        cursor.execute("""
            SELECT COUNT(*) FROM recognitions 
            WHERE timestamp > datetime('now', '-1 day')
        """)
        stats['last_24h'] = cursor.fetchone()[0]
        
        conn.close()
        
        return stats
    
    def delete_old_records(self, days: int = 30) -> int:
        """
        Delete records older than specified days.
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of deleted records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM recognitions 
            WHERE timestamp < datetime('now', ?)
        """, (f"-{days} days",))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        logger.info(f"Deleted {deleted} old records")
        return deleted


def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def decode_base64_to_image(data: str) -> Optional[np.ndarray]:
    """Decode base64 string to image"""
    try:
        img_bytes = base64.b64decode(data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def create_record_from_result(
    plate_text: str,
    confidence: float,
    plate_type: str,
    bbox: List[float],
    vehicle_type: str = "",
    camera_id: str = "",
    source: str = "",
    image: Optional[np.ndarray] = None,
    province: str = "",
    processing_time_ms: float = 0.0,
) -> RecognitionRecord:
    """
    Create a RecognitionRecord from processing result.
    
    Args:
        plate_text: Recognized plate text
        confidence: Recognition confidence
        plate_type: Type of plate
        bbox: Bounding box coordinates
        vehicle_type: Type of vehicle
        camera_id: Camera identifier
        source: Source type
        image: Optional image for thumbnail
        province: Province name
        processing_time_ms: Processing time
        
    Returns:
        RecognitionRecord
    """
    thumbnail_data = ""
    if image is not None:
        # Create thumbnail
        thumb = cv2.resize(image, (200, 100))
        thumbnail_data = encode_image_to_base64(thumb)
    
    return RecognitionRecord(
        timestamp=datetime.now().isoformat(),
        plate_text=plate_text,
        confidence=confidence,
        plate_type=plate_type,
        province=province,
        thumbnail_data=thumbnail_data,
        vehicle_type=vehicle_type,
        camera_id=camera_id,
        source=source,
        processing_time_ms=processing_time_ms,
        bbox_x1=bbox[0] if len(bbox) > 0 else 0,
        bbox_y1=bbox[1] if len(bbox) > 1 else 0,
        bbox_x2=bbox[2] if len(bbox) > 2 else 0,
        bbox_y2=bbox[3] if len(bbox) > 3 else 0,
    )
