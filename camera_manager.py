"""
Thread-safe singleton camera manager.

Provides a single VideoCapture instance shared across video streaming and analysis
to prevent device contention on macOS and other platforms.
"""
import cv2
import threading
import time
from typing import Optional, Tuple
import numpy as np


class CameraManager:
    """Thread-safe singleton for camera access."""
    
    _instance: Optional["CameraManager"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "CameraManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._cap_lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_time: float = 0
        self._frame_lock = threading.Lock()
        
        print("[CameraManager] Singleton initialized")
    
    def _ensure_open(self) -> bool:
        """Open camera if not already open. Must be called with _cap_lock held."""
        if self._cap is None or not self._cap.isOpened():
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
            self._cap = cv2.VideoCapture(0)
            if not self._cap.isOpened():
                print("[CameraManager] WARNING: Failed to open camera")
                return False
            print("[CameraManager] Camera opened successfully")
        return True
    
    def is_open(self) -> bool:
        """Check if camera is currently open."""
        with self._cap_lock:
            return self._cap is not None and self._cap.isOpened()
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera (thread-safe).
        
        Returns:
            (success, frame) tuple. Frame is None if read failed.
        """
        with self._cap_lock:
            if not self._ensure_open():
                return False, None
            
            ret, frame = self._cap.read()
            if ret and frame is not None:
                # Cache the frame for analysis requests
                with self._frame_lock:
                    self._last_frame = frame.copy()
                    self._last_frame_time = time.time()
            return ret, frame
    
    def get_last_frame(self, max_age: float = 0.5) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the most recently captured frame (for analysis without blocking stream).
        
        Args:
            max_age: Maximum age in seconds for the cached frame.
            
        Returns:
            (success, frame) tuple. If cached frame is too old, reads a new one.
        """
        with self._frame_lock:
            if self._last_frame is not None:
                age = time.time() - self._last_frame_time
                if age < max_age:
                    return True, self._last_frame.copy()
        
        # Cache miss or too old - read fresh frame
        return self.read_frame()
    
    def release(self):
        """Release the camera (thread-safe)."""
        with self._cap_lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                    print("[CameraManager] Camera released")
                except Exception as e:
                    print(f"[CameraManager] Error releasing camera: {e}")
                self._cap = None
        
        with self._frame_lock:
            self._last_frame = None
            self._last_frame_time = 0


# Module-level singleton instance
_camera_manager: Optional[CameraManager] = None


def get_camera_manager() -> CameraManager:
    """Get the singleton CameraManager instance."""
    global _camera_manager
    if _camera_manager is None:
        _camera_manager = CameraManager()
    return _camera_manager
