"""RTSP network camera video source"""

import cv2
import time
from typing import Optional, Tuple
import numpy as np
from .video_source import VideoSource


class RTSPSource(VideoSource):
    """RTSP network camera video source"""
    
    def __init__(self, rtsp_url: str, target_resolution: Optional[Tuple[int, int]] = None,
                 reconnect_timeout: int = 5, buffer_size: int = 1, name: str = "RTSP_Camera"):
        """
        Initialize RTSP camera source
        
        Args:
            rtsp_url: RTSP stream URL (e.g., rtsp://192.168.1.100:8554/stream)
            target_resolution: Target resolution as (width, height), None for native
            reconnect_timeout: Seconds to wait before reconnecting on failure
            buffer_size: Frame buffer size (1 = latest frame only)
            name: Name for this source
        """
        super().__init__(name)
        self.rtsp_url = rtsp_url
        self.target_resolution = target_resolution
        self.reconnect_timeout = reconnect_timeout
        self.buffer_size = buffer_size
        self.cap = None
        self._fps = 25.0  # Default FPS for IP cameras
        self._resolution = (640, 480)
        self._last_reconnect_attempt = 0
        
    def open(self) -> bool:
        """Open the RTSP stream"""
        try:
            # Use FFMPEG backend for better RTSP support
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            if not self.is_opened():
                print(f"Failed to open RTSP stream: {self.rtsp_url}")
                return False
            
            # Set buffer size to reduce latency (get latest frame)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            
            # Get stream properties
            self._fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self._fps == 0 or self._fps > 60:  # Sanity check
                self._fps = 25.0  # Default for IP cameras
            
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._resolution = (width, height) if width > 0 and height > 0 else (640, 480)
            
            print(f"RTSP Stream opened: {self.name}, Resolution: {self._resolution}, FPS: {self._fps}")
            self.reset_stats()
            
            return True
            
        except Exception as e:
            print(f"Error opening RTSP stream: {e}")
            return False
    
    def _try_reconnect(self) -> bool:
        """Attempt to reconnect to the RTSP stream"""
        current_time = time.time()
        
        if current_time - self._last_reconnect_attempt < self.reconnect_timeout:
            return False
        
        self._last_reconnect_attempt = current_time
        print(f"Attempting to reconnect to RTSP stream: {self.rtsp_url}")
        
        self.release()
        time.sleep(1)  # Brief pause before reconnecting
        
        return self.open()
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Get the next frame from the RTSP stream"""
        if not self.is_opened():
            # Try to reconnect
            if not self._try_reconnect():
                return False, None, time.time()
        
        timestamp = time.time()
        ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
            self._last_frame_time = timestamp
            
            # Resize if target resolution specified
            if self.target_resolution and frame.shape[1::-1] != self.target_resolution:
                frame = cv2.resize(frame, self.target_resolution)
            
            return True, frame, timestamp
        else:
            self.dropped_frames += 1
            # Try to reconnect on read failure
            self._try_reconnect()
            return False, None, timestamp
    
    def release(self):
        """Release the RTSP stream"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print(f"RTSP Stream released: {self.name}")
    
    def is_opened(self) -> bool:
        """Check if the RTSP stream is opened"""
        return self.cap is not None and self.cap.isOpened()
    
    def get_fps(self) -> float:
        """Get the FPS of the RTSP stream"""
        return self._fps
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get the resolution of the RTSP stream"""
        return self._resolution
    
    def __del__(self):
        """Destructor to ensure stream is released"""
        self.release()

