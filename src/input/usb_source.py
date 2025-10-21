"""USB camera video source"""

import cv2
import time
from typing import Optional, Tuple
import numpy as np
from .video_source import VideoSource


class USBSource(VideoSource):
    """USB camera video source using OpenCV"""
    
    def __init__(self, device_id: int = 0, target_resolution: Optional[Tuple[int, int]] = None, 
                 name: str = "USB_Camera"):
        """
        Initialize USB camera source
        
        Args:
            device_id: Camera device ID (default 0)
            target_resolution: Target resolution as (width, height), None for native
            name: Name for this source
        """
        super().__init__(name)
        self.device_id = device_id
        self.target_resolution = target_resolution
        self.cap = None
        self._fps = 30.0  # Default FPS
        self._resolution = (640, 480)  # Default resolution
        
    def open(self) -> bool:
        """Open the USB camera"""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                print(f"Failed to open USB camera {self.device_id}")
                return False
            
            # Set camera properties if target resolution is specified
            if self.target_resolution:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_resolution[1])
            
            # Get actual FPS and resolution
            self._fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self._fps == 0:
                self._fps = 30.0  # Fallback
            
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._resolution = (width, height)
            
            print(f"USB Camera opened: {self.name}, Resolution: {self._resolution}, FPS: {self._fps}")
            self.reset_stats()
            
            return True
            
        except Exception as e:
            print(f"Error opening USB camera: {e}")
            return False
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Get the next frame from the USB camera"""
        if not self.is_opened():
            return False, None, 0.0
        
        timestamp = time.time()
        ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
            self._last_frame_time = timestamp
            
            # Resize if target resolution is different from camera resolution
            if self.target_resolution and frame.shape[1::-1] != self.target_resolution:
                frame = cv2.resize(frame, self.target_resolution)
            
            return True, frame, timestamp
        else:
            self.dropped_frames += 1
            return False, None, timestamp
    
    def release(self):
        """Release the USB camera"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print(f"USB Camera released: {self.name}")
    
    def is_opened(self) -> bool:
        """Check if the USB camera is opened"""
        return self.cap is not None and self.cap.isOpened()
    
    def get_fps(self) -> float:
        """Get the FPS of the USB camera"""
        return self._fps
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get the resolution of the USB camera"""
        return self._resolution
    
    def __del__(self):
        """Destructor to ensure camera is released"""
        self.release()

