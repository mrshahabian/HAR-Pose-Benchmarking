"""Video file source"""

import cv2
import time
from typing import Optional, Tuple
import numpy as np
from .video_source import VideoSource


class FileSource(VideoSource):
    """Video file source using OpenCV"""
    
    def __init__(self, file_path: str, target_resolution: Optional[Tuple[int, int]] = None,
                 loop: bool = True, name: str = "Video_File"):
        """
        Initialize video file source
        
        Args:
            file_path: Path to video file
            target_resolution: Target resolution as (width, height), None for native
            loop: Whether to loop the video when it reaches the end
            name: Name for this source
        """
        super().__init__(name)
        self.file_path = file_path
        self.target_resolution = target_resolution
        self.loop = loop
        self.cap = None
        self._fps = 30.0
        self._resolution = (640, 480)
        self._total_frames = 0
        
    def open(self) -> bool:
        """Open the video file"""
        try:
            self.cap = cv2.VideoCapture(self.file_path)
            
            if not self.cap.isOpened():
                print(f"Failed to open video file: {self.file_path}")
                return False
            
            # Get video properties
            self._fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self._fps == 0:
                self._fps = 30.0  # Fallback
            
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._resolution = (width, height)
            
            self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video File opened: {self.name}")
            print(f"  Path: {self.file_path}")
            print(f"  Resolution: {self._resolution}, FPS: {self._fps}, Total Frames: {self._total_frames}")
            
            self.reset_stats()
            
            return True
            
        except Exception as e:
            print(f"Error opening video file: {e}")
            return False
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Get the next frame from the video file"""
        if not self.is_opened():
            return False, None, 0.0
        
        timestamp = time.time()
        ret, frame = self.cap.read()
        
        # If end of video and looping is enabled, restart
        if not ret and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
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
            return False, None, timestamp
    
    def release(self):
        """Release the video file"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print(f"Video File released: {self.name}")
    
    def is_opened(self) -> bool:
        """Check if the video file is opened"""
        return self.cap is not None and self.cap.isOpened()
    
    def get_fps(self) -> float:
        """Get the FPS of the video file"""
        return self._fps
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get the resolution of the video file"""
        return self._resolution
    
    def get_total_frames(self) -> int:
        """Get the total number of frames in the video"""
        return self._total_frames
    
    def __del__(self):
        """Destructor to ensure file is released"""
        self.release()

