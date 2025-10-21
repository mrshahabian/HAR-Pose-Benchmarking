"""Abstract base class for video sources"""

from abc import ABC, abstractmethod
import time
from typing import Optional, Tuple
import numpy as np


class VideoSource(ABC):
    """Abstract interface for video input sources"""
    
    def __init__(self, name: str = "VideoSource"):
        self.name = name
        self.frame_count = 0
        self.dropped_frames = 0
        self.start_time = None
        self._last_frame_time = None
        
    @abstractmethod
    def open(self) -> bool:
        """Open the video source
        
        Returns:
            bool: True if successfully opened, False otherwise
        """
        pass
    
    @abstractmethod
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Get the next frame from the video source
        
        Returns:
            Tuple containing:
                - success (bool): True if frame was retrieved successfully
                - frame (np.ndarray or None): The frame data
                - timestamp (float): Timestamp when frame was captured
        """
        pass
    
    @abstractmethod
    def release(self):
        """Release the video source and cleanup resources"""
        pass
    
    @abstractmethod
    def is_opened(self) -> bool:
        """Check if the video source is currently open
        
        Returns:
            bool: True if opened, False otherwise
        """
        pass
    
    @abstractmethod
    def get_fps(self) -> float:
        """Get the frames per second of the video source
        
        Returns:
            float: FPS value
        """
        pass
    
    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """Get the resolution (width, height) of the video source
        
        Returns:
            Tuple[int, int]: (width, height)
        """
        pass
    
    def reset_stats(self):
        """Reset frame statistics"""
        self.frame_count = 0
        self.dropped_frames = 0
        self.start_time = time.time()
        self._last_frame_time = None
    
    def get_stats(self) -> dict:
        """Get statistics about the video source
        
        Returns:
            dict: Statistics including frame count, dropped frames, etc.
        """
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        actual_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'name': self.name,
            'frame_count': self.frame_count,
            'dropped_frames': self.dropped_frames,
            'elapsed_time': elapsed_time,
            'actual_fps': actual_fps
        }

