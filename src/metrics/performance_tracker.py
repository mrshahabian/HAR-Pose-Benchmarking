"""Performance metrics tracker for FPS, latency, and dropped frames"""

import time
from typing import Dict, List
import numpy as np


class PerformanceTracker:
    """Track performance metrics during benchmarking"""
    
    def __init__(self, warmup_frames: int = 20):
        """
        Initialize performance tracker
        
        Args:
            warmup_frames: Number of frames to skip before collecting metrics
        """
        self.warmup_frames = warmup_frames
        
        # Timing data
        self.frame_times: List[float] = []
        self.capture_times: List[float] = []
        self.inference_times: List[float] = []
        self.end_to_end_times: List[float] = []
        
        # Counters
        self.frame_count = 0
        self.dropped_frames = 0
        self.warmup_complete = False
        
        # Start time
        self.start_time = None
        self.last_frame_time = None
        
    def start(self):
        """Start tracking"""
        self.start_time = time.time()
        self.last_frame_time = self.start_time
        self.frame_count = 0
        self.dropped_frames = 0
        self.warmup_complete = False
        
    def record_frame(self, capture_time: float, inference_time: float, 
                    end_to_end_time: float, dropped: bool = False):
        """
        Record metrics for a frame
        
        Args:
            capture_time: Time taken to capture frame (seconds)
            inference_time: Time taken for inference (seconds)
            end_to_end_time: Total time from capture to result (seconds)
            dropped: Whether frame was dropped
        """
        self.frame_count += 1
        
        # Skip warmup frames
        if self.frame_count <= self.warmup_frames:
            if self.frame_count == self.warmup_frames:
                self.warmup_complete = True
                print(f"Warmup complete ({self.warmup_frames} frames)")
            return
        
        if dropped:
            self.dropped_frames += 1
            return
        
        # Record timings
        current_time = time.time()
        if self.last_frame_time is not None:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
        
        self.capture_times.append(capture_time)
        self.inference_times.append(inference_time)
        self.end_to_end_times.append(end_to_end_time)
        
        self.last_frame_time = current_time
    
    def get_metrics(self) -> Dict:
        """
        Calculate and return performance metrics
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.frame_times:
            return {
                'fps': 0.0,
                'fps_std': 0.0,
                'latency_ms': 0.0,
                'latency_std_ms': 0.0,
                'capture_time_ms': 0.0,
                'inference_time_ms': 0.0,
                'end_to_end_latency_ms': 0.0,
                'dropped_frames': self.dropped_frames,
                'total_frames': self.frame_count,
                'processed_frames': 0
            }
        
        # Calculate FPS from frame times
        fps_values = [1.0 / ft for ft in self.frame_times if ft > 0]
        avg_fps = np.mean(fps_values) if fps_values else 0
        std_fps = np.std(fps_values) if fps_values else 0
        
        # Calculate latencies
        avg_capture_time = np.mean(self.capture_times) * 1000  # Convert to ms
        avg_inference_time = np.mean(self.inference_times) * 1000
        avg_end_to_end_time = np.mean(self.end_to_end_times) * 1000
        std_end_to_end_time = np.std(self.end_to_end_times) * 1000
        
        # Percentiles for latency
        p50_latency = np.percentile(self.end_to_end_times, 50) * 1000
        p95_latency = np.percentile(self.end_to_end_times, 95) * 1000
        p99_latency = np.percentile(self.end_to_end_times, 99) * 1000
        
        processed_frames = len(self.frame_times)
        
        return {
            'fps': float(avg_fps),
            'fps_std': float(std_fps),
            'fps_min': float(min(fps_values)) if fps_values else 0,
            'fps_max': float(max(fps_values)) if fps_values else 0,
            'latency_ms': float(avg_end_to_end_time),
            'latency_std_ms': float(std_end_to_end_time),
            'latency_p50_ms': float(p50_latency),
            'latency_p95_ms': float(p95_latency),
            'latency_p99_ms': float(p99_latency),
            'capture_time_ms': float(avg_capture_time),
            'inference_time_ms': float(avg_inference_time),
            'end_to_end_latency_ms': float(avg_end_to_end_time),
            'dropped_frames': self.dropped_frames,
            'total_frames': self.frame_count,
            'processed_frames': processed_frames,
            'drop_rate': self.dropped_frames / self.frame_count if self.frame_count > 0 else 0
        }
    
    def print_summary(self):
        """Print performance summary"""
        metrics = self.get_metrics()
        
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        print(f"FPS:                    {metrics['fps']:.2f} ± {metrics['fps_std']:.2f}")
        print(f"FPS Range:              [{metrics['fps_min']:.2f}, {metrics['fps_max']:.2f}]")
        print(f"Latency (avg):          {metrics['latency_ms']:.2f} ms")
        print(f"Latency (p50/p95/p99):  {metrics['latency_p50_ms']:.2f} / {metrics['latency_p95_ms']:.2f} / {metrics['latency_p99_ms']:.2f} ms")
        print(f"Capture Time:           {metrics['capture_time_ms']:.2f} ms")
        print(f"Inference Time:         {metrics['inference_time_ms']:.2f} ms")
        print(f"Dropped Frames:         {metrics['dropped_frames']} / {metrics['total_frames']} ({metrics['drop_rate']*100:.2f}%)")
        print("="*60 + "\n")

