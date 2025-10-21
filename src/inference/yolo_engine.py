"""YOLO inference engine wrapper"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from ultralytics import YOLO
from .backend_factory import BackendFactory


class YOLOEngine:
    """Unified interface for YOLO pose estimation inference"""
    
    def __init__(self, model_name: str, backend: str = 'pytorch',
                 imgsz: Tuple[int, int] = (640, 640),
                 conf_threshold: float = 0.5,
                 device: str = 'auto'):
        """
        Initialize YOLO inference engine
        
        Args:
            model_name: Model name (e.g., 'yolov11n-pose')
            backend: Backend type ('pytorch', 'tensorrt', 'openvino', 'onnx')
            imgsz: Input image size (width, height)
            conf_threshold: Confidence threshold for detections
            device: Device to use ('cpu', 'cuda', 'auto')
        """
        self.model_name = model_name
        self.backend = backend
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.device = device
        
        # Initialize backend factory
        self.factory = BackendFactory()
        
        # Load model
        self.model = None
        self.load_model()
        
        # Statistics
        self.inference_count = 0
        self.total_inference_time = 0.0
        
    def load_model(self):
        """Load the YOLO model"""
        try:
            self.model = self.factory.load_model(
                self.model_name,
                backend=self.backend,
                imgsz=self.imgsz,
                device=self.device
            )
            print(f"YOLO model loaded: {self.model_name} ({self.backend})")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for inference
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            Preprocessed frame
        """
        # Resize to target size if needed
        if frame.shape[1::-1] != self.imgsz:
            frame = cv2.resize(frame, self.imgsz)
        
        return frame
    
    def inference(self, frame: np.ndarray, verbose: bool = False) -> Tuple[Dict, float]:
        """
        Run inference on a frame
        
        Args:
            frame: Input frame (BGR format)
            verbose: Print verbose output
            
        Returns:
            Tuple of (results dict, inference time in seconds)
        """
        # Preprocess
        processed_frame = self.preprocess(frame)
        
        # Run inference with timing
        start_time = time.time()
        
        results = self.model(
            processed_frame,
            conf=self.conf_threshold,
            verbose=verbose,
            imgsz=self.imgsz
        )
        
        inference_time = time.time() - start_time
        
        # Update statistics
        self.inference_count += 1
        self.total_inference_time += inference_time
        
        # Parse results
        parsed_results = self.parse_results(results)
        
        return parsed_results, inference_time
    
    def parse_results(self, results) -> Dict:
        """
        Parse YOLO results into a standardized format
        
        Args:
            results: Raw YOLO results
            
        Returns:
            Dictionary containing parsed results
        """
        parsed = {
            'detections': [],
            'keypoints': [],
            'boxes': [],
            'confidences': []
        }
        
        if len(results) == 0:
            return parsed
        
        result = results[0]  # Single image inference
        
        # Extract boxes
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confidences = result.boxes.conf.cpu().numpy()
            
            parsed['boxes'] = boxes.tolist()
            parsed['confidences'] = confidences.tolist()
        
        # Extract keypoints for pose estimation
        if result.keypoints is not None and len(result.keypoints) > 0:
            keypoints = result.keypoints.xy.cpu().numpy()  # [num_person, num_keypoints, 2]
            keypoint_conf = result.keypoints.conf.cpu().numpy()  # [num_person, num_keypoints]
            
            # Format: list of persons, each with list of keypoints
            for person_idx in range(len(keypoints)):
                person_keypoints = []
                for kp_idx in range(len(keypoints[person_idx])):
                    kp = keypoints[person_idx][kp_idx]
                    conf = keypoint_conf[person_idx][kp_idx]
                    
                    person_keypoints.append({
                        'x': float(kp[0]),
                        'y': float(kp[1]),
                        'confidence': float(conf)
                    })
                
                parsed['keypoints'].append(person_keypoints)
        
        parsed['detections'] = len(parsed['boxes'])
        
        return parsed
    
    def visualize(self, frame: np.ndarray, results: Dict, 
                  draw_boxes: bool = True, draw_keypoints: bool = True) -> np.ndarray:
        """
        Visualize detection results on frame
        
        Args:
            frame: Input frame
            results: Parsed results from inference
            draw_boxes: Whether to draw bounding boxes
            draw_keypoints: Whether to draw keypoints
            
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        
        # Draw bounding boxes
        if draw_boxes and len(results['boxes']) > 0:
            for box, conf in zip(results['boxes'], results['confidences']):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_frame, f"{conf:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw keypoints
        if draw_keypoints and len(results['keypoints']) > 0:
            # COCO keypoint connections (skeleton)
            skeleton = [
                [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                [2, 4], [3, 5], [4, 6], [5, 7]
            ]
            
            for person_keypoints in results['keypoints']:
                # Draw keypoints
                for kp in person_keypoints:
                    if kp['confidence'] > 0.5:
                        x, y = int(kp['x']), int(kp['y'])
                        cv2.circle(vis_frame, (x, y), 3, (0, 0, 255), -1)
                
                # Draw skeleton connections
                for connection in skeleton:
                    kp1_idx, kp2_idx = connection[0] - 1, connection[1] - 1
                    
                    if (kp1_idx < len(person_keypoints) and 
                        kp2_idx < len(person_keypoints)):
                        
                        kp1 = person_keypoints[kp1_idx]
                        kp2 = person_keypoints[kp2_idx]
                        
                        if kp1['confidence'] > 0.5 and kp2['confidence'] > 0.5:
                            pt1 = (int(kp1['x']), int(kp1['y']))
                            pt2 = (int(kp2['x']), int(kp2['y']))
                            cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 2)
        
        return vis_frame
    
    def get_stats(self) -> Dict:
        """
        Get inference statistics
        
        Returns:
            Dictionary with statistics
        """
        avg_inference_time = (self.total_inference_time / self.inference_count 
                             if self.inference_count > 0 else 0)
        
        return {
            'model_name': self.model_name,
            'backend': self.backend,
            'inference_count': self.inference_count,
            'total_inference_time': self.total_inference_time,
            'avg_inference_time': avg_inference_time,
            'avg_fps': 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        }
    
    def reset_stats(self):
        """Reset inference statistics"""
        self.inference_count = 0
        self.total_inference_time = 0.0

