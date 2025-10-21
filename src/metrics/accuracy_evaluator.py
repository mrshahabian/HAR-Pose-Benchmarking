"""Accuracy evaluation for pose estimation (mAP, PCK, OKS)"""

import json
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path


class AccuracyEvaluator:
    """Evaluate pose estimation accuracy using COCO metrics"""
    
    # COCO keypoint sigmas for OKS calculation
    COCO_KEYPOINT_SIGMAS = np.array([
        .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89
    ]) / 10.0
    
    def __init__(self, annotations_path: Optional[str] = None):
        """
        Initialize accuracy evaluator
        
        Args:
            annotations_path: Path to COCO format annotations JSON file
        """
        self.annotations_path = annotations_path
        self.ground_truth = None
        self.predictions = []
        
        if annotations_path and Path(annotations_path).exists():
            self.load_annotations(annotations_path)
    
    def load_annotations(self, annotations_path: str):
        """
        Load ground truth annotations in COCO format
        
        Args:
            annotations_path: Path to annotations JSON file
        """
        try:
            with open(annotations_path, 'r') as f:
                self.ground_truth = json.load(f)
            print(f"Loaded ground truth annotations from {annotations_path}")
        except Exception as e:
            print(f"Error loading annotations: {e}")
            self.ground_truth = None
    
    def add_prediction(self, image_id: int, keypoints: List[List[Dict]], 
                      boxes: List[List[float]], scores: List[float]):
        """
        Add a prediction for evaluation
        
        Args:
            image_id: Image identifier
            keypoints: List of keypoints per person [person][keypoint]['x', 'y', 'confidence']
            boxes: Bounding boxes [person][x1, y1, x2, y2]
            scores: Detection confidence scores [person]
        """
        for i, (person_kp, box, score) in enumerate(zip(keypoints, boxes, scores)):
            # Convert keypoints to COCO format: [x1, y1, v1, x2, y2, v2, ...]
            kp_flat = []
            for kp in person_kp:
                kp_flat.extend([kp['x'], kp['y'], 2 if kp['confidence'] > 0.5 else 1])
            
            self.predictions.append({
                'image_id': image_id,
                'category_id': 1,  # Person category
                'keypoints': kp_flat,
                'score': score,
                'bbox': box
            })
    
    def calculate_oks(self, pred_kp: np.ndarray, gt_kp: np.ndarray, 
                     gt_area: float) -> float:
        """
        Calculate Object Keypoint Similarity (OKS)
        
        Args:
            pred_kp: Predicted keypoints [num_keypoints, 3] (x, y, visibility)
            gt_kp: Ground truth keypoints [num_keypoints, 3]
            gt_area: Area of ground truth bounding box
            
        Returns:
            OKS score (0-1)
        """
        # Reshape to [num_keypoints, 3]
        pred_kp = pred_kp.reshape(-1, 3)
        gt_kp = gt_kp.reshape(-1, 3)
        
        # Calculate distances
        dx = pred_kp[:, 0] - gt_kp[:, 0]
        dy = pred_kp[:, 1] - gt_kp[:, 1]
        distances = np.sqrt(dx**2 + dy**2)
        
        # Visibility flags (2 = visible, 1 = labeled but not visible, 0 = not labeled)
        visible = gt_kp[:, 2] > 0
        
        # Calculate OKS
        s = np.sqrt(gt_area)
        k = self.COCO_KEYPOINT_SIGMAS[:len(distances)]
        
        # e = d^2 / (2 * s^2 * k^2)
        e = distances[visible] / (2 * s * k[visible]**2 + 1e-9)
        oks = np.sum(np.exp(-e)) / np.sum(visible) if np.sum(visible) > 0 else 0
        
        return float(oks)
    
    def calculate_pck(self, pred_kp: np.ndarray, gt_kp: np.ndarray,
                     threshold: float = 0.5) -> float:
        """
        Calculate Percentage of Correct Keypoints (PCK)
        
        Args:
            pred_kp: Predicted keypoints [num_keypoints, 3]
            gt_kp: Ground truth keypoints [num_keypoints, 3]
            threshold: Distance threshold as fraction of bounding box diagonal
            
        Returns:
            PCK score (0-1)
        """
        pred_kp = pred_kp.reshape(-1, 3)
        gt_kp = gt_kp.reshape(-1, 3)
        
        # Calculate distances
        dx = pred_kp[:, 0] - gt_kp[:, 0]
        dy = pred_kp[:, 1] - gt_kp[:, 1]
        distances = np.sqrt(dx**2 + dy**2)
        
        # Visibility
        visible = gt_kp[:, 2] > 0
        
        # Calculate bounding box diagonal for normalization
        gt_visible = gt_kp[visible, :2]
        if len(gt_visible) < 2:
            return 0.0
        
        bbox_diag = np.sqrt(
            (np.max(gt_visible[:, 0]) - np.min(gt_visible[:, 0]))**2 +
            (np.max(gt_visible[:, 1]) - np.min(gt_visible[:, 1]))**2
        )
        
        # Count correct keypoints
        normalized_dist = distances[visible] / (bbox_diag + 1e-9)
        correct = np.sum(normalized_dist < threshold)
        total = np.sum(visible)
        
        pck = correct / total if total > 0 else 0
        return float(pck)
    
    def evaluate(self) -> Dict:
        """
        Evaluate predictions against ground truth
        
        Returns:
            Dictionary containing accuracy metrics
        """
        if self.ground_truth is None or not self.predictions:
            return {
                'map_keypoint': 0.0,
                'oks_avg': 0.0,
                'oks_median': 0.0,
                'pck_0.5': 0.0,
                'pck_0.2': 0.0,
                'num_predictions': len(self.predictions),
                'num_ground_truth': 0,
                'num_matched': 0
            }
        
        # This is a simplified evaluation
        # Full implementation would use pycocotools for proper mAP calculation
        
        oks_scores = []
        pck_05_scores = []
        pck_02_scores = []
        
        # Match predictions to ground truth (simplified)
        gt_annotations = self.ground_truth.get('annotations', [])
        
        for pred in self.predictions:
            # Find matching ground truth annotation
            matching_gt = None
            for gt in gt_annotations:
                if gt['image_id'] == pred['image_id']:
                    matching_gt = gt
                    break
            
            if matching_gt is None:
                continue
            
            # Convert to numpy arrays
            pred_kp = np.array(pred['keypoints'])
            gt_kp = np.array(matching_gt['keypoints'])
            gt_area = matching_gt.get('area', 1.0)
            
            # Calculate metrics
            oks = self.calculate_oks(pred_kp, gt_kp, gt_area)
            pck_05 = self.calculate_pck(pred_kp, gt_kp, threshold=0.5)
            pck_02 = self.calculate_pck(pred_kp, gt_kp, threshold=0.2)
            
            oks_scores.append(oks)
            pck_05_scores.append(pck_05)
            pck_02_scores.append(pck_02)
        
        # Calculate averages
        metrics = {
            'map_keypoint': float(np.mean(oks_scores)) if oks_scores else 0.0,
            'oks_avg': float(np.mean(oks_scores)) if oks_scores else 0.0,
            'oks_median': float(np.median(oks_scores)) if oks_scores else 0.0,
            'pck_0.5': float(np.mean(pck_05_scores)) if pck_05_scores else 0.0,
            'pck_0.2': float(np.mean(pck_02_scores)) if pck_02_scores else 0.0,
            'num_predictions': len(self.predictions),
            'num_ground_truth': len(gt_annotations),
            'num_matched': len(oks_scores)
        }
        
        return metrics
    
    def reset(self):
        """Reset predictions"""
        self.predictions.clear()
    
    def print_summary(self):
        """Print accuracy metrics summary"""
        metrics = self.evaluate()
        
        print("\n" + "="*60)
        print("ACCURACY METRICS")
        print("="*60)
        print(f"mAP (Keypoint):         {metrics['map_keypoint']:.4f}")
        print(f"OKS (avg):              {metrics['oks_avg']:.4f}")
        print(f"OKS (median):           {metrics['oks_median']:.4f}")
        print(f"PCK@0.5:                {metrics['pck_0.5']:.4f}")
        print(f"PCK@0.2:                {metrics['pck_0.2']:.4f}")
        print(f"Matched Predictions:    {metrics['num_matched']} / {metrics['num_predictions']}")
        print("="*60 + "\n")

