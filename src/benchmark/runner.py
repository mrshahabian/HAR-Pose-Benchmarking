"""Main benchmark runner orchestrating all components"""

import time
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
from tqdm import tqdm

from ..input import VideoSource, USBSource, RTSPSource, FileSource
from ..inference import YOLOEngine
from ..metrics import PerformanceTracker, AccuracyEvaluator, SystemMonitor
from .config import BenchmarkConfig


class BenchmarkRunner:
    """Main benchmark orchestrator"""
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize benchmark runner
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.results = []
        
        # Create output directory
        self.output_dir = Path(self.config.get('output_dir', 'results/benchmarks'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get device info
        self.device_info = config.get_device_info()
        
        print("Benchmark Runner initialized")
        print(f"Device: {self.device_info.get('machine', 'Unknown')}")
        print(f"GPU: {self.device_info.get('gpu', 'None')}")
    
    def create_video_source(self, source_config: Dict) -> Optional[VideoSource]:
        """
        Create video source from configuration
        
        Args:
            source_config: Video source configuration
            
        Returns:
            VideoSource instance or None if creation fails
        """
        source_type = source_config.get('type', 'usb')
        
        try:
            if source_type == 'usb':
                device_id = source_config.get('device', 0)
                name = source_config.get('name', 'USB_Camera')
                return USBSource(device_id=device_id, name=name)
            
            elif source_type == 'rtsp':
                url = source_config.get('url')
                name = source_config.get('name', 'RTSP_Stream')
                return RTSPSource(rtsp_url=url, name=name)
            
            elif source_type == 'file':
                path = source_config.get('path')
                loop = source_config.get('loop', True)
                name = source_config.get('name', 'Video_File')
                return FileSource(file_path=path, loop=loop, name=name)
            
            else:
                print(f"Unknown source type: {source_type}")
                return None
                
        except Exception as e:
            print(f"Error creating video source: {e}")
            return None
    
    def run_single_benchmark(self, model_name: str, backend: str, 
                           resolution: List[int], source_config: Dict) -> Dict:
        """
        Run a single benchmark configuration
        
        Args:
            model_name: YOLO model name
            backend: Backend type
            resolution: Input resolution [width, height]
            source_config: Video source configuration
            
        Returns:
            Dictionary containing benchmark results
        """
        print("\n" + "="*70)
        print(f"BENCHMARK: {model_name} | {backend} | {resolution} | {source_config.get('name', 'Unknown')}")
        print("="*70)
        
        # Create video source
        video_source = self.create_video_source(source_config)
        if video_source is None:
            print("Failed to create video source, skipping...")
            return None
        
        # Set target resolution
        video_source.target_resolution = tuple(resolution)
        
        # Open video source
        if not video_source.open():
            print("Failed to open video source, skipping...")
            return None
        
        # Initialize YOLO engine
        try:
            yolo_engine = YOLOEngine(
                model_name=model_name,
                backend=backend,
                imgsz=tuple(resolution),
                conf_threshold=self.config.get('confidence_threshold', 0.5)
            )
        except Exception as e:
            print(f"Failed to initialize YOLO engine: {e}")
            video_source.release()
            return None
        
        # Initialize metrics
        perf_tracker = PerformanceTracker(
            warmup_frames=self.config.get('warmup_frames', 20)
        )
        system_monitor = SystemMonitor(interval=1.0)
        
        # Accuracy evaluator (optional)
        accuracy_evaluator = None
        if self.config.get('annotations', {}).get('enabled', False):
            ann_path = self.config.get('annotations', {}).get('coco_json_path')
            if ann_path:
                accuracy_evaluator = AccuracyEvaluator(ann_path)
        
        # Start monitoring
        perf_tracker.start()
        system_monitor.start()
        
        # Run benchmark
        duration = self.config.get('duration_seconds', 60)
        start_time = time.time()
        frame_count = 0
        
        # For saving sample visualizations
        sample_frames = []
        save_vis = self.config.get('save_visualizations', True)
        
        print(f"Running benchmark for {duration} seconds...")
        
        with tqdm(total=duration, desc="Progress", unit="s") as pbar:
            while time.time() - start_time < duration:
                loop_start = time.time()
                
                # Capture frame
                capture_start = time.time()
                success, frame, timestamp = video_source.get_frame()
                capture_time = time.time() - capture_start
                
                if not success or frame is None:
                    perf_tracker.record_frame(capture_time, 0, 0, dropped=True)
                    continue
                
                # Run inference
                e2e_start = time.time()
                results, inference_time = yolo_engine.inference(frame, verbose=False)
                end_to_end_time = time.time() - e2e_start
                
                # Record metrics
                perf_tracker.record_frame(capture_time, inference_time, end_to_end_time, dropped=False)
                
                # Add to accuracy evaluator if enabled
                if accuracy_evaluator and results['keypoints']:
                    accuracy_evaluator.add_prediction(
                        image_id=frame_count,
                        keypoints=results['keypoints'],
                        boxes=results['boxes'],
                        scores=results['confidences']
                    )
                
                # Save sample visualization (first 5 frames after warmup)
                if save_vis and perf_tracker.warmup_complete and len(sample_frames) < 5:
                    vis_frame = yolo_engine.visualize(frame, results)
                    sample_frames.append(vis_frame)
                
                frame_count += 1
                
                # Update progress bar
                elapsed = time.time() - start_time
                pbar.update(elapsed - pbar.n)
        
        # Stop monitoring
        system_monitor.stop()
        
        # Collect results
        perf_metrics = perf_tracker.get_metrics()
        system_metrics = system_monitor.get_metrics()
        
        # Accuracy metrics (if available)
        accuracy_metrics = {}
        if accuracy_evaluator:
            accuracy_metrics = accuracy_evaluator.evaluate()
        
        # Combine all metrics
        result = {
            'timestamp': datetime.now().isoformat(),
            'device_info': self.device_info,
            'model': model_name,
            'backend': backend,
            'resolution': resolution,
            'source': source_config.get('name', 'Unknown'),
            'source_type': source_config.get('type', 'unknown'),
            'performance': perf_metrics,
            'system': system_metrics,
            'accuracy': accuracy_metrics
        }
        
        # Print summaries
        perf_tracker.print_summary()
        system_monitor.print_summary()
        if accuracy_evaluator:
            accuracy_evaluator.print_summary()
        
        # Save sample visualizations
        if save_vis and sample_frames:
            self._save_visualizations(sample_frames, model_name, backend, resolution, source_config)
        
        # Cleanup
        video_source.release()
        
        return result
    
    def _save_visualizations(self, frames: List[np.ndarray], model_name: str,
                           backend: str, resolution: List[int], source_config: Dict):
        """Save sample visualization frames"""
        vis_dir = self.output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        source_name = source_config.get('name', 'unknown').replace(' ', '_')
        
        for i, frame in enumerate(frames):
            filename = f"{model_name}_{backend}_{resolution[0]}x{resolution[1]}_{source_name}_{timestamp}_frame{i}.jpg"
            filepath = vis_dir / filename
            cv2.imwrite(str(filepath), frame)
        
        print(f"Saved {len(frames)} visualization frames to {vis_dir}")
    
    def run_all_benchmarks(self, skip_confirmation: bool = False) -> List[Dict]:
        """
        Run all benchmark configurations
        
        Args:
            skip_confirmation: Skip user confirmation prompt
        
        Returns:
            List of benchmark results
        """
        # Validate configuration
        if not self.config.validate():
            print("Invalid configuration, aborting benchmarks")
            return []
        
        # Print configuration
        self.config.print_summary()
        
        # Filter backends by device
        backends = self.config.filter_backends_by_device()
        
        # Get benchmark parameters
        models = self.config.get('models', [])
        resolutions = self.config.get('resolutions', [[640, 640]])
        sources = self.config.get('sources', [])
        
        # Calculate total benchmarks
        total_benchmarks = len(models) * len(backends) * len(resolutions) * len(sources)
        print(f"\nTotal benchmarks to run: {total_benchmarks}")
        print(f"Estimated time: ~{total_benchmarks * self.config.get('duration_seconds', 60) / 60:.1f} minutes\n")
        
        # Confirm with user
        if not skip_confirmation:
            input("Press Enter to start benchmarking...")
        
        # Run all combinations
        benchmark_count = 0
        for model in models:
            for backend in backends:
                for resolution in resolutions:
                    for source_config in sources:
                        benchmark_count += 1
                        print(f"\n[{benchmark_count}/{total_benchmarks}] Running benchmark...")
                        
                        try:
                            result = self.run_single_benchmark(
                                model_name=model,
                                backend=backend,
                                resolution=resolution,
                                source_config=source_config
                            )
                            
                            if result:
                                self.results.append(result)
                                # Save intermediate results
                                self.save_results()
                        
                        except Exception as e:
                            print(f"Error running benchmark: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
        
        print(f"\n{'='*70}")
        print(f"BENCHMARKING COMPLETE")
        print(f"Completed: {len(self.results)} / {total_benchmarks} benchmarks")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*70}\n")
        
        return self.results
    
    def save_results(self):
        """Save benchmark results to JSON and CSV"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON
        if self.config.get('export_json', True):
            json_path = self.output_dir / f'benchmark_results_{timestamp}.json'
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"Results saved to {json_path}")
        
        # Save CSV (flattened)
        if self.config.get('export_csv', True) and self.results:
            csv_path = self.output_dir / f'benchmark_results_{timestamp}.csv'
            self._save_csv(csv_path)
            print(f"Results saved to {csv_path}")
    
    def _save_csv(self, csv_path: Path):
        """Save results to CSV with flattened structure"""
        if not self.results:
            return
        
        # Flatten nested dictionaries
        flattened_results = []
        for result in self.results:
            flat = {
                'timestamp': result['timestamp'],
                'device': result['device_info'].get('machine', 'Unknown'),
                'gpu': result['device_info'].get('gpu', 'None'),
                'model': result['model'],
                'backend': result['backend'],
                'resolution': f"{result['resolution'][0]}x{result['resolution'][1]}",
                'source': result['source'],
                'source_type': result['source_type'],
            }
            
            # Add performance metrics
            for key, value in result['performance'].items():
                flat[f'perf_{key}'] = value
            
            # Add system metrics
            for key, value in result['system'].items():
                flat[f'sys_{key}'] = value
            
            # Add accuracy metrics
            for key, value in result.get('accuracy', {}).items():
                flat[f'acc_{key}'] = value
            
            flattened_results.append(flat)
        
        # Write CSV
        if flattened_results:
            keys = flattened_results[0].keys()
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(flattened_results)

