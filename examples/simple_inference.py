#!/usr/bin/env python3
"""Simple example of running YOLO pose inference on a single image or video"""

import sys
from pathlib import Path
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import YOLOEngine


def process_image(image_path: str, model_name: str = 'yolov11n-pose'):
    """Process a single image"""
    print(f"Processing image: {image_path}")
    
    # Initialize YOLO engine
    yolo = YOLOEngine(
        model_name=model_name,
        backend='pytorch',
        imgsz=(640, 640),
        conf_threshold=0.5
    )
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Run inference
    results, inference_time = yolo.inference(frame)
    
    print(f"\nResults:")
    print(f"  Inference time: {inference_time*1000:.2f} ms")
    print(f"  Detections: {results['detections']} person(s)")
    print(f"  FPS: {1.0/inference_time:.1f}")
    
    # Visualize
    vis_frame = yolo.visualize(frame, results)
    
    # Save result
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_result.jpg"
    cv2.imwrite(str(output_path), vis_frame)
    print(f"\nSaved result to: {output_path}")
    
    # Display (if in GUI environment)
    try:
        cv2.imshow('Pose Detection', vis_frame)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("(Display not available in this environment)")


def process_video(video_path: str, model_name: str = 'yolov11n-pose', max_frames: int = 100):
    """Process a video file"""
    print(f"Processing video: {video_path}")
    
    # Initialize YOLO engine
    yolo = YOLOEngine(
        model_name=model_name,
        backend='pytorch',
        imgsz=(640, 640),
        conf_threshold=0.5
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps} FPS")
    
    # Process frames
    frame_count = 0
    total_inference_time = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results, inference_time = yolo.inference(frame, verbose=False)
        total_inference_time += inference_time
        
        # Visualize
        vis_frame = yolo.visualize(frame, results)
        
        # Display
        cv2.imshow('Pose Detection', vis_frame)
        
        frame_count += 1
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print statistics
    avg_inference_time = total_inference_time / frame_count
    avg_fps = 1.0 / avg_inference_time
    
    print(f"\nProcessed {frame_count} frames")
    print(f"Average inference time: {avg_inference_time*1000:.2f} ms")
    print(f"Average FPS: {avg_fps:.1f}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple YOLO pose inference example')
    parser.add_argument('input', help='Path to image or video file')
    parser.add_argument('--model', default='yolov11n-pose', help='Model name')
    parser.add_argument('--max-frames', type=int, default=100, help='Max frames for video')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1
    
    # Detect if image or video
    ext = input_path.suffix.lower()
    
    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        process_image(str(input_path), args.model)
    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
        process_video(str(input_path), args.model, args.max_frames)
    else:
        print(f"Unknown file type: {ext}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

