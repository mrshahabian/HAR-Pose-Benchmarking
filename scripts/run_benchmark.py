#!/usr/bin/env python3
"""CLI script to run YOLO pose estimation benchmarks"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark import BenchmarkRunner, BenchmarkConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run YOLO Pose Estimation Benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks from config file
  python scripts/run_benchmark.py --config configs/benchmark_config.yaml
  
  # Run specific models only
  python scripts/run_benchmark.py --models yolov11n-pose yolov11s-pose
  
  # Run specific backend
  python scripts/run_benchmark.py --backends pytorch --duration 30
  
  # Run with USB camera only
  python scripts/run_benchmark.py --sources usb --device 0
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/benchmark_config.yaml',
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        help='Model names to benchmark (e.g., yolov11n-pose yolov11s-pose)'
    )
    
    parser.add_argument(
        '--backends', '-b',
        nargs='+',
        choices=['pytorch', 'tensorrt', 'openvino', 'onnx'],
        help='Backends to use'
    )
    
    parser.add_argument(
        '--resolutions', '-r',
        nargs='+',
        help='Resolutions as WIDTHxHEIGHT (e.g., 640x640 960x960)'
    )
    
    parser.add_argument(
        '--sources', '-s',
        choices=['usb', 'rtsp', 'file'],
        help='Video source type to use'
    )
    
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='USB camera device ID (for usb source)'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='Video file path (for file source)'
    )
    
    parser.add_argument(
        '--rtsp-url',
        type=str,
        help='RTSP stream URL (for rtsp source)'
    )
    
    parser.add_argument(
        '--duration', '-d',
        type=int,
        help='Duration of each benchmark in seconds'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Disable saving sample visualizations'
    )
    
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt (auto-confirm)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = BenchmarkConfig(args.config)
    
    # Override configuration with command line arguments
    if args.models:
        config.set('models', args.models)
    
    if args.backends:
        config.set('backends', args.backends)
    
    if args.resolutions:
        # Parse resolution strings (e.g., "640x640" -> [640, 640])
        resolutions = []
        for res_str in args.resolutions:
            w, h = map(int, res_str.split('x'))
            resolutions.append([w, h])
        config.set('resolutions', resolutions)
    
    if args.sources:
        # Create source configuration
        sources = []
        if args.sources == 'usb':
            sources.append({
                'type': 'usb',
                'device': args.device,
                'name': f'USB_Camera_{args.device}'
            })
        elif args.sources == 'file':
            if not args.file:
                print("Error: --file required for file source")
                return 1
            sources.append({
                'type': 'file',
                'path': args.file,
                'loop': True,
                'name': 'Video_File'
            })
        elif args.sources == 'rtsp':
            if not args.rtsp_url:
                print("Error: --rtsp-url required for rtsp source")
                return 1
            sources.append({
                'type': 'rtsp',
                'url': args.rtsp_url,
                'name': 'RTSP_Stream'
            })
        
        config.set('sources', sources)
    
    if args.duration:
        config.set('duration_seconds', args.duration)
    
    if args.output_dir:
        config.set('output_dir', args.output_dir)
    
    if args.no_visualizations:
        config.set('save_visualizations', False)
    
    # Create and run benchmark
    print("\n" + "="*70)
    print("YOLO POSE ESTIMATION BENCHMARK")
    print("="*70)
    
    runner = BenchmarkRunner(config)
    
    try:
        results = runner.run_all_benchmarks(skip_confirmation=args.yes)
        
        if results:
            print(f"\n✓ Benchmark completed successfully!")
            print(f"  Results saved to: {config.get('output_dir')}")
            return 0
        else:
            print("\n✗ No benchmarks completed")
            return 1
    
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        return 130
    
    except Exception as e:
        print(f"\n✗ Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

