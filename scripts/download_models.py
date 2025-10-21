#!/usr/bin/env python3
"""Download YOLO pose estimation models"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import BackendFactory


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Download YOLO pose estimation models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        default=['yolov11n-pose', 'yolov11s-pose', 'yolov11m-pose'],
        help='Model names to download'
    )
    
    parser.add_argument(
        '--export', '-e',
        nargs='+',
        choices=['onnx', 'tensorrt', 'openvino'],
        help='Export models to specific formats'
    )
    
    parser.add_argument(
        '--resolution', '-r',
        type=str,
        default='640x640',
        help='Resolution for export (e.g., 640x640)'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory to store models'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    resolution = (width, height)
    
    # Create backend factory
    factory = BackendFactory(model_dir=args.model_dir)
    
    print("="*70)
    print("YOLO MODEL DOWNLOADER")
    print("="*70)
    print(f"Models: {', '.join(args.models)}")
    print(f"Model directory: {args.model_dir}")
    if args.export:
        print(f"Export formats: {', '.join(args.export)}")
        print(f"Export resolution: {resolution}")
    print("="*70 + "\n")
    
    # Download models
    for model_name in args.models:
        print(f"\nDownloading {model_name}...")
        try:
            model_path = factory.download_model(model_name)
            print(f"✓ Downloaded: {model_path}")
            
            # Export if requested
            if args.export:
                for export_format in args.export:
                    print(f"  Exporting to {export_format}...")
                    try:
                        exported_path = factory.export_model(
                            model_name,
                            backend=export_format,
                            imgsz=resolution,
                            force=False
                        )
                        print(f"  ✓ Exported: {exported_path}")
                    except Exception as e:
                        print(f"  ✗ Export failed: {e}")
        
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")
            continue
    
    # Cleanup: Remove any .pt files in current directory that might be temporary downloads
    print("\nCleaning up temporary files...")
    import os
    from pathlib import Path
    cwd = Path.cwd()
    for pt_file in cwd.glob("*.pt"):
        # Only remove if it's in the root and we have it in models/
        model_dir_path = Path(args.model_dir) / pt_file.name
        if pt_file.parent == cwd and model_dir_path.exists():
            try:
                pt_file.unlink()
                print(f"  Removed temporary file: {pt_file.name}")
            except:
                pass
    
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    print(f"Models saved to: {args.model_dir}/")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

