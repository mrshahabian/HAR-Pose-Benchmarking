#!/usr/bin/env python3
"""Download sample video for testing"""

import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def main():
    """Download sample video"""
    print("="*70)
    print("Sample Video Downloader")
    print("="*70)
    print()
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "data" / "test_videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample videos (example URLs - replace with actual test videos)
    samples = {
        "person_walking.mp4": "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4",
        # Add more sample videos as needed
    }
    
    print("This script helps you download sample videos for testing.")
    print("You can also:")
    print("  1. Copy your own video files to: data/test_videos/")
    print("  2. Use a USB camera directly")
    print("  3. Use an RTSP stream from an IP camera")
    print()
    
    choice = input("Download sample video? (y/n): ")
    
    if choice.lower() == 'y':
        for filename, url in samples.items():
            output_path = output_dir / filename
            
            if output_path.exists():
                print(f"✓ {filename} already exists")
                continue
            
            print(f"Downloading {filename}...")
            try:
                download_url(url, output_path)
                print(f"✓ Downloaded: {output_path}")
            except Exception as e:
                print(f"✗ Failed to download {filename}: {e}")
                print(f"  You can manually download a video and place it in: {output_dir}")
    
    print()
    print("="*70)
    print("You can now run benchmarks with video files:")
    print(f"  python scripts/run_benchmark.py --sources file --file {output_dir}/your_video.mp4")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

