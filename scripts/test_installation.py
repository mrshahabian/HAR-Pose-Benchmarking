#!/usr/bin/env python3
"""Test script to verify installation and environment"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    tests = {
        "OpenCV": lambda: __import__("cv2"),
        "NumPy": lambda: __import__("numpy"),
        "PyTorch": lambda: __import__("torch"),
        "Ultralytics": lambda: __import__("ultralytics"),
        "Pandas": lambda: __import__("pandas"),
        "Matplotlib": lambda: __import__("matplotlib"),
        "Seaborn": lambda: __import__("seaborn"),
        "YAML": lambda: __import__("yaml"),
        "psutil": lambda: __import__("psutil"),
    }
    
    results = {}
    for name, import_func in tests.items():
        try:
            import_func()
            results[name] = "✓"
            print(f"  ✓ {name}")
        except ImportError as e:
            results[name] = "✗"
            print(f"  ✗ {name}: {e}")
    
    return all(v == "✓" for v in results.values())


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"  ✓ CUDA available")
            print(f"    Device: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA version: {torch.version.cuda}")
        else:
            print("  ⚠ CUDA not available (CPU-only mode)")
        return True
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False


def test_nvidia_gpu():
    """Test NVIDIA GPU monitoring"""
    print("\nTesting NVIDIA GPU monitoring...")
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        print(f"  ✓ NVIDIA GPU detected: {name}")
        pynvml.nvmlShutdown()
        return True
    except Exception as e:
        print(f"  ⚠ NVIDIA GPU monitoring not available: {e}")
        return True  # Not critical


def test_backends():
    """Test available inference backends"""
    print("\nTesting inference backends...")
    
    from src.inference import BackendFactory
    available = BackendFactory.get_available_backends()
    
    print(f"  Available backends: {', '.join(available)}")
    
    for backend in ['pytorch', 'onnx', 'tensorrt', 'openvino']:
        status = "✓" if backend in available else "✗"
        print(f"    {status} {backend}")
    
    return 'pytorch' in available


def test_camera():
    """Test camera access"""
    print("\nTesting camera access...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("  ✓ Camera accessible (device 0)")
            ret, frame = cap.read()
            if ret:
                print(f"    Resolution: {frame.shape[1]}x{frame.shape[0]}")
            cap.release()
            return True
        else:
            print("  ⚠ No camera found on device 0")
            return True  # Not critical for file-based testing
    except Exception as e:
        print(f"  ⚠ Camera test failed: {e}")
        return True


def test_project_structure():
    """Test if project structure is correct"""
    print("\nTesting project structure...")
    
    root = Path(__file__).parent.parent
    required_paths = [
        "src/input",
        "src/inference",
        "src/metrics",
        "src/benchmark",
        "configs",
        "scripts",
        "data/test_videos",
        "data/annotations",
        "results/benchmarks",
    ]
    
    all_exist = True
    for path_str in required_paths:
        path = root / path_str
        if path.exists():
            print(f"  ✓ {path_str}")
        else:
            print(f"  ✗ {path_str} (missing)")
            all_exist = False
    
    return all_exist


def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    try:
        from src.benchmark import BenchmarkConfig
        
        config_path = Path(__file__).parent.parent / "configs" / "benchmark_config.yaml"
        config = BenchmarkConfig(str(config_path))
        
        if config.validate():
            print("  ✓ Configuration valid")
            return True
        else:
            print("  ✗ Configuration invalid")
            return False
    except Exception as e:
        print(f"  ✗ Configuration test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("HatH Pipeline Installation Test")
    print("="*70)
    
    tests = [
        ("Package imports", test_imports),
        ("CUDA support", test_cuda),
        ("NVIDIA GPU", test_nvidia_gpu),
        ("Inference backends", test_backends),
        ("Camera access", test_camera),
        ("Project structure", test_project_structure),
        ("Configuration", test_config),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results[name] = False
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    
    print("="*70)
    if all_passed:
        print("✓ All tests passed! Installation successful.")
        print("\nYou can now run benchmarks with:")
        print("  python scripts/run_benchmark.py --help")
        return 0
    else:
        print("⚠ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

