# Jetson Setup Guide

## Problem Solved

On Jetson devices (Nano, Xavier, Orin), PyTorch is pre-installed with CUDA support optimized for ARM64 architecture. Standard virtual environments isolate these system packages, causing CUDA to become unavailable. This guide explains how the project now handles Jetson devices properly.

## Solution: System Site Packages

The solution uses `--system-site-packages` flag when creating the virtual environment. This allows the venv to:
- ✅ Inherit system PyTorch with CUDA support
- ✅ Inherit system OpenCV optimized for Jetson
- ✅ Keep project-specific packages isolated
- ✅ Maintain dependency management

## Quick Setup

```bash
cd /home/reza/Documents/HAR-Pose-Benchmarking

# Option 1: Use quick start (auto-detects Jetson)
bash scripts/quick_start.sh

# Option 2: Use Jetson-specific script directly
bash scripts/setup_jetson.sh
```

## What Was Changed

### 1. New Files Created

#### `requirements-jetson.txt`
- Minimal requirements file for Jetson
- Excludes: `torch`, `torchvision`, `opencv-python`, system packages
- Includes: `ultralytics`, `pynvml`, `tqdm`, `onnx`, `onnxruntime` (CPU only)
- Reason: Uses system versions to access CUDA

#### `scripts/setup_jetson.sh`
- Automated setup script for Jetson devices
- Creates venv with `--system-site-packages`
- Installs only necessary additional packages
- Creates required directories
- Runs installation test

### 2. Modified Files

#### `scripts/quick_start.sh`
- Added Jetson device detection
- Checks for `/etc/nv_tegra_release` or `tegra` in kernel
- Automatically redirects to `setup_jetson.sh` on Jetson devices
- Standard setup for other platforms

#### `GETTING_STARTED.md`
- Added prominent Jetson section at the top
- Explains why Jetson is different
- Provides verification steps
- Includes troubleshooting guide

#### `README.md`
- Updated installation section
- Added Jetson quick setup instructions
- Links to detailed documentation

## Verification Steps

After setup, verify CUDA is working:

```bash
source venv/bin/activate

# Test 1: Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test 2: Check device name
python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0)}')"

# Test 3: Check CUDA version
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Test 4: Run full installation test
python scripts/test_installation.py
```

Expected output:
```
CUDA available: True
Device: Orin
CUDA version: 12.6
```

## How It Works

### Standard venv (before fix)
```bash
python3 -m venv venv
# ❌ Completely isolated
# ❌ No access to system PyTorch
# ❌ No CUDA support
```

### Jetson venv (with fix)
```bash
python3 -m venv --system-site-packages venv
# ✅ Inherits system packages
# ✅ Access to PyTorch with CUDA
# ✅ Project packages isolated in venv
```

### Package Sources

When using `--system-site-packages`:

| Package | Source | Reason |
|---------|--------|--------|
| torch | System `/usr/local/lib/python3.10/dist-packages` | Jetson-optimized with CUDA |
| torchvision | System `/home/reza/vision` | Jetson-optimized |
| opencv | System | Jetson-optimized with GStreamer |
| numpy | System | Pre-installed |
| pandas | System | Pre-installed |
| ultralytics | venv | Needs latest version |
| pynvml | venv | Not in system |
| tqdm | venv | Not in system |

## Troubleshooting

### Issue: CUDA not available in venv

**Symptoms:**
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Solution:**
1. Delete existing venv: `rm -rf venv`
2. Run setup script: `bash scripts/setup_jetson.sh`
3. Verify flag was used: `cat venv/pyvenv.cfg | grep system-site-packages`
   - Should show: `include-system-site-packages = true`

### Issue: Package conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed...
```

**Solution:**
Use `requirements-jetson.txt` instead of `requirements.txt`:
```bash
pip install -r requirements-jetson.txt
```

### Issue: onnxruntime-gpu installation fails

**Symptoms:**
```
ERROR: No matching distribution found for onnxruntime-gpu>=1.15.0
```

**Solution:**
This is expected! ARM64 doesn't have GPU version. The CPU version is used automatically via `requirements-jetson.txt`.

## Architecture Explanation

```
┌─────────────────────────────────────────────────────┐
│ Virtual Environment (venv)                          │
│                                                     │
│  ┌────────────────────────────────────────────┐   │
│  │ Project-Specific Packages                   │   │
│  │ • ultralytics                               │   │
│  │ • pynvml                                    │   │
│  │ • tqdm                                      │   │
│  │ • onnx, onnxruntime                        │   │
│  └────────────────────────────────────────────┘   │
│                                                     │
│  ┌────────────────────────────────────────────┐   │
│  │ System Site Packages (--system-site-packages)│  │
│  │ • torch (with CUDA 12.6)                   │   │
│  │ • torchvision                              │   │
│  │ • opencv (Jetson-optimized)                │   │
│  │ • numpy, pandas, matplotlib                │   │
│  └────────────────────────────────────────────┘   │
│                      ↓                              │
│                 inherits from                       │
│                      ↓                              │
└────────────────────┼────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│ System Python Installation                          │
│ /usr/local/lib/python3.10/dist-packages            │
│ • PyTorch with CUDA support                         │
│ • Jetson-specific optimizations                     │
└─────────────────────────────────────────────────────┘
```

## Manual Setup (Advanced)

If you need to set up manually:

```bash
# 1. Remove old venv
rm -rf venv

# 2. Create venv with system site packages
python3 -m venv --system-site-packages venv

# 3. Activate
source venv/bin/activate

# 4. Install Jetson requirements
pip install -r requirements-jetson.txt

# 5. Verify
python -c "import torch; print(torch.cuda.is_available())"
```

## Next Steps

After successful setup:

1. **Download models:**
   ```bash
   python scripts/download_models.py --models yolov11n-pose
   ```

2. **Run a test benchmark:**
   ```bash
   python scripts/run_benchmark.py --models yolov11n-pose --backends pytorch --duration 30
   ```

3. **Check GPU utilization:**
   ```bash
   # In another terminal
   watch -n 1 tegrastats
   ```

## References

- [Python venv documentation](https://docs.python.org/3/library/venv.html)
- [NVIDIA Jetson PyTorch installation](https://forums.developer.nvidia.com/t/pytorch-for-jetson)
- [System site packages explanation](https://docs.python.org/3/library/site.html)

