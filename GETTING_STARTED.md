# Getting Started with HatH Pipeline

## 🤖 Special Instructions for Jetson Devices

If you're using an **NVIDIA Jetson device** (Nano, Xavier, Orin, etc.), please follow these special instructions:

### Jetson Quick Setup

```bash
cd /home/reza/Documents/HAR-Pose-Benchmarking

# Use Jetson-specific setup script
bash scripts/setup_jetson.sh
```

The Jetson setup script will:
- Create a virtual environment with `--system-site-packages` flag
- Use the pre-installed PyTorch with CUDA support
- Use the Jetson-optimized OpenCV
- Install only the additional packages needed (ultralytics, pynvml, etc.)

### Why Jetson is Different

Jetson devices come with **pre-optimized PyTorch and CUDA** that's specially built for ARM64 architecture. A standard virtual environment would isolate these packages and break CUDA support. The Jetson setup uses `--system-site-packages` to inherit these optimized packages while keeping project-specific dependencies isolated.

### Verify CUDA is Working

After setup, verify CUDA is accessible in your virtual environment:

```bash
source venv/bin/activate
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0)}')"
```

You should see:
```
CUDA available: True
CUDA device: Orin (or your device name)
```

### Troubleshooting Jetson Setup

**Problem:** CUDA not available in venv  
**Solution:** Make sure you created the venv with `--system-site-packages` flag

**Problem:** `onnxruntime-gpu` installation fails  
**Solution:** This is expected - GPU version is not available for ARM64. The CPU version is used instead.

**Problem:** Package conflicts  
**Solution:** Use `requirements-jetson.txt` instead of `requirements.txt`

---

## 🚀 Quick Start Guide (5 minutes)

### Step 1: Clone and Setup

```bash
cd /home/reza/Documents/HatHpipeline

# Quick setup (automated)
bash scripts/quick_start.sh
```

**Note:** The script auto-detects Jetson devices and uses appropriate setup.

### Step 2: Test Installation

```bash
# Activate virtual environment
source venv/bin/activate

# Run tests
python scripts/test_installation.py
```

### Step 3: Download Models

```bash
# Download YOLOv11-pose models
python scripts/download_models.py --models yolo11n-pose
```

### Step 4: Run Your First Benchmark

```bash
# Quick 30-second test with USB camera
python scripts/run_benchmark.py \
  --models yolo11n-pose \
  --backends pytorch \
  --sources usb \
  --device 0 \
  --duration 30
```

**Don't have a camera?** Use a video file:

```bash
python scripts/run_benchmark.py \
  --models yolo11n-pose \
  --backends pytorch \
  --sources file \
  --file path/to/your/video.mp4 \
  --duration 30
```

### Step 5: View Results

Results are automatically saved to `results/benchmarks/`:
- JSON file with complete data
- CSV file for spreadsheet analysis
- Sample visualizations

```bash
# Generate charts
python scripts/visualize_results.py \
  --input results/benchmarks/benchmark_results_*.json
```

---

## 📋 Common Use Cases

### Use Case 1: Compare Models on Your Device

```bash
python scripts/run_benchmark.py \
  --models yolo11n-pose yolo11s-pose yolo11m-pose \
  --backends pytorch \
  --duration 60
```

**Result:** Find which model gives best FPS/accuracy trade-off

### Use Case 2: Test Different Backends

```bash
# Compare PyTorch vs TensorRT (NVIDIA GPU required)
python scripts/run_benchmark.py \
  --models yolo11n-pose \
  --backends pytorch tensorrt \
  --duration 60
```

**Result:** Measure speedup from backend optimization

### Use Case 3: Resolution Impact

```bash
python scripts/run_benchmark.py \
  --models yolo11n-pose \
  --resolutions 640x640 960x960 \
  --backends pytorch \
  --duration 60
```

**Result:** Understand FPS vs accuracy trade-off with resolution

### Use Case 4: Full Benchmark Suite

```bash
# Edit config first
nano configs/benchmark_config.yaml

# Run all combinations
python scripts/run_benchmark.py --config configs/benchmark_config.yaml
```

**Result:** Complete performance profile of your system

---

## 🎯 Platform-Specific Instructions

### Laptop (CPU Only)

```bash
# Install with OpenVINO for better CPU performance
pip install openvino

# Run benchmark
python scripts/run_benchmark.py \
  --models yolov11n-pose \
  --backends pytorch openvino \
  --duration 60
```

**Expected FPS:** 5-15 FPS (depends on CPU)

### Desktop with NVIDIA GPU

```bash
# Export model to TensorRT
python scripts/download_models.py \
  --models yolov11n-pose \
  --export tensorrt \
  --resolution 640x640

# Run benchmark
python scripts/run_benchmark.py \
  --models yolov11n-pose \
  --backends pytorch tensorrt \
  --duration 60
```

**Expected FPS:** 30-100+ FPS (depends on GPU)

### NVIDIA Jetson

```bash
# Set power mode
sudo nvpmodel -m 0  # MAXN mode

# Export to TensorRT
python scripts/download_models.py \
  --models yolov11n-pose \
  --export tensorrt

# Run benchmark
python scripts/run_benchmark.py \
  --models yolov11n-pose \
  --backends tensorrt \
  --duration 120
```

**Expected FPS:** 10-30 FPS (depends on Jetson model)

---

## 🔍 Interpreting Results

### Performance Metrics

```json
{
  "fps": 28.5,              // ⬆️ Higher is better
  "latency_ms": 35.1,       // ⬇️ Lower is better
  "latency_p95_ms": 42.3,   // 95% of frames < this
  "dropped_frames": 12      // ⬇️ Lower is better
}
```

**What to look for:**
- **FPS > 25**: Smooth real-time performance
- **Latency < 50ms**: Good responsiveness
- **Drop rate < 5%**: Reliable processing

### System Metrics

```json
{
  "cpu_percent_avg": 45.2,  // CPU usage
  "gpu_percent_avg": 78.3,  // GPU usage
  "vram_mb_avg": 1024,      // Video RAM
  "power_watts_avg": 15.2   // Power draw
}
```

**What to look for:**
- **GPU usage 70-90%**: Well optimized
- **CPU usage < 60%**: Room for parallel tasks
- **Power < budget**: Within thermal/power constraints

### Accuracy Metrics (if enabled)

```json
{
  "map_keypoint": 0.68,     // ⬆️ Higher is better
  "oks": 0.72,              // ⬆️ Higher is better
  "pck_0.5": 0.85           // ⬆️ Higher is better
}
```

**What to look for:**
- **OKS > 0.7**: Good pose accuracy
- **PCK@0.5 > 0.8**: Reliable keypoint detection

---

## 📊 Decision Matrix

### Choose Your Model

| Goal | Model | Resolution | Backend |
|------|-------|-----------|---------|
| **Max FPS** | yolov11n-pose | 640×640 | TensorRT |
| **Best Accuracy** | yolov11m-pose | 960×960 | PyTorch |
| **Balanced** | yolov11s-pose | 640×640 | TensorRT |
| **Low Power** | yolov11n-pose | 640×640 | TensorRT |
| **CPU Only** | yolov11n-pose | 640×640 | OpenVINO |

### Hardware Recommendations

**For 30+ FPS Real-time:**
- Laptop: High-end Intel/AMD with OpenVINO
- Desktop: NVIDIA RTX 2060 or better with TensorRT
- Jetson: Xavier NX or Orin Nano with TensorRT

**For 15-30 FPS:**
- Laptop: Mid-range with PyTorch
- Desktop: GTX 1660 or better with TensorRT
- Jetson: Nano with TensorRT

**For Offline Processing:**
- Any system with PyTorch

---

## 🐛 Troubleshooting

### Problem: Camera not found

```bash
# Check available cameras
ls -l /dev/video*

# Test camera
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# Grant permissions
sudo chmod 666 /dev/video0
```

### Problem: CUDA not available

```bash
# Check CUDA
nvidia-smi
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Problem: Low FPS

**Diagnose:**
1. Check GPU usage: `nvidia-smi -l 1`
2. Try smaller model: `--models yolov11n-pose`
3. Lower resolution: `--resolutions 640x640`
4. Use TensorRT: `--backends tensorrt`

### Problem: Out of memory

**Solutions:**
1. Use smaller model (nano instead of medium)
2. Lower resolution
3. Close other GPU applications
4. Check VRAM: `nvidia-smi`

---

## 📚 Next Steps

### 1. Customize Configuration

Edit `configs/benchmark_config.yaml`:
- Add your camera sources
- Choose models to test
- Set benchmark duration
- Configure output options

### 2. Run Full Benchmark

```bash
python scripts/run_benchmark.py --config configs/benchmark_config.yaml
```

### 3. Analyze Results

```bash
# Generate charts
python scripts/visualize_results.py --input results/benchmarks/*.json

# Open CSV in spreadsheet
libreoffice results/benchmarks/*.csv
```

### 4. Deploy Best Configuration

Use the best-performing configuration in your application:

```python
from src.inference import YOLOEngine
from src.input import USBSource

# Use your best config
yolo = YOLOEngine(
    model_name='yolov11n-pose',  # From benchmark
    backend='tensorrt',          # From benchmark
    imgsz=(640, 640)            # From benchmark
)

camera = USBSource(device_id=0)
camera.open()

# Your application code here
```

### 5. Extend for Your Use Case

See `examples/simple_inference.py` for integration examples.

---

## 💡 Tips & Best Practices

### For Benchmarking

1. **Close other applications** to avoid interference
2. **Run multiple times** (3-5) for consistency
3. **Let system warm up** (first run may be slower)
4. **Test realistic scenarios** (actual camera, real videos)
5. **Document your setup** (GPU model, drivers, power settings)

### For Production

1. **Use TensorRT** on NVIDIA hardware (2-5x faster)
2. **Enable persistence mode** on desktop GPUs
3. **Set appropriate power mode** on Jetson
4. **Monitor thermal throttling** in extended use
5. **Implement fallbacks** for dropped frames

### For Development

1. **Start with PyTorch** (easier debugging)
2. **Use nano model** (faster iteration)
3. **Test with files** first (reproducible)
4. **Enable visualizations** to verify results
5. **Check accuracy metrics** with ground truth

---

## 📖 Additional Resources

- **Full Documentation:** `README.md`
- **Technical Details:** `TECHNICAL_DETAILS.md`
- **Project Summary:** `PROJECT_SUMMARY.md`
- **Example Code:** `examples/simple_inference.py`

## 🆘 Getting Help

If you encounter issues:

1. Run diagnostics: `python scripts/test_installation.py`
2. Check logs in console output
3. Review error messages carefully
4. Consult `TECHNICAL_DETAILS.md`
5. Open an issue with:
   - Your hardware specs
   - Command you ran
   - Error message
   - Output of test_installation.py

---

## ✅ Checklist

Before running production benchmarks:

- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Models downloaded (`download_models.py`)
- [ ] Installation tested (`test_installation.py`)
- [ ] Configuration customized (`benchmark_config.yaml`)
- [ ] Camera/video source tested
- [ ] Output directory writable
- [ ] Sufficient disk space for results
- [ ] GPU drivers up to date (if using GPU)
- [ ] No other heavy applications running

---

**Ready to benchmark?** Run: `python scripts/run_benchmark.py --help`

**Questions?** See `README.md` or `TECHNICAL_DETAILS.md`

**Good luck with your benchmarking! 🚀**

