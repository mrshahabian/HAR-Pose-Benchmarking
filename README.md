# Hospital at Home (HatH) Pipeline - YOLO Pose Estimation Benchmark

A comprehensive benchmarking pipeline for evaluating YOLO pose estimation models across different hardware platforms, input sources, and inference backends. Designed for the Hospital at Home project to help select the optimal hardware and model configuration for real-time patient monitoring.

## Features

- **Multi-Model Support**: Benchmark YOLOv8-pose variants (nano, small, medium, large, xlarge)
- **Multiple Input Sources**: USB cameras, RTSP streams, video files
- **Backend Optimization**: PyTorch, TensorRT (NVIDIA), OpenVINO (Intel CPU)
- **Resolution Testing**: Compare performance at 640×640 and 960×960
- **Comprehensive Metrics**:
  - Performance: FPS, latency (p50/p95/p99), dropped frames
  - System: CPU/GPU utilization, VRAM, RAM, power consumption
  - Accuracy: mAP-keypoint, PCK, OKS (optional with ground truth)
- **Visualization**: Automatic generation of comparison charts
- **Cross-Platform**: Laptop, Desktop PC, NVIDIA Jetson

## Project Structure

```
HatHpipeline/
├── src/
│   ├── input/              # Video source implementations
│   ├── inference/          # YOLO inference engine
│   ├── metrics/            # Performance, accuracy, system monitoring
│   └── benchmark/          # Benchmark orchestration
├── scripts/
│   ├── run_benchmark.py    # Main CLI entry point
│   ├── download_models.py  # Download and export models
│   └── visualize_results.py # Generate comparison charts
├── configs/
│   └── benchmark_config.yaml # Configuration file
├── data/
│   ├── test_videos/        # Sample videos
│   └── annotations/        # Ground truth (optional)
├── results/
│   └── benchmarks/         # Output results (CSV, JSON, visualizations)
└── requirements.txt
```

## Installation

### Quick Start

```bash
cd /home/reza/Documents/HAR-Pose-Benchmarking
bash scripts/quick_start.sh
```

The quick start script automatically detects your platform and sets up the environment appropriately.

### 🤖 Jetson Devices (Nano, Xavier, Orin)

**Jetson devices require special setup** to use the pre-installed PyTorch with CUDA support:

```bash
# Use Jetson-specific setup script
bash scripts/setup_jetson.sh
```

This creates a virtual environment with `--system-site-packages` to access Jetson-optimized PyTorch/CUDA while keeping project dependencies isolated.

**See [GETTING_STARTED.md](GETTING_STARTED.md)** for detailed Jetson setup instructions.

### Manual Installation (Non-Jetson)

#### 1. Clone Repository

```bash
cd /home/reza/Documents/HatHpipeline
```

#### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows
```

#### 3. Install Dependencies

```bash
# Basic installation (PyTorch backend)
pip install -r requirements.txt

# For TensorRT (NVIDIA Jetson/GPU)
pip install nvidia-tensorrt

# For OpenVINO (Intel CPU optimization)
pip install openvino
```

### 4. Download Models

```bash
# Download all YOLOv11-pose models
python scripts/download_models.py

# Download and export for specific backend
python scripts/download_models.py --export tensorrt --resolution 640x640
```

## Quick Start

### 1. Configure Benchmark

Edit `configs/benchmark_config.yaml`:

```yaml
models:
  - yolov11n-pose
  - yolov11s-pose
  - yolov11m-pose

resolutions:
  - [640, 640]
  - [960, 960]

backends:
  - pytorch
  - tensorrt  # If available

sources:
  - type: usb
    device: 0
    name: "USB_Webcam"

duration_seconds: 60
```

### 2. Run Benchmark

```bash
# Run all benchmarks from config
python scripts/run_benchmark.py --config configs/benchmark_config.yaml

# Quick test with USB camera only
python scripts/run_benchmark.py --sources usb --device 0 --duration 30

# Test specific models and backends
python scripts/run_benchmark.py --models yolov11n-pose yolov11s-pose --backends pytorch

# Test with video file
python scripts/run_benchmark.py --sources file --file data/test_videos/sample.mp4
```

### 3. Visualize Results

```bash
# Generate comparison charts
python scripts/visualize_results.py --input results/benchmarks/benchmark_results_*.json

# Export as PDF
python scripts/visualize_results.py --input results/benchmarks/results.json --format pdf
```

## Usage Examples

### Benchmark on Different Devices

**Laptop (CPU only):**
```bash
python scripts/run_benchmark.py \
  --models yolov11n-pose \
  --backends pytorch openvino \
  --resolutions 640x640 \
  --duration 60
```

**Desktop with NVIDIA GPU:**
```bash
python scripts/run_benchmark.py \
  --models yolov11n-pose yolov11s-pose yolov11m-pose \
  --backends pytorch tensorrt \
  --resolutions 640x640 960x960 \
  --duration 120
```

**NVIDIA Jetson:**
```bash
# Set power mode first
sudo nvpmodel -m 0  # MAXN mode

python scripts/run_benchmark.py \
  --config configs/benchmark_config.yaml
```

### Test Multiple Input Sources

```bash
# USB camera
python scripts/run_benchmark.py --sources usb --device 0

# RTSP stream (IP camera)
python scripts/run_benchmark.py --sources rtsp --rtsp-url rtsp://192.168.1.100:8554/stream

# Video file
python scripts/run_benchmark.py --sources file --file data/test_videos/sample.mp4
```

## Configuration

### Benchmark Configuration (`configs/benchmark_config.yaml`)

Key settings:

- **models**: List of YOLO models to test
- **resolutions**: Input resolutions to benchmark
- **backends**: Inference backends (pytorch, tensorrt, openvino)
- **sources**: Video input configurations
- **duration_seconds**: Duration of each benchmark run
- **warmup_frames**: Frames to skip before metrics collection
- **confidence_threshold**: Minimum detection confidence
- **save_visualizations**: Save sample frames with pose overlay

### Video Sources

**USB Camera:**
```yaml
- type: usb
  device: 0  # Camera device ID
  name: "USB_Webcam"
```

**RTSP Stream:**
```yaml
- type: rtsp
  url: "rtsp://192.168.1.100:8554/stream"
  name: "IP_Camera"
```

**Video File:**
```yaml
- type: file
  path: "data/test_videos/sample.mp4"
  loop: true
  name: "Test_Video"
```

## Metrics Explained

### Performance Metrics

- **FPS**: Frames processed per second (higher is better)
- **Latency**: End-to-end time from capture to result (lower is better)
  - p50: Median latency
  - p95: 95th percentile (most frames complete within this time)
  - p99: 99th percentile
- **Dropped Frames**: Frames not processed in real-time

### System Metrics

- **CPU Usage**: Processor utilization (%)
- **GPU Usage**: GPU utilization (%) [NVIDIA only]
- **RAM/VRAM**: Memory usage (MB)
- **Power Draw**: Power consumption (Watts) [Jetson/NVIDIA]

### Accuracy Metrics (Optional)

Requires ground truth annotations in COCO format:

- **mAP-keypoint**: Mean average precision for keypoints
- **OKS**: Object Keypoint Similarity (COCO metric)
- **PCK**: Percentage of Correct Keypoints

## Output Files

Results are saved in `results/benchmarks/`:

- `benchmark_results_YYYYMMDD_HHMMSS.json`: Complete results
- `benchmark_results_YYYYMMDD_HHMMSS.csv`: Flattened CSV for analysis
- `visualizations/`: Sample frames with pose overlay
- Comparison charts (FPS, latency, resource usage, etc.)

## Hardware Setup

### NVIDIA Jetson

1. **Set Power Mode:**
   ```bash
   sudo nvpmodel -m 0  # MAXN
   # or
   sudo nvpmodel -m 1  # 15W mode
   ```

2. **Monitor Resources:**
   ```bash
   tegrastats  # Real-time stats
   ```

3. **Install TensorRT:**
   ```bash
   # Usually pre-installed on JetPack
   pip install pycuda
   ```

### Desktop with NVIDIA GPU

1. **Install NVIDIA Drivers:**
   ```bash
   nvidia-smi  # Verify installation
   ```

2. **Install CUDA Toolkit** (if not present)

3. **Install TensorRT:**
   ```bash
   pip install nvidia-tensorrt
   ```

### CPU-Only System

1. **Install OpenVINO** (optional, for optimization):
   ```bash
   pip install openvino
   ```

2. **Use PyTorch backend** (always available)

## Troubleshooting

### Camera Access Issues

```bash
# Check camera devices
ls -l /dev/video*

# Grant permissions
sudo chmod 666 /dev/video0

# Test with OpenCV
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### CUDA/GPU Not Detected

```bash
# Check CUDA
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"

# Check PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"
```

### TensorRT Export Fails

- Ensure NVIDIA GPU is available
- Check TensorRT compatibility with your PyTorch version
- Try exporting to ONNX first: `--export onnx`

### Low FPS on Jetson

- Set power mode to MAXN: `sudo nvpmodel -m 0`
- Use TensorRT backend for best performance
- Start with nano model (yolov11n-pose)
- Reduce resolution to 640×640

### Memory Issues

- Reduce batch size (models use batch=1 by default)
- Lower resolution
- Close other applications
- Use lighter model (nano instead of medium)

## Development Roadmap

This benchmarking pipeline is the foundation for the full Hospital at Home platform:

- [ ] Multi-sensor data collection (sensors, cameras)
- [ ] Multiple communication protocols (TCP/UDP, Serial, USB, Bluetooth)
- [ ] API integration (REST, GraphQL)
- [ ] Database connectivity
- [ ] Message queuing (ZMQ, MQTT)
- [ ] Real-time monitoring dashboard
- [ ] Alert system
- [ ] Role-based access control
- [ ] Privacy-preserving AI
- [ ] Edge deployment

## Contributing

This is an active research project. Contributions welcome!

## License

[Specify your license]

## Citation

If you use this benchmark in your research, please cite:

```
@software{hath_benchmark_2025,
  title={Hospital at Home YOLO Pose Estimation Benchmark},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/HatHpipeline}
}
```

## Support

For questions or issues:
- Open a GitHub issue
- Contact: [your-email@example.com]

## Acknowledgments

- YOLO models: [Ultralytics](https://github.com/ultralytics/ultralytics)
- COCO keypoint format: [COCO Dataset](https://cocodataset.org/)
- Hospital at Home initiative

---

**Note**: This pipeline is designed for research and development. Ensure compliance with privacy regulations (HIPAA, GDPR) before deploying in production healthcare environments.

