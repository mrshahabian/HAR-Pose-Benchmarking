# Technical Details - HatH Pipeline

## Architecture Overview

### Component Hierarchy

```
BenchmarkRunner
    ├── BenchmarkConfig (Configuration Management)
    ├── VideoSource (Input Layer)
    │   ├── USBSource
    │   ├── RTSPSource
    │   └── FileSource
    ├── YOLOEngine (Inference Layer)
    │   └── BackendFactory (Backend Management)
    └── Metrics (Monitoring Layer)
        ├── PerformanceTracker
        ├── SystemMonitor
        └── AccuracyEvaluator
```

## Video Input Pipeline

### Abstract VideoSource Interface

All video sources implement:
- `open()`: Initialize and open the source
- `get_frame()`: Return (success, frame, timestamp)
- `release()`: Cleanup resources
- `get_fps()`: Return source FPS
- `get_resolution()`: Return (width, height)

### Frame Timing

```
Timestamp Capture → Frame Acquisition → Preprocessing → Inference → Post-processing
    │                      │                  │             │             │
    └──────────────────────┴──────────────────┴─────────────┴─────────────┘
                        End-to-End Latency
```

## Inference Backends

### PyTorch (Default)
- Format: `.pt` (native PyTorch)
- Pros: Universal compatibility, easy debugging
- Cons: Slower than optimized backends
- Best for: Development, CPU-only systems

### TensorRT
- Format: `.engine`
- Pros: Best performance on NVIDIA GPUs (2-5x faster)
- Cons: Platform-specific, requires NVIDIA GPU
- Best for: Production deployment on Jetson/NVIDIA GPUs

### OpenVINO
- Format: `.xml` + `.bin`
- Pros: Optimized for Intel CPUs (1.5-3x faster than PyTorch)
- Cons: Best on Intel CPUs, variable GPU support
- Best for: CPU-only inference, Intel hardware

### ONNX Runtime
- Format: `.onnx`
- Pros: Cross-platform, good CPU/GPU support
- Cons: Moderate optimization
- Best for: Cross-platform deployment

## Performance Metrics

### FPS Calculation

```python
fps = 1.0 / avg_frame_time
```

Where `frame_time` is the interval between consecutive frames.

### Latency Components

1. **Capture Time**: Time to acquire frame from source
2. **Inference Time**: Time for model forward pass
3. **End-to-End Latency**: Total time from capture to results

```
latency_total = t_capture + t_preprocess + t_inference + t_postprocess
```

### Dropped Frame Detection

A frame is considered "dropped" if:
- Camera read fails
- Processing can't keep up with input FPS
- Buffer overflow

```python
drop_rate = dropped_frames / total_frames
```

## System Monitoring

### CPU/RAM Monitoring (psutil)

```python
cpu_percent = psutil.cpu_percent(interval=0.1)
ram_mb = psutil.virtual_memory().used / (1024**2)
```

### GPU Monitoring (NVML)

```python
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
util = pynvml.nvmlDeviceGetUtilizationRates(handle)
gpu_percent = util.gpu
mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
vram_mb = mem_info.used / (1024**2)
```

### Power Monitoring

**Desktop GPU:**
```python
power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
power_watts = power_mw / 1000.0
```

**Jetson (tegrastats):**
```bash
tegrastats --interval 100
# Parse output: VDD_IN 5123mW
```

## Accuracy Metrics

### Object Keypoint Similarity (OKS)

COCO keypoint metric:

```
OKS = Σ exp(-d_i² / (2 * s² * κ_i²)) / Σ visible_i
```

Where:
- `d_i`: Euclidean distance for keypoint i
- `s`: √(object_area)
- `κ_i`: Per-keypoint constant (varies by body part)

### Percentage of Correct Keypoints (PCK)

```
PCK@α = (# keypoints with normalized_distance < α) / total_keypoints
```

Normalized by bounding box diagonal.

## Optimization Tips

### For Maximum FPS

1. **Use TensorRT** on NVIDIA GPUs (Jetson, desktop)
2. **Lower resolution** (640×640 vs 960×960)
3. **Use nano model** (yolov11n-pose)
4. **Reduce confidence threshold** (fewer detections)
5. **Optimize video source** (reduce camera resolution at source)

### For Best Accuracy

1. **Use medium/large model** (yolov11m-pose, yolov11l-pose)
2. **Higher resolution** (960×960 or 1280×1280)
3. **Lower confidence threshold** (detect more poses)
4. **Multiple backends** for ensemble

### For Power Efficiency

1. **Jetson power modes**:
   - MAXN: Maximum performance
   - 15W/30W: Balanced (better FPS/W)
2. **Model selection**: Nano model has best FPS/W
3. **Dynamic resolution**: Reduce during idle periods

## Benchmarking Best Practices

### Warmup Period

Skip first 10-20 frames to allow:
- Model initialization (JIT compilation)
- Cache warmup
- Thermal stabilization

### Duration

Recommended test durations:
- Quick test: 30 seconds
- Standard: 60 seconds
- Thorough: 120-300 seconds

Longer tests better capture:
- Thermal throttling
- Memory leaks
- Performance stability

### Multiple Runs

Run each configuration 3-5 times:
- Calculate mean and std deviation
- Detect outliers
- Ensure reproducibility

## Hardware-Specific Notes

### NVIDIA Jetson

**Power Modes:**
```bash
# List available modes
sudo nvpmodel -q

# Set to MAXN (maximum performance)
sudo nvpmodel -m 0

# Monitor in real-time
tegrastats
```

**Jetson Orin vs Xavier:**
- Orin: Better TensorRT INT8 support
- Xavier: Lower power consumption
- Both: Use TensorRT for best performance

### Desktop NVIDIA GPU

**Check GPU Usage:**
```bash
nvidia-smi -l 1  # Update every second
```

**Enable persistence mode:**
```bash
sudo nvidia-smi -pm 1
```

### Intel CPU (OpenVINO)

**Thread Configuration:**
```python
# Set in config
device_settings:
  cpu:
    num_threads: 4  # Physical cores recommended
```

**Best Performance:**
- Use OpenVINO backend
- INT8 quantization (if supported)
- Enable CPU pinning

## Troubleshooting Performance Issues

### Low FPS

**Diagnose:**
1. Check GPU usage: `nvidia-smi`
   - Low GPU usage → CPU bottleneck
   - High GPU usage → GPU bottleneck
2. Check latency breakdown:
   - High capture_time → I/O bottleneck
   - High inference_time → Model too heavy

**Solutions:**
- CPU bottleneck: Reduce preprocessing, use threading
- GPU bottleneck: Lower resolution, smaller model
- I/O bottleneck: Use faster source, reduce resolution at source

### High Latency

**Check:**
1. Is latency consistent or spiky?
   - Consistent: Model too slow
   - Spiky: Thermal throttling or resource contention

**Solutions:**
- Use TensorRT/OpenVINO
- Batch size = 1 for real-time
- Reduce model size

### Dropped Frames

**Common causes:**
1. Processing slower than capture rate
2. Network latency (RTSP)
3. USB bandwidth limits

**Solutions:**
- Reduce input FPS
- Use frame skipping
- Buffer management

## Model Selection Guide

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|-----------|-------|----------|----------|
| YOLOv11n-pose | ~3M | Fastest | Good | Real-time, edge devices |
| YOLOv11s-pose | ~9M | Fast | Better | Balanced |
| YOLOv11m-pose | ~20M | Moderate | Best | Accuracy-critical |
| YOLOv11l-pose | ~40M | Slow | Excellent | Research, offline |
| YOLOv11x-pose | ~60M | Slowest | Best | Maximum accuracy |

## Resolution Trade-offs

| Resolution | FPS Impact | Accuracy | Memory | Use Case |
|-----------|-----------|----------|--------|----------|
| 640×640 | Baseline | Good | Low | Real-time monitoring |
| 960×960 | ~40% slower | Better | Medium | Balanced |
| 1280×1280 | ~60% slower | Best | High | Accuracy-critical |

## Future Enhancements

### Planned Features
- [ ] Multi-person tracking (ID assignment)
- [ ] Temporal smoothing (Kalman filter)
- [ ] Action recognition
- [ ] Fall detection
- [ ] Privacy masking
- [ ] Distributed inference (multiple cameras)
- [ ] Real-time dashboard
- [ ] Alert system

### Backend Extensions
- [ ] Apple Neural Engine (CoreML)
- [ ] Qualcomm Hexagon DSP
- [ ] WebAssembly (browser deployment)
- [ ] ROCm (AMD GPU)

## References

- YOLO: [Ultralytics Documentation](https://docs.ultralytics.com/)
- COCO Keypoints: [COCO Dataset](https://cocodataset.org/#keypoints-2020)
- TensorRT: [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- OpenVINO: [Intel OpenVINO](https://docs.openvino.ai/)
- NVML: [NVIDIA Management Library](https://developer.nvidia.com/nvidia-management-library-nvml)

