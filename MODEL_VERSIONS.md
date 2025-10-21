# YOLO Model Versions

## Available Models

The pipeline supports **YOLO11-pose** (latest) and **YOLOv8-pose** models from Ultralytics.

### YOLO11-pose Variants (Latest - October 2024)

**Important:** YOLO11 uses `yolo11` not `yolov11` (no "v")

| Model | Size | Parameters | Speed | Accuracy | Use Case |
|-------|------|------------|-------|----------|----------|
| **yolo11n-pose** | 6.0 MB | ~3M | Fastest | Good | Real-time, edge devices |
| **yolo11s-pose** | 19.4 MB | ~10M | Fast | Better | Balanced performance |
| **yolo11m-pose** | 40.5 MB | ~25M | Moderate | Best | Accuracy-focused |
| **yolo11l-pose** | ~65 MB | ~45M | Slow | Excellent | High accuracy needs |
| **yolo11x-pose** | ~90 MB | ~70M | Slowest | Best | Maximum accuracy |

### YOLOv8-pose Variants (Previous Stable)

| Model | Size | Parameters | Speed | Accuracy | Use Case |
|-------|------|------------|-------|----------|----------|
| **yolov8n-pose** | 6.5 MB | ~3M | Fastest | Good | Real-time, edge devices |
| **yolov8s-pose** | 22.4 MB | ~11M | Fast | Better | Balanced performance |
| **yolov8m-pose** | 50.8 MB | ~26M | Moderate | Best | Accuracy-focused |
| **yolov8l-pose** | ~85 MB | ~44M | Slow | Excellent | High accuracy needs |
| **yolov8x-pose** | ~120 MB | ~69M | Slowest | Best | Maximum accuracy |

### Default Configuration

The benchmark uses YOLO11-pose models by default (latest version):
- `yolo11n-pose` - Nano variant (recommended for testing)
- `yolo11s-pose` - Small variant (balanced)
- `yolo11m-pose` - Medium variant (higher accuracy)

## Download Models

```bash
# Download YOLO11-pose models (latest - recommended)
python scripts/download_models.py --models yolo11n-pose

# Download all YOLO11 default models
python scripts/download_models.py --models yolo11n-pose yolo11s-pose yolo11m-pose

# Download YOLOv8-pose models (previous stable)
python scripts/download_models.py --models yolov8n-pose yolov8s-pose yolov8m-pose

# Download and export for TensorRT
python scripts/download_models.py --models yolo11n-pose --export tensorrt --resolution 640x640
```

**Important Naming:**
- YOLO11: Use `yolo11` (no "v") - e.g., `yolo11n-pose`
- YOLOv8: Use `yolov8` (with "v") - e.g., `yolov8n-pose`

## Model Storage

Models are automatically cached by Ultralytics in:
- `~/.ultralytics/` - Main cache directory
- Current directory - When explicitly specified

The download script will show you the exact location after download.

## Future Models

When YOLOv9-pose, YOLOv10-pose, or newer versions are released, simply update the model names in `configs/benchmark_config.yaml`:

```yaml
models:
  - yolov9n-pose  # When available
  - yolov10n-pose  # When available
```

The pipeline is designed to work with any YOLO pose model following the Ultralytics naming convention.

## Notes

### Available Models (as of October 2024)

- ✅ **YOLO11-pose**: **Available & RECOMMENDED** (Latest version)
  - Download: `python scripts/download_models.py --models yolo11n-pose`
  - Naming: `yolo11` (no "v") - e.g., `yolo11n-pose`, `yolo11s-pose`
  - Variants: nano, small, medium, large, xlarge
  - **This is the latest and best-performing version**
  
- ✅ **YOLOv8-pose**: Available (Previous stable)
  - Download: `python scripts/download_models.py --models yolov8n-pose`
  - Naming: `yolov8` (with "v") - e.g., `yolov8n-pose`, `yolov8s-pose`
  - Variants: nano, small, medium, large, xlarge
  - Use if you need maximum stability/compatibility

- ❌ **YOLOv7-pose**: **Not available**
  - YOLOv7 exists for detection but NOT for pose estimation
  
- ❌ **YOLOv9-pose**: Not available
- ❌ **YOLOv10-pose**: Not available
- ❌ **YOLOv5-pose**: Not available (despite documentation claims)

**Recommendation:** Use **YOLO11-pose** (latest) or **YOLOv8-pose** (stable).

Models are downloaded from: https://github.com/ultralytics/assets/releases/

### How to Check Available Models

Visit the Ultralytics releases page to see what's currently available:
- https://github.com/ultralytics/ultralytics/releases
- https://github.com/ultralytics/assets/releases

Or try downloading and the script will tell you if it doesn't exist.

## Customization

To use a custom model:

1. Place your `.pt` file in the project directory
2. Update `configs/benchmark_config.yaml`:
   ```yaml
   models:
     - path/to/your/custom-model.pt
   ```

3. Run the benchmark as usual

