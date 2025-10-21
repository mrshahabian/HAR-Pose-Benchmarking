# Changelog

All notable changes to the HatH Pipeline project.

## [0.1.2] - 2024-10-21

### Added
- ✅ **YOLO11-pose Support**: Discovered and added support for latest YOLO11-pose models
  - Naming: `yolo11n-pose` (no "v" - different from YOLOv8)
  - Downloaded: nano (6.0MB), small (20MB), medium (41MB)
  - Updated default configuration to use YOLO11-pose
- **Model Compatibility Testing**: Verified which YOLO pose models actually exist
  - ✅ YOLO11-pose: Available (latest)
  - ✅ YOLOv8-pose: Available (stable)
  - ❌ YOLOv7-pose: Not available
  - ❌ YOLOv9/v10/v5-pose: Not available

### Changed
- **Default Models**: Changed from YOLOv8-pose to YOLO11-pose (latest version)
- **Documentation**: Updated MODEL_VERSIONS.md with accurate availability information

## [0.1.1] - 2024-10-21

### Fixed
- **Model Download**: Models are now permanently saved to `models/` directory instead of Ultralytics cache
- **Cleanup**: Temporary `.pt` files in root directory are automatically cleaned up after download
- **Model Loading**: Pipeline now loads models from local `models/` directory when available

### Added
- `--yes` flag to `run_benchmark.py` for non-interactive mode (skips confirmation)
- Automatic cleanup of temporary model files
- `MODEL_VERSIONS.md` documentation explaining available YOLO versions
- Better error messages when downloading non-existent models

### Changed
- Default models changed from `yolov11*-pose` to `yolov8*-pose` in config
- Models now stored in `models/` directory (already in `.gitignore`)
- Download script shows final location of saved models

### Technical Details
- `backend_factory.py`: Enhanced `download_model()` to copy from Ultralytics cache to local `models/`
- `backend_factory.py`: Updated `load_model()` to prefer local models directory
- `download_models.py`: Added cleanup routine for temporary files
- `run_benchmark.py`: Added `--yes` argument and `skip_confirmation` parameter

## [0.1.0] - 2024-10-21

### Initial Release
- Complete YOLO pose estimation benchmarking pipeline
- Multi-model support (YOLOv8-pose variants)
- Multiple input sources (USB, RTSP, video files)
- Backend optimization (PyTorch, TensorRT, OpenVINO, ONNX)
- Comprehensive metrics collection (FPS, latency, resources, accuracy)
- Automated visualization generation
- Cross-platform support (laptop, desktop, Jetson)
- Complete documentation (README, guides, technical details)

---

## Model Storage

### Current Behavior (v0.1.1+)
```
models/                    # Git-ignored, stores all model files
├── yolov8n-pose.pt       # Nano model (6.5 MB)
├── yolov8s-pose.pt       # Small model (23 MB)
└── yolov8m-pose.pt       # Medium model (51 MB)
```

### Download Process
1. Script downloads model from Ultralytics servers
2. Model is temporarily saved to current directory
3. Model is copied to `models/` directory
4. Temporary file is automatically cleaned up
5. Future loads use the local `models/` directory

### Benefits
- ✅ Models persisted across runs
- ✅ No re-downloading on each use
- ✅ Clean project root (no scattered `.pt` files)
- ✅ Git-ignored (not committed to repository)
- ✅ Easy to share (just tar the models directory)

