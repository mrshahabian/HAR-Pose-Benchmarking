# 🚀 Jetson Quick Reference

## ✅ Your Setup is Complete!

CUDA is working in your virtual environment! All tests passed.

## Daily Usage

### Activate Environment
```bash
cd ~/Documents/HAR-Pose-Benchmarking
source venv/bin/activate
```

### Verify CUDA
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Run Benchmark
```bash
# Quick 30-second test
python scripts/run_benchmark.py \
  --models yolov11n-pose \
  --backends pytorch \
  --duration 30

# Full benchmark
python scripts/run_benchmark.py \
  --config configs/benchmark_config.yaml
```

### Monitor GPU
```bash
# In another terminal
watch -n 1 tegrastats
```

## If You Need to Reinstall

```bash
rm -rf venv
bash scripts/setup_jetson.sh
```

## Documentation

- **This file:** Quick reference
- **JETSON_FIX_COMPLETE.md:** Success report & details
- **JETSON_SETUP.md:** Complete technical guide
- **GETTING_STARTED.md:** Usage examples
- **README.md:** Full documentation

## Status

```
✓ CUDA: Working (12.6)
✓ Device: Orin
✓ PyTorch: 2.3.0 with CUDA
✓ Ultralytics: Installed
✓ All tests: Passing
```

**You're ready to benchmark! 🎉**
