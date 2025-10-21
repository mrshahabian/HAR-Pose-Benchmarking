# Viewing Benchmark Visualizations

## 📊 Generating Visualizations

After running benchmarks, generate visualization charts:

```bash
# Activate virtual environment
source venv/bin/activate

# Generate charts from latest benchmark results
python scripts/visualize_results.py --input results/benchmarks/benchmark_results_*.json

# Or specify a specific result file
python scripts/visualize_results.py --input results/benchmarks/benchmark_results_20251021_195724.json
```

## 📁 Output Files

Visualizations are saved to `results/visualizations/`:

```
results/visualizations/
├── fps_comparison.png          # FPS bar chart across models
├── latency_distribution.png    # Latency box plots
├── resource_usage.png          # CPU/GPU/RAM/VRAM heatmaps
├── power_efficiency.png        # FPS per Watt comparison
└── summary_report.txt          # Text summary of top performers
```

## 🖼️ Viewing the Charts

### Option 1: File Manager (Easiest)

```bash
# Open the visualizations folder
xdg-open results/visualizations/
```

Then double-click any PNG file to view it.

### Option 2: Image Viewer (Command Line)

```bash
# Ubuntu/GNOME
eog results/visualizations/fps_comparison.png

# KDE
gwenview results/visualizations/*.png

# Lightweight viewer
feh results/visualizations/*.png

# View all at once (slideshow)
eog results/visualizations/*.png
```

### Option 3: From IDE

If using VS Code or other IDE:
1. Navigate to `results/visualizations/` in the file explorer
2. Click on any `.png` file
3. The image will open in the editor

## 📄 Reading the Text Summary

```bash
# View in terminal
cat results/visualizations/summary_report.txt

# Or open in text editor
nano results/visualizations/summary_report.txt
```

## 📊 Understanding the Charts

### 1. FPS Comparison (`fps_comparison.png`)

**What it shows:** Frames per second for each model at different resolutions

**How to read:**
- Taller bars = faster (better for real-time)
- Compares models side-by-side
- Separate panels for 640×640 and 960×960

**What to look for:**
- **> 30 FPS**: Good for real-time applications
- **> 60 FPS**: Excellent, very smooth
- Nano models typically fastest

### 2. Latency Distribution (`latency_distribution.png`)

**What it shows:** Box plot of end-to-end latency

**How to read:**
- Box = 25th to 75th percentile
- Line in box = median
- Whiskers = min/max (excluding outliers)
- Lower = better (faster response)

**What to look for:**
- **< 50ms**: Good for real-time
- **< 30ms**: Excellent responsiveness
- Small box = consistent performance

### 3. Resource Usage (`resource_usage.png`)

**What it shows:** Heatmap of CPU, GPU, RAM, VRAM usage

**How to read:**
- Darker red = higher usage
- Numbers show exact percentages/MB
- Rows = models, Columns = backends

**What to look for:**
- High GPU usage = well optimized
- Low CPU + high GPU = good parallelization
- VRAM usage helps size your GPU

### 4. Power Efficiency (`power_efficiency.png`)

**What it shows:** FPS per Watt (higher = more efficient)

**How to read:**
- Taller bars = more efficient
- Important for battery-powered devices
- Compares energy efficiency

**What to look for:**
- YOLO11n typically most efficient
- Nano models best FPS/W ratio
- Critical for Jetson/embedded

## 📈 Example Results Interpretation

From your benchmark summary:

```
Highest FPS:
  yolov8n-pose: 85.53 FPS at 640×640
  
Lowest Latency:
  yolov8n-pose: 10.88 ms
  
Best Power Efficiency:
  yolo11n-pose: 3.46 FPS/W
```

**What this means:**
- ✅ Both nano models excellent for real-time (>80 FPS)
- ✅ Latency under 11ms = very responsive
- ✅ YOLO11n slightly more power-efficient
- 📊 Perfect for edge deployment!

## 🎨 Customizing Visualizations

### Change Output Format

```bash
# Generate as PDF instead of PNG
python scripts/visualize_results.py --input results/*.json --format pdf

# Generate as SVG (vector graphics)
python scripts/visualize_results.py --input results/*.json --format svg

# High resolution (for publications)
python scripts/visualize_results.py --input results/*.json --dpi 600
```

### Change Output Directory

```bash
python scripts/visualize_results.py \
  --input results/benchmarks/*.json \
  --output-dir my_custom_folder
```

## 🔍 Comparing Multiple Benchmarks

If you've run benchmarks on different devices:

```bash
# Device 1 (Laptop)
python scripts/visualize_results.py \
  --input laptop_results.json \
  --output-dir visualizations/laptop

# Device 2 (Jetson)
python scripts/visualize_results.py \
  --input jetson_results.json \
  --output-dir visualizations/jetson

# Now compare the two folders manually
```

## 💡 Tips

1. **Keep Results Organized**
   - Name result files by device: `laptop_benchmark.json`, `jetson_benchmark.json`
   - Create separate visualization folders per device

2. **Share Results**
   - Charts are PNG files - easy to share via email/docs
   - Include `summary_report.txt` for quick overview

3. **Publication-Ready**
   - Use `--format pdf --dpi 600` for high-quality charts
   - SVG format for scalable graphics

4. **Quick Comparison**
   - Open all PNGs at once: `eog results/visualizations/*.png`
   - Use arrow keys to flip between charts

## 🐛 Troubleshooting

### "No module named matplotlib"

```bash
pip install matplotlib seaborn
```

### Charts Don't Open

```bash
# Install image viewer
sudo apt install eog  # Ubuntu/GNOME
# or
sudo apt install gwenview  # KDE

# Or view in browser
firefox results/visualizations/fps_comparison.png
```

### Empty/Missing Charts

- Make sure you have benchmark results: `ls results/benchmarks/*.json`
- Check the JSON file has data: `cat results/benchmarks/*.json`
- Re-run benchmark if needed

## 📚 Related Documentation

- **Running Benchmarks:** `GETTING_STARTED.md`
- **Understanding Metrics:** `TECHNICAL_DETAILS.md`
- **Model Selection:** `MODEL_VERSIONS.md`

---

**Quick Command Reference:**

```bash
# Generate charts
python scripts/visualize_results.py --input results/benchmarks/*.json

# View charts
xdg-open results/visualizations/

# Read summary
cat results/visualizations/summary_report.txt
```

Happy benchmarking! 📊🚀

