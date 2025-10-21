#!/bin/bash
# Quick start script for HatH Pipeline

set -e

echo "======================================================================"
echo "HatH Pipeline Quick Start"
echo "======================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip > /dev/null
pip install -r requirements.txt > /dev/null 2>&1
echo "✓ Dependencies installed"

# Test installation
echo ""
echo "Testing installation..."
python scripts/test_installation.py

# Download models
echo ""
read -p "Download YOLO models? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading YOLO models..."
    python scripts/download_models.py --models yolov11n-pose
    echo "✓ Models downloaded"
fi

echo ""
echo "======================================================================"
echo "Setup Complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Edit configuration:"
echo "     nano configs/benchmark_config.yaml"
echo ""
echo "  3. Run a quick test:"
echo "     python scripts/run_benchmark.py --models yolov11n-pose --backends pytorch --duration 30"
echo ""
echo "  4. Run full benchmark:"
echo "     python scripts/run_benchmark.py --config configs/benchmark_config.yaml"
echo ""
echo "  5. Visualize results:"
echo "     python scripts/visualize_results.py --input results/benchmarks/benchmark_results_*.json"
echo ""
echo "For help:"
echo "  python scripts/run_benchmark.py --help"
echo ""

