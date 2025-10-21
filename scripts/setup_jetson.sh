#!/bin/bash
# Setup script for Jetson devices
# This script creates a virtual environment with system-site-packages
# to access Jetson-optimized PyTorch and CUDA

set -e

echo "======================================================================"
echo "HatH Pipeline - Jetson Setup"
echo "======================================================================"
echo ""

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Verify we're on a Jetson device
if [ ! -f "/etc/nv_tegra_release" ] && ! uname -a | grep -q "tegra"; then
    echo "⚠ Warning: This doesn't appear to be a Jetson device."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Remove existing venv if present
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
    echo "✓ Removed old venv"
    echo ""
fi

# Create virtual environment with system site packages
echo "Creating virtual environment with system-site-packages..."
python3 -m venv --system-site-packages venv
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip upgraded"
echo ""

# Install Jetson-specific requirements
echo "Installing dependencies from requirements-jetson.txt..."
echo "This will install only packages not already in the system..."
echo "(Note: NumPy is constrained to <2.0 for PyTorch compatibility)"
pip install -r requirements-jetson.txt
echo "✓ Dependencies installed"
echo ""

# Create necessary directories
echo "Creating project directories..."
mkdir -p results/benchmarks
mkdir -p data/test_videos
mkdir -p data/annotations
echo "✓ Directories created"
echo ""

# Test installation
echo "======================================================================"
echo "Testing Installation"
echo "======================================================================"
python scripts/test_installation.py

echo ""
echo "======================================================================"
echo "Setup Complete!"
echo "======================================================================"
echo ""
echo "Your virtual environment is configured to use:"
echo "  • System PyTorch with CUDA support"
echo "  • System OpenCV optimized for Jetson"
echo "  • Project-specific packages (ultralytics, etc.)"
echo ""
echo "To verify CUDA is working:"
echo "  source venv/bin/activate"
echo "  python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'"
echo ""
echo "Next steps:"
echo "  1. Download models:"
echo "     python scripts/download_models.py --models yolov11n-pose"
echo ""
echo "  2. Run a quick test:"
echo "     python scripts/run_benchmark.py --models yolov11n-pose --backends pytorch --duration 30"
echo ""

