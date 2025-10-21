#!/bin/bash
# Quick project structure verification

echo "========================================================================"
echo "HatH Pipeline - Project Structure Check"
echo "========================================================================"
echo ""

PROJECT_ROOT="/home/reza/Documents/HatHpipeline"
cd "$PROJECT_ROOT"

# Count files
echo "📁 File Count:"
echo "   Python source files:     $(find src -name "*.py" | wc -l)"
echo "   Script files:            $(find scripts -name "*.py" -o -name "*.sh" | wc -l)"
echo "   Configuration files:     $(find configs -name "*.yaml" | wc -l)"
echo "   Documentation files:     $(find . -maxdepth 1 -name "*.md" | wc -l)"
echo ""

# Check key files
echo "📄 Key Files:"
files=(
    "requirements.txt"
    "setup.py"
    "configs/benchmark_config.yaml"
    "README.md"
    "GETTING_STARTED.md"
    "TECHNICAL_DETAILS.md"
    "PROJECT_SUMMARY.md"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ $file (missing)"
    fi
done
echo ""

# Check directories
echo "📂 Directories:"
dirs=(
    "src/input"
    "src/inference"
    "src/metrics"
    "src/benchmark"
    "scripts"
    "configs"
    "examples"
    "data/test_videos"
    "data/annotations"
    "results/benchmarks"
)

for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "   ✓ $dir"
    else
        echo "   ✗ $dir (missing)"
    fi
done
echo ""

# Check module structure
echo "🔍 Module Structure:"
modules=(
    "src/__init__.py"
    "src/input/__init__.py"
    "src/inference/__init__.py"
    "src/metrics/__init__.py"
    "src/benchmark/__init__.py"
)

for module in "${modules[@]}"; do
    if [ -f "$module" ]; then
        echo "   ✓ $module"
    else
        echo "   ✗ $module (missing)"
    fi
done
echo ""

# Count lines of code
echo "📊 Code Statistics:"
total_lines=$(find src scripts -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
echo "   Total lines of code:     $total_lines"
echo ""

# Check scripts are executable
echo "🔧 Script Permissions:"
for script in scripts/*.py scripts/*.sh; do
    if [ -x "$script" ]; then
        echo "   ✓ $script (executable)"
    else
        echo "   ⚠ $script (not executable - run: chmod +x $script)"
    fi
done
echo ""

echo "========================================================================"
echo "✓ Project structure verification complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Run: bash scripts/quick_start.sh"
echo "  2. Or manually: python scripts/test_installation.py"
echo "  3. Then: python scripts/run_benchmark.py --help"
echo ""

