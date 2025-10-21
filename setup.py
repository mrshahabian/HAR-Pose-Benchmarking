"""Setup script for HatH Pipeline"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="hath-pipeline",
    version="0.1.0",
    description="Hospital at Home YOLO Pose Estimation Benchmarking Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/HatHpipeline",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "psutil>=5.9.0",
        "pynvml>=11.5.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
        "tqdm>=4.65.0",
        "pillow>=10.0.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "gpu": ["onnxruntime-gpu>=1.15.0"],
        "dev": ["pytest", "black", "flake8", "mypy"],
    },
    entry_points={
        "console_scripts": [
            "hath-benchmark=scripts.run_benchmark:main",
            "hath-download-models=scripts.download_models:main",
            "hath-visualize=scripts.visualize_results:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

