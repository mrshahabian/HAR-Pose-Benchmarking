"""Factory for creating YOLO models with different backends"""

import os
from pathlib import Path
from typing import Optional, Tuple
from ultralytics import YOLO


class BackendFactory:
    """Factory for loading and exporting YOLO models with different backends"""
    
    SUPPORTED_BACKENDS = ['pytorch', 'tensorrt', 'openvino', 'onnx']
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize backend factory
        
        Args:
            model_dir: Directory to store downloaded and exported models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def get_model_path(self, model_name: str, backend: str = 'pytorch') -> Path:
        """
        Get the path to a model file for a specific backend
        
        Args:
            model_name: Model name (e.g., 'yolov11n-pose')
            backend: Backend type ('pytorch', 'tensorrt', 'openvino', 'onnx')
            
        Returns:
            Path to model file
        """
        if backend == 'pytorch':
            return self.model_dir / f"{model_name}.pt"
        elif backend == 'onnx':
            return self.model_dir / f"{model_name}.onnx"
        elif backend == 'tensorrt':
            return self.model_dir / f"{model_name}.engine"
        elif backend == 'openvino':
            return self.model_dir / f"{model_name}_openvino_model"
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def download_model(self, model_name: str) -> Path:
        """
        Download a YOLO model and save it to local models directory
        
        Args:
            model_name: Model name (e.g., 'yolov8n-pose')
            
        Returns:
            Path to downloaded model in local models directory
        """
        import shutil
        
        # Local path in our models directory
        local_model_path = self.model_dir / f"{model_name}.pt"
        
        # Check if already exists locally
        if local_model_path.exists():
            print(f"✓ Model already exists: {model_name}")
            print(f"  Location: {local_model_path}")
            return local_model_path
        
        print(f"Downloading model: {model_name}")
        
        try:
            # Create models directory if it doesn't exist
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            # Load model - Ultralytics will download to its cache
            model = YOLO(model_name)
            
            # Find where Ultralytics cached the model
            cached_path = None
            
            # Check model object attributes
            if hasattr(model, 'ckpt_path') and model.ckpt_path:
                cached_path = Path(model.ckpt_path)
            
            # If not found, check common Ultralytics cache locations
            if cached_path is None or not cached_path.exists():
                from ultralytics import settings
                possible_locations = [
                    Path(f"{model_name}.pt"),  # Current directory
                    Path.home() / '.config' / 'Ultralytics' / f"{model_name}.pt",
                    Path.home() / '.cache' / 'ultralytics' / f"{model_name}.pt",
                ]
                
                # Also check settings
                try:
                    weights_dir = Path(settings.get('weights_dir', Path.home() / '.config' / 'Ultralytics'))
                    possible_locations.append(weights_dir / f"{model_name}.pt")
                except:
                    pass
                
                for path in possible_locations:
                    if path.exists():
                        cached_path = path
                        break
            
            # Copy model to our local models directory
            if cached_path and cached_path.exists():
                shutil.copy2(cached_path, local_model_path)
                print(f"✓ Model downloaded and saved: {model_name}")
                print(f"  Source: {cached_path}")
                print(f"  Saved to: {local_model_path}")
                
                # Clean up if it was downloaded to current directory
                if cached_path.parent == Path.cwd() and cached_path != local_model_path:
                    try:
                        cached_path.unlink()
                        print(f"  Cleaned up temporary file")
                    except:
                        pass
                
                return local_model_path
            else:
                print(f"⚠ Model loaded but couldn't find file to copy")
                print(f"  Model will be used from Ultralytics cache")
                return Path(f"{model_name}.pt")
            
        except Exception as e:
            print(f"✗ Error downloading model {model_name}: {e}")
            print(f"  Note: Some model versions may not exist yet (e.g., yolov11)")
            print(f"  Available: YOLOv8-pose, YOLOv5-pose")
            # Return the expected path anyway - Ultralytics will handle it
            return Path(f"{model_name}.pt")
    
    def export_model(self, model_name: str, backend: str, 
                    imgsz: Tuple[int, int] = (640, 640),
                    force: bool = False) -> Path:
        """
        Export a YOLO model to a specific backend format
        
        Args:
            model_name: Model name (e.g., 'yolov11n-pose')
            backend: Backend type ('onnx', 'tensorrt', 'openvino')
            imgsz: Input image size (width, height)
            force: Force re-export even if file exists
            
        Returns:
            Path to exported model
        """
        if backend == 'pytorch':
            # PyTorch doesn't need export
            return self.download_model(model_name)
        
        exported_path = self.get_model_path(model_name, backend)
        
        # Check if already exported
        if not force:
            if backend == 'openvino':
                if exported_path.exists() and (exported_path / f"{model_name}.xml").exists():
                    print(f"Model already exported for {backend}: {exported_path}")
                    return exported_path
            else:
                if exported_path.exists():
                    print(f"Model already exported for {backend}: {exported_path}")
                    return exported_path
        
        # Download source PyTorch model first
        pt_model_path = self.download_model(model_name)
        
        print(f"Exporting {model_name} to {backend} format (resolution: {imgsz})...")
        
        try:
            model = YOLO(model_name)  # Ultralytics handles the .pt extension
            
            # Export based on backend
            if backend == 'onnx':
                export_path = model.export(format='onnx', imgsz=list(imgsz))
            elif backend == 'tensorrt':
                # TensorRT export (requires NVIDIA GPU)
                export_path = model.export(format='engine', imgsz=list(imgsz), half=True)
            elif backend == 'openvino':
                # OpenVINO export
                export_path = model.export(format='openvino', imgsz=list(imgsz))
            else:
                raise ValueError(f"Export not supported for backend: {backend}")
            
            print(f"Model exported successfully to: {export_path}")
            return Path(export_path)
            
        except Exception as e:
            print(f"Error exporting model to {backend}: {e}")
            print(f"Falling back to PyTorch backend")
            return pt_model_path
    
    def load_model(self, model_name: str, backend: str = 'pytorch',
                   imgsz: Tuple[int, int] = (640, 640),
                   device: str = 'auto') -> YOLO:
        """
        Load a YOLO model with specified backend
        
        Args:
            model_name: Model name (e.g., 'yolov11n-pose')
            backend: Backend type ('pytorch', 'tensorrt', 'openvino', 'onnx')
            imgsz: Input image size (width, height)
            device: Device to use ('cpu', 'cuda', 'auto')
            
        Returns:
            Loaded YOLO model
        """
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend: {backend}. Supported: {self.SUPPORTED_BACKENDS}")
        
        print(f"Loading model: {model_name} with backend: {backend}")
        
        try:
            if backend == 'pytorch':
                # Check if model exists in our local models directory
                local_model_path = self.model_dir / f"{model_name}.pt"
                
                if local_model_path.exists():
                    # Load from local directory
                    print(f"  Loading from local directory: {local_model_path}")
                    model = YOLO(str(local_model_path))
                else:
                    # Download and save to local directory first
                    downloaded_path = self.download_model(model_name)
                    if downloaded_path.exists():
                        model = YOLO(str(downloaded_path))
                    else:
                        # Fallback to Ultralytics handling
                        model = YOLO(model_name)
            else:
                # Export and load for other backends
                model_path = self.export_model(model_name, backend, imgsz=imgsz)
                
                if backend == 'onnx':
                    model = YOLO(str(model_path))
                elif backend == 'tensorrt':
                    model = YOLO(str(model_path))
                elif backend == 'openvino':
                    # OpenVINO expects the directory path
                    model = YOLO(str(model_path))
            
            print(f"Model loaded successfully: {model_name} ({backend})")
            return model
            
        except Exception as e:
            print(f"Error loading model with {backend} backend: {e}")
            raise
    
    @staticmethod
    def get_available_backends() -> list:
        """
        Get list of available backends on current system
        
        Returns:
            List of available backend names
        """
        available = ['pytorch']  # PyTorch always available with ultralytics
        
        # Check for ONNX Runtime
        try:
            import onnxruntime
            available.append('onnx')
        except ImportError:
            pass
        
        # Check for TensorRT (NVIDIA GPU)
        try:
            import tensorrt
            available.append('tensorrt')
        except ImportError:
            pass
        
        # Check for OpenVINO
        try:
            import openvino
            available.append('openvino')
        except ImportError:
            pass
        
        return available

