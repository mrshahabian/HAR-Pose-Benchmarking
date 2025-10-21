"""Configuration management for benchmarking"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import platform


class BenchmarkConfig:
    """Manage benchmark configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_default_config()
        
        if config_path and Path(config_path).exists():
            self.load_from_file(config_path)
    
    def _load_default_config(self) -> Dict:
        """Load default configuration"""
        return {
            'models': ['yolov11n-pose', 'yolov11s-pose', 'yolov11m-pose'],
            'resolutions': [[640, 640], [960, 960]],
            'backends': ['pytorch'],
            'sources': [
                {'type': 'usb', 'device': 0, 'name': 'USB_Webcam'}
            ],
            'duration_seconds': 60,
            'warmup_frames': 20,
            'confidence_threshold': 0.5,
            'output_dir': 'results/benchmarks',
            'save_visualizations': True,
            'export_csv': True,
            'export_json': True,
            'annotations': {
                'enabled': False,
                'coco_json_path': None
            },
            'device_settings': {
                'auto_detect': True
            }
        }
    
    def load_from_file(self, config_path: str):
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to YAML configuration file
        """
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            
            # Merge with default config
            self._merge_config(user_config)
            print(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration")
    
    def _merge_config(self, user_config: Dict):
        """Merge user configuration with defaults"""
        for key, value in user_config.items():
            if key in self.config and isinstance(self.config[key], dict) and isinstance(value, dict):
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def save_to_file(self, output_path: str):
        """
        Save configuration to YAML file
        
        Args:
            output_path: Path to output YAML file
        """
        try:
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            print(f"Configuration saved to {output_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set configuration value
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
    
    def get_device_info(self) -> Dict:
        """
        Get information about the current device
        
        Returns:
            Dictionary containing device information
        """
        device_info = {
            'platform': platform.system(),
            'processor': platform.processor(),
            'machine': platform.machine(),
            'python_version': platform.python_version(),
        }
        
        # Check for NVIDIA GPU
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            device_info['gpu'] = gpu_name
            pynvml.nvmlShutdown()
        except:
            device_info['gpu'] = 'None'
        
        # Check if Jetson
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                device_info['jetson'] = f.read().strip()
        except:
            device_info['jetson'] = 'No'
        
        return device_info
    
    def filter_backends_by_device(self) -> List[str]:
        """
        Filter available backends based on current device
        
        Returns:
            List of available backends
        """
        from ..inference.backend_factory import BackendFactory
        
        requested_backends = self.get('backends', ['pytorch'])
        available_backends = BackendFactory.get_available_backends()
        
        # Filter to only available backends
        filtered = [b for b in requested_backends if b in available_backends]
        
        if not filtered:
            print("Warning: No requested backends available. Using PyTorch.")
            filtered = ['pytorch']
        
        return filtered
    
    def validate(self) -> bool:
        """
        Validate configuration
        
        Returns:
            True if configuration is valid, False otherwise
        """
        required_keys = ['models', 'resolutions', 'backends', 'sources']
        
        for key in required_keys:
            if key not in self.config:
                print(f"Error: Missing required configuration key: {key}")
                return False
        
        # Validate models
        if not self.config['models']:
            print("Error: No models specified")
            return False
        
        # Validate resolutions
        if not self.config['resolutions']:
            print("Error: No resolutions specified")
            return False
        
        # Validate sources
        if not self.config['sources']:
            print("Error: No video sources specified")
            return False
        
        return True
    
    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "="*60)
        print("BENCHMARK CONFIGURATION")
        print("="*60)
        print(f"Models:                 {', '.join(self.config['models'])}")
        print(f"Resolutions:            {self.config['resolutions']}")
        print(f"Backends:               {', '.join(self.config['backends'])}")
        print(f"Duration:               {self.config['duration_seconds']}s")
        print(f"Warmup Frames:          {self.config['warmup_frames']}")
        print(f"Output Directory:       {self.config['output_dir']}")
        print(f"Video Sources:          {len(self.config['sources'])}")
        for i, source in enumerate(self.config['sources']):
            print(f"  {i+1}. {source.get('name', source['type'])}: {source}")
        print("="*60 + "\n")
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access"""
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-like assignment"""
        self.config[key] = value

