"""System resource monitoring (CPU, GPU, memory, power)"""

import os
import time
import threading
import subprocess
from typing import Dict, List, Optional
import psutil


class SystemMonitor:
    """Monitor system resources during benchmarking"""
    
    def __init__(self, interval: float = 1.0):
        """
        Initialize system monitor
        
        Args:
            interval: Sampling interval in seconds
        """
        self.interval = interval
        self.monitoring = False
        self.monitor_thread = None
        
        # Storage for measurements
        self.cpu_percent: List[float] = []
        self.ram_mb: List[float] = []
        self.gpu_percent: List[float] = []
        self.vram_mb: List[float] = []
        self.power_watts: List[float] = []
        self.temperature_celsius: List[float] = []
        
        # Detect available monitoring capabilities
        self.has_nvidia_gpu = self._check_nvidia_gpu()
        self.is_jetson = self._check_jetson()
        
        # Initialize NVIDIA monitoring if available
        self.nvml_initialized = False
        if self.has_nvidia_gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.pynvml = pynvml
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                print("NVIDIA GPU monitoring enabled (NVML)")
            except Exception as e:
                print(f"NVML not available: {e}")
                self.nvml_initialized = False
    
    def _check_nvidia_gpu(self) -> bool:
        """Check if NVIDIA GPU is available"""
        try:
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, 
                                  timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def _check_jetson(self) -> bool:
        """Check if running on NVIDIA Jetson"""
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                return True
        except FileNotFoundError:
            return False
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        return psutil.cpu_percent(interval=0.1)
    
    def _get_ram_usage(self) -> float:
        """Get RAM usage in MB"""
        return psutil.virtual_memory().used / (1024 * 1024)
    
    def _get_gpu_usage_nvml(self) -> tuple:
        """Get GPU usage using NVML (NVIDIA Management Library)"""
        if not self.nvml_initialized:
            return 0.0, 0.0, 0.0, 0.0
        
        try:
            # GPU utilization
            util = self.pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            gpu_percent = util.gpu
            
            # VRAM usage
            mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            vram_mb = mem_info.used / (1024 * 1024)
            
            # Power consumption
            try:
                power_mw = self.pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                power_watts = power_mw / 1000.0
            except:
                power_watts = 0.0
            
            # Temperature
            try:
                temp = self.pynvml.nvmlDeviceGetTemperature(
                    self.gpu_handle, 
                    self.pynvml.NVML_TEMPERATURE_GPU
                )
            except:
                temp = 0.0
            
            return gpu_percent, vram_mb, power_watts, temp
            
        except Exception as e:
            print(f"Error reading GPU metrics: {e}")
            return 0.0, 0.0, 0.0, 0.0
    
    def _get_gpu_usage_nvidia_smi(self) -> tuple:
        """Get GPU usage using nvidia-smi command"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                values = result.stdout.strip().split(',')
                gpu_percent = float(values[0])
                vram_mb = float(values[1])
                power_watts = float(values[2]) if values[2] != '[N/A]' else 0.0
                temp = float(values[3])
                return gpu_percent, vram_mb, power_watts, temp
                
        except Exception as e:
            pass
        
        return 0.0, 0.0, 0.0, 0.0
    
    def _get_jetson_stats(self) -> tuple:
        """Get Jetson statistics using tegrastats"""
        # For Jetson, power monitoring requires parsing tegrastats output
        # This is a simplified version - full implementation would parse tegrastats
        try:
            # Try to read from tegrastats
            result = subprocess.run(
                ['tegrastats', '--interval', '100'],
                capture_output=True,
                text=True,
                timeout=1
            )
            
            # Parse tegrastats output (example: VDD_IN 5123mW)
            power_watts = 0.0
            if 'VDD_IN' in result.stdout:
                for line in result.stdout.split('\n'):
                    if 'VDD_IN' in line:
                        # Extract power value
                        parts = line.split('VDD_IN')[1].split('mW')[0].strip()
                        power_watts = float(parts) / 1000.0
                        break
            
            return power_watts
            
        except Exception:
            return 0.0
    
    def _monitor_loop(self):
        """Main monitoring loop (runs in separate thread)"""
        while self.monitoring:
            # CPU and RAM
            cpu = self._get_cpu_usage()
            ram = self._get_ram_usage()
            
            self.cpu_percent.append(cpu)
            self.ram_mb.append(ram)
            
            # GPU metrics
            if self.has_nvidia_gpu:
                if self.nvml_initialized:
                    gpu, vram, power, temp = self._get_gpu_usage_nvml()
                else:
                    gpu, vram, power, temp = self._get_gpu_usage_nvidia_smi()
                
                self.gpu_percent.append(gpu)
                self.vram_mb.append(vram)
                self.power_watts.append(power)
                self.temperature_celsius.append(temp)
            
            # Jetson-specific power monitoring
            elif self.is_jetson:
                power = self._get_jetson_stats()
                self.power_watts.append(power)
            
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring in background thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("System monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=2)
        print("System monitoring stopped")
    
    def get_metrics(self) -> Dict:
        """
        Get aggregated system metrics
        
        Returns:
            Dictionary containing system metrics
        """
        import numpy as np
        
        metrics = {
            'cpu_percent_avg': float(np.mean(self.cpu_percent)) if self.cpu_percent else 0.0,
            'cpu_percent_max': float(np.max(self.cpu_percent)) if self.cpu_percent else 0.0,
            'cpu_percent_std': float(np.std(self.cpu_percent)) if self.cpu_percent else 0.0,
            'ram_mb_avg': float(np.mean(self.ram_mb)) if self.ram_mb else 0.0,
            'ram_mb_max': float(np.max(self.ram_mb)) if self.ram_mb else 0.0,
            'gpu_percent_avg': float(np.mean(self.gpu_percent)) if self.gpu_percent else 0.0,
            'gpu_percent_max': float(np.max(self.gpu_percent)) if self.gpu_percent else 0.0,
            'vram_mb_avg': float(np.mean(self.vram_mb)) if self.vram_mb else 0.0,
            'vram_mb_max': float(np.max(self.vram_mb)) if self.vram_mb else 0.0,
            'power_watts_avg': float(np.mean(self.power_watts)) if self.power_watts else 0.0,
            'power_watts_max': float(np.max(self.power_watts)) if self.power_watts else 0.0,
            'temperature_celsius_avg': float(np.mean(self.temperature_celsius)) if self.temperature_celsius else 0.0,
            'temperature_celsius_max': float(np.max(self.temperature_celsius)) if self.temperature_celsius else 0.0,
        }
        
        return metrics
    
    def reset(self):
        """Reset all measurements"""
        self.cpu_percent.clear()
        self.ram_mb.clear()
        self.gpu_percent.clear()
        self.vram_mb.clear()
        self.power_watts.clear()
        self.temperature_celsius.clear()
    
    def print_summary(self):
        """Print system metrics summary"""
        metrics = self.get_metrics()
        
        print("\n" + "="*60)
        print("SYSTEM RESOURCE METRICS")
        print("="*60)
        print(f"CPU Usage (avg/max):    {metrics['cpu_percent_avg']:.1f}% / {metrics['cpu_percent_max']:.1f}%")
        print(f"RAM Usage (avg/max):    {metrics['ram_mb_avg']:.0f} MB / {metrics['ram_mb_max']:.0f} MB")
        
        if metrics['gpu_percent_avg'] > 0:
            print(f"GPU Usage (avg/max):    {metrics['gpu_percent_avg']:.1f}% / {metrics['gpu_percent_max']:.1f}%")
            print(f"VRAM Usage (avg/max):   {metrics['vram_mb_avg']:.0f} MB / {metrics['vram_mb_max']:.0f} MB")
        
        if metrics['power_watts_avg'] > 0:
            print(f"Power Draw (avg/max):   {metrics['power_watts_avg']:.2f} W / {metrics['power_watts_max']:.2f} W")
        
        if metrics['temperature_celsius_avg'] > 0:
            print(f"Temperature (avg/max):  {metrics['temperature_celsius_avg']:.1f}°C / {metrics['temperature_celsius_max']:.1f}°C")
        
        print("="*60 + "\n")
    
    def __del__(self):
        """Cleanup NVML on destruction"""
        if self.nvml_initialized:
            try:
                self.pynvml.nvmlShutdown()
            except:
                pass

