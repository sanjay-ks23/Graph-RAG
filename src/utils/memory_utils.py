"""CUDA memory management utilities"""

import torch
import gc
import os
from functools import wraps
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def clear_cuda_cache():
    """Clear CUDA cache and run garbage collection aggressively"""
    if torch.cuda.is_available():
        # Run garbage collection multiple times
        gc.collect()
        gc.collect()
        
        # Empty CUDA cache
        torch.cuda.empty_cache()
        
        # Synchronize to ensure all operations are complete
        torch.cuda.synchronize()
        
        # Run garbage collection again
        gc.collect()

def with_memory_cleanup(func):
    """Decorator to automatically clean up memory before and after function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
        
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
        
        return result
    return wrapper

def get_gpu_memory_info():
    """Get GPU memory information"""
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
    allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)  # GB
    reserved_memory = torch.cuda.memory_reserved(device) / (1024**3)  # GB
    free_memory = total_memory - allocated_memory
    
    info = f"""
GPU Memory Status:
  Total: {total_memory:.2f} GB
  Allocated: {allocated_memory:.2f} GB
  Reserved: {reserved_memory:.2f} GB
  Free: {free_memory:.2f} GB
"""
    return info

def log_gpu_memory():
    """Log GPU memory information"""
    if torch.cuda.is_available():
        logger.info(get_gpu_memory_info())

def set_memory_efficient_mode():
    """Set PyTorch to memory-efficient mode"""
    if torch.cuda.is_available():
        # Enable TF32 for better performance on Ampere GPUs (using new API)
        try:
            # New API (PyTorch 2.9+)
            torch.backends.cuda.matmul.fp32_precision = 'tf32'
            torch.backends.cudnn.conv.fp32_precision = 'tf32'
        except AttributeError:
            # Fallback to old API for older PyTorch versions
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudnn benchmarking for better performance
        torch.backends.cudnn.benchmark = True
        
        logger.info("Set memory-efficient mode for CUDA")
