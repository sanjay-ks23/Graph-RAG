"""CUDA memory management utilities"""

import torch
import gc
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def clear_cuda_cache():
    """Clear CUDA cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Cleared CUDA cache and ran garbage collection")

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
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudnn benchmarking for better performance
        torch.backends.cudnn.benchmark = True
        
        logger.info("Set memory-efficient mode for CUDA")
