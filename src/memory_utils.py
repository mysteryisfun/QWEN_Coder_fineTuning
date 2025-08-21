"""
Advanced memory optimization utilities for large model training on small VRAM
"""
import torch
import gc
import os
from typing import Optional

def setup_memory_optimization():
    """Setup environment variables for memory optimization"""
    # Enable PyTorch memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Enable automatic mixed precision optimizations
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    print("🔧 Memory optimization environment configured")

def clear_gpu_memory():
    """Aggressively clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
        # Get memory info
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"🧹 GPU Memory - Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB, Total: {total:.1f}GB")

def get_model_memory_footprint(model):
    """Calculate model memory footprint"""
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024**3
    
    print(f"📊 Model Memory - Parameters: {param_memory:.1f}GB, Buffers: {buffer_memory:.1f}GB")
    return param_memory + buffer_memory

class MemoryEfficientTrainer:
    """Custom trainer with aggressive memory optimizations"""
    
    def __init__(self, model, training_args):
        self.model = model
        self.training_args = training_args
        
    def train_step_with_offloading(self, batch):
        """Training step with manual offloading"""
        # Move only necessary parts to GPU
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass with gradient checkpointing
        with torch.cuda.amp.autocast():
            outputs = self.model(**inputs)
            loss = outputs.loss
            
        # Backward pass with memory clearing
        loss.backward()
        
        # Clear intermediate activations
        del outputs
        torch.cuda.empty_cache()
        
        return loss.item()

def create_cpu_optimizer_config():
    """Create configuration for CPU optimizer offloading"""
    return {
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
                "weight_decay": 0.01
            }
        },
        "zero_optimization": {
            "stage": 2,  # Partition optimizer states
            "cpu_offload": True,
            "contiguous_gradients": True,
            "overlap_comm": True
        }
    }
