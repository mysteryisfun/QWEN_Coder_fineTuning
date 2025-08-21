"""
Configuration management for Qwen Coder fine-tuning
"""
import os
from pathlib import Path
from typing import Dict, Any


def get_training_config() -> Dict[str, Any]:
    """Get default training configuration optimized for RTX 3050 4GB"""
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    config = {
        # Model configuration - Back to 7B as requested
        'model_name': 'Qwen/Qwen2.5-Coder-7B-Instruct',
        'max_length': 256,  # Very small sequences for testing
        
        # Data configuration
        'data_dir': str(project_root / 'data'),
        'max_train_samples': 50,    # Minimal for testing
        'max_val_samples': 10,      # Minimal for testing  
        'max_test_samples': 10,     # Minimal for testing
        
        # Training parameters - Optimized for 4GB VRAM
        'train_batch_size': 1,  # Minimal batch size
        'eval_batch_size': 1,   # Minimal batch size
        'gradient_accumulation_steps': 32,  # High accumulation = effective batch size 32
        'num_epochs': 1,        # Quick test
        'learning_rate': 1e-4,  # Lower LR for stability
        'weight_decay': 0.001,
        'warmup_ratio': 0.03,
        
        # LoRA configuration
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'target_modules': [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        
        # Output configuration
        'output_dir': str(project_root / 'models' / 'qwen-coder-finetune'),
        'run_name': 'qwen-coder-unittest-finetune',
        'logging_steps': 10,
        'save_steps': 250,
        'eval_steps': 250,
        
        # Hardware optimization
        'use_cuda': True,
        'mixed_precision': True,
        'gradient_checkpointing': True,
    }
    
    return config


def get_test_config() -> Dict[str, Any]:
    """Get configuration for testing with minimal resources"""
    config = get_training_config()
    
    # Override for testing
    config.update({
        'max_train_samples': 100,
        'max_val_samples': 20,
        'max_test_samples': 20,
        'num_epochs': 1,
        'save_steps': 50,
        'eval_steps': 50,
        'logging_steps': 5,
    })
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration"""
    required_keys = [
        'model_name', 'data_dir', 'output_dir',
        'train_batch_size', 'eval_batch_size',
        'num_epochs', 'learning_rate'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate paths
    data_dir = Path(config['data_dir'])
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return True


def print_config(config: Dict[str, Any]):
    """Print configuration in a readable format"""
    print("=" * 50)
    print("TRAINING CONFIGURATION")
    print("=" * 50)
    
    sections = {
        'Model': ['model_name', 'max_length'],
        'Data': ['max_train_samples', 'max_val_samples', 'max_test_samples'],
        'Training': ['train_batch_size', 'eval_batch_size', 'gradient_accumulation_steps', 
                    'num_epochs', 'learning_rate', 'weight_decay'],
        'LoRA': ['lora_r', 'lora_alpha', 'lora_dropout'],
        'Output': ['output_dir', 'run_name'],
    }
    
    for section, keys in sections.items():
        print(f"\n{section}:")
        print("-" * 20)
        for key in keys:
            if key in config:
                print(f"  {key}: {config[key]}")
    
    print("=" * 50)
