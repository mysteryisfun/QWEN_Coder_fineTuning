#!/usr/bin/env python3
"""
Main training script for Qwen Coder fine-tuning
Usage: python main.py [--test] [--config path/to/config.json]
"""
import argparse
import logging
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config import get_training_config, get_test_config, validate_config, print_config
from data_utils import load_processed_data, create_datasets_and_loaders
from model_utils import load_and_setup_model, create_training_arguments
from training_utils import TrainingManager, create_trainer
from memory_utils import setup_memory_optimization, clear_gpu_memory, get_model_memory_footprint

logger = logging.getLogger(__name__)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Qwen Coder Fine-tuning')
    parser.add_argument('--test', action='store_true', help='Run in test mode with minimal data')
    parser.add_argument('--config', type=str, help='Path to custom config file')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.test:
        config = get_test_config()
        print("🧪 Running in TEST MODE with minimal data")
    else:
        config = get_training_config()
        print("🚀 Running in FULL TRAINING MODE")
    
    # Validate configuration
    validate_config(config)
    print_config(config)
    
    print("\n" + "="*50)
    print("STARTING QWEN CODER FINE-TUNING")
    print("="*50)
    
    # Setup advanced memory optimization
    setup_memory_optimization()
    
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    try:
        # Step 1: Load processed data
        print("\n📊 Step 1: Loading processed data...")
        train_data, val_data, test_data, metadata = load_processed_data(config['data_dir'])
        print(f"✅ Data loaded - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Step 2: Load and setup model
        print("\n🤖 Step 2: Loading and setting up model...")
        model, tokenizer = load_and_setup_model(config)
        
        # Check model memory footprint
        model_memory = get_model_memory_footprint(model)
        print("✅ Model loaded and configured with QLoRA")
        
        # Step 3: Create datasets and data loaders
        print("\n📚 Step 3: Creating datasets and data loaders...")
        train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = create_datasets_and_loaders(
            train_data, val_data, test_data, tokenizer, config
        )
        print(f"✅ Datasets created - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        # Step 4: Create training arguments and trainer
        print("\n⚙️ Step 4: Setting up training configuration...")
        training_args = create_training_arguments(config)
        
        trainer = create_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            training_args=training_args,
            data_collator=train_loader.collate_fn
        )
        print("✅ Trainer configured")
        
        # Step 5: Initialize training manager
        training_manager = TrainingManager(config)
        
        if not args.eval_only:
            # Step 6: Train the model
            print("\n🏋️ Step 5: Starting training...")
            training_metrics = training_manager.train_model(trainer)
            print("✅ Training completed!")
            
            # Print training summary
            print(f"\n📈 Training Summary:")
            print(f"   Final Loss: {training_metrics['train_loss']:.4f}")
            print(f"   Total Steps: {training_metrics['train_steps']}")
            print(f"   Max GPU Memory: {training_metrics['max_gpu_memory_gb']:.2f} GB")
        
        # Step 7: Evaluate the model
        print("\n📊 Step 6: Evaluating model...")
        eval_metrics = training_manager.evaluate_model(trainer)
        print("✅ Evaluation completed!")
        
        # Print evaluation summary
        print(f"\n📋 Evaluation Summary:")
        print(f"   Eval Loss: {eval_metrics['eval_loss']:.4f}")
        print(f"   Eval Runtime: {eval_metrics['eval_runtime']:.2f}s")
        
        # Step 8: Test generation (optional)
        print("\n🧪 Step 7: Testing model generation...")
        test_prompt = """Write unit tests for the following Python code:

```python
def add_numbers(a, b):
    return a + b
```

Requirements:
- Write comprehensive unit tests using pytest
- Include edge cases and error handling
- Use descriptive test names
- Add assertions to verify correctness

Unit tests:"""
        
        from training_utils import test_model_generation
        generated_text = test_model_generation(model, tokenizer, test_prompt)
        print(f"✅ Generated unit test (first 200 chars):")
        print(f"   {generated_text[:200]}...")
        
        print("\n🎉 FINE-TUNING COMPLETED SUCCESSFULLY!")
        print(f"📁 Model saved to: {config['output_dir']}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"\n❌ Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
