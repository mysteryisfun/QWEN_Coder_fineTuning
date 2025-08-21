"""
Training utilities for Qwen Coder fine-tuning
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import torch
from transformers import Trainer, TrainerCallback
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MemoryCallback(TrainerCallback):
    """Callback to monitor GPU memory usage during training"""
    
    def __init__(self):
        self.max_memory = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            current_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            self.max_memory = max(self.max_memory, current_memory)
            
            # Log every 50 steps
            if state.global_step % 50 == 0:
                logger.info(f"Step {state.global_step} - GPU Memory: {current_memory:.2f} GB (Max: {self.max_memory:.2f} GB)")


class TrainingManager:
    """Manages the complete training process"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = config['output_dir']
        self.setup_logging()
        
    def setup_logging(self):
        """Setup training logging"""
        os.makedirs(f"{self.output_dir}/logs", exist_ok=True)
        
        # Create training log file
        log_file = f"{self.output_dir}/logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logger.info(f"Training logging setup - Log file: {log_file}")
        
    def save_config(self):
        """Save training configuration"""
        config_path = f"{self.output_dir}/training_config.json"
        
        # Add timestamp
        config_with_timestamp = self.config.copy()
        config_with_timestamp['training_started'] = datetime.now().isoformat()
        
        with open(config_path, 'w') as f:
            json.dump(config_with_timestamp, f, indent=2)
            
        logger.info(f"Training config saved to {config_path}")
        
    def train_model(self, trainer: Trainer) -> Dict[str, Any]:
        """Execute the training process"""
        logger.info("Starting model training")
        
        # Save config
        self.save_config()
        
        # Add memory callback
        memory_callback = MemoryCallback()
        trainer.add_callback(memory_callback)
        
        try:
            # Start training
            train_result = trainer.train()
            
            # Save the model
            logger.info("Saving trained model")
            trainer.save_model()
            trainer.save_state()
            
            # Save training metrics
            metrics = {
                'train_loss': train_result.training_loss,
                'train_steps': train_result.global_step,
                'max_gpu_memory_gb': memory_callback.max_memory,
                'training_completed': datetime.now().isoformat()
            }
            
            metrics_path = f"{self.output_dir}/training_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Training completed successfully!")
            logger.info(f"Final train loss: {train_result.training_loss:.4f}")
            logger.info(f"Max GPU memory used: {memory_callback.max_memory:.2f} GB")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
            
    def evaluate_model(self, trainer: Trainer) -> Dict[str, Any]:
        """Evaluate the trained model"""
        logger.info("Starting model evaluation")
        
        try:
            eval_result = trainer.evaluate()
            
            # Save evaluation metrics
            eval_metrics = {
                'eval_loss': eval_result['eval_loss'],
                'eval_runtime': eval_result['eval_runtime'],
                'eval_samples_per_second': eval_result['eval_samples_per_second'],
                'evaluation_completed': datetime.now().isoformat()
            }
            
            eval_path = f"{self.output_dir}/evaluation_metrics.json"
            with open(eval_path, 'w') as f:
                json.dump(eval_metrics, f, indent=2)
            
            logger.info(f"Evaluation completed!")
            logger.info(f"Eval loss: {eval_result['eval_loss']:.4f}")
            
            return eval_metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_args,
    data_collator
) -> Trainer:
    """Create Hugging Face Trainer"""
    logger.info("Creating Trainer")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    logger.info("Trainer created successfully")
    return trainer


def test_model_generation(model, tokenizer, test_prompt: str, max_length: int = 512) -> str:
    """Test model generation with a sample prompt"""
    logger.info("Testing model generation")
    
    # Tokenize input
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = inputs.to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after input)
    input_length = len(test_prompt)
    generated_only = generated_text[input_length:].strip()
    
    logger.info(f"Generated text length: {len(generated_only)} characters")
    return generated_only
