"""
Model loading and configuration def create_quantization_config() -> BitsAndBytesConfig:
    
    logger.info("Creating 4-bit quantization - all GPU to avoid device conflicts")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        # No CPU offload to avoid device conflicts
    )
    
    return quantization_configder fine-tuning
"""
import logging
from typing import Dict, Any, Tuple, Optional

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)

logger = logging.getLogger(__name__)


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load and configure tokenizer"""
    logger.info(f"Loading tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set up special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Tokenizer loaded - Vocab size: {tokenizer.vocab_size}")
    return tokenizer


def create_quantization_config() -> BitsAndBytesConfig:
    """Create 4-bit quantization config for maximum compression"""
    logger.info("Creating 4-bit quantization - GPU only to avoid device conflicts")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        # NO CPU offload - keep everything on GPU
    )
    
    return quantization_config


def load_model(model_name: str, quantization_config: BitsAndBytesConfig) -> AutoModelForCausalLM:
    """Load model with FORCED GPU-only configuration"""
    logger.info(f"Loading 7B model FORCED GPU-only with 4-bit quantization: {model_name}")
    
    # FORCE everything on GPU - no CPU at all
    device_map = {}
    
    # All embeddings on GPU
    device_map["model.embed_tokens"] = "cuda:0"
    device_map["model.norm"] = "cuda:0"
    device_map["lm_head"] = "cuda:0"
    
    # ALL 28 layers on GPU (no CPU offloading)
    for i in range(28):
        device_map[f"model.layers.{i}"] = "cuda:0"
    
    logger.info("FORCED GPU-only: ALL 28 layers + embeddings on cuda:0")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,  # Force our GPU-only mapping
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    logger.info(f"Model loaded - Parameters: {model.num_parameters():,}")
    return model


def create_lora_config(config: Dict[str, Any]) -> LoraConfig:
    """Create LoRA configuration for QLoRA fine-tuning"""
    logger.info("Creating LoRA configuration")
    
    lora_config = LoraConfig(
        r=config.get('lora_r', 16),  # Rank
        lora_alpha=config.get('lora_alpha', 32),  # Alpha scaling
        target_modules=config.get('target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        lora_dropout=config.get('lora_dropout', 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    logger.info(f"LoRA config - Rank: {lora_config.r}, Alpha: {lora_config.lora_alpha}")
    return lora_config


def setup_model_for_training(
    model: AutoModelForCausalLM, 
    lora_config: LoraConfig
) -> AutoModelForCausalLM:
    """Prepare model for QLoRA training"""
    logger.info("Setting up model for QLoRA training")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
    logger.info(f"Total parameters: {all_params:,}")
    
    return model


def create_training_arguments(config: Dict[str, Any]) -> TrainingArguments:
    """Create training arguments"""
    logger.info("Creating training arguments")
    
    training_args = TrainingArguments(
        # Output and logging
        output_dir=config['output_dir'],
        logging_dir=config.get('logging_dir', f"{config['output_dir']}/logs"),
        logging_steps=config.get('logging_steps', 10),
        save_steps=config.get('save_steps', 500),
        eval_steps=config.get('eval_steps', 500),
        
        # Training parameters
        num_train_epochs=config.get('num_epochs', 3),
        per_device_train_batch_size=config['train_batch_size'],
        per_device_eval_batch_size=config['eval_batch_size'],
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 32),  # High for memory efficiency
        learning_rate=config.get('learning_rate', 2e-4),
        weight_decay=config.get('weight_decay', 0.001),
        warmup_ratio=config.get('warmup_ratio', 0.03),
        
        # Memory optimization - EXTREME settings
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False,
        
        # Mixed precision with optimizations
        fp16=True,
        fp16_opt_level="O2",  # Aggressive FP16
        
        # Advanced memory techniques
        max_grad_norm=1.0,  # Gradient clipping
        
        # CPU optimizer offloading (experimental)
        optim="adamw_torch_fused",  # Faster optimizer
        
        # Evaluation and saving
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Early stopping
        save_total_limit=2,
        
        # Reporting
        report_to=["tensorboard"],
        run_name=config.get('run_name', 'qwen-coder-finetune'),
        
        # Other
        remove_unused_columns=False,
        push_to_hub=False,
    )
    
    logger.info(f"Training config - Epochs: {training_args.num_train_epochs}, LR: {training_args.learning_rate}")
    return training_args


def load_and_setup_model(config: Dict[str, Any]) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load and setup complete model pipeline"""
    logger.info("Starting model loading and setup")
    
    # Load tokenizer
    tokenizer = load_tokenizer(config['model_name'])
    
    # Create quantization config
    quantization_config = create_quantization_config()
    
    # Load model
    model = load_model(config['model_name'], quantization_config)
    
    # Create LoRA config
    lora_config = create_lora_config(config)
    
    # Setup for training
    model = setup_model_for_training(model, lora_config)
    
    logger.info("Model loading and setup completed")
    return model, tokenizer
