"""
Data handling module for Qwen Coder fine-tuning
"""
import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import psutil

logger = logging.getLogger(__name__)


class PromptTemplate:
    """Handles prompt formatting for unit test generation"""
    
    def __init__(self):
        self.system_prompt = """You are an expert Python developer specialized in writing comprehensive unit tests. Generate high-quality unit tests for the given Python code."""
        
        self.user_template = """Write unit tests for the following Python code:

```python
{code}
```

Requirements:
- Write comprehensive unit tests using pytest
- Include edge cases and error handling
- Use descriptive test names
- Add assertions to verify correctness

Unit tests:"""

    def format_input(self, code: str, question: str = None) -> str:
        """Format the input for the model"""
        if question and question.strip():
            enhanced_template = f"""Problem: {question}

Write unit tests for the following Python code:

```python
{code}
```

Requirements:
- Write comprehensive unit tests using pytest
- Include edge cases and error handling
- Use descriptive test names
- Add assertions to verify correctness

Unit tests:"""
            return enhanced_template
        else:
            return self.user_template.format(code=code)

    def format_training_example(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """Format a training sample for the model"""
        code = sample.get('code_ground_truth', '')
        question = sample.get('question', '')
        
        # Get unit tests - handle different formats
        unit_tests = sample.get('unit_tests', [])
        test_code = ''
        
        if isinstance(unit_tests, list) and len(unit_tests) > 0:
            first_test = unit_tests[0]
            
            if isinstance(first_test, dict):
                test_code = first_test.get('code', '')
                if not test_code:
                    possible_keys = ['test', 'unit_test', 'test_code', 'content']
                    for key in possible_keys:
                        if key in first_test and first_test[key]:
                            test_code = first_test[key]
                            break
            elif isinstance(first_test, str):
                test_code = first_test
            else:
                test_code = str(first_test)
                
        elif isinstance(unit_tests, str):
            test_code = unit_tests
        
        input_text = self.format_input(code, question)
        output_text = test_code.strip() if test_code else ''
        
        return {
            'input': input_text,
            'output': output_text,
            'full_text': f"{input_text}\n\n{output_text}"
        }


class UnitTestDataset(Dataset):
    """Memory-efficient PyTorch Dataset for unit test generation"""
    
    def __init__(self, data: List[Dict], tokenizer, prompt_template: PromptTemplate, max_length: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_length = max_length
        
        logger.info(f"Created dataset with {len(data)} samples (lazy loading)")
        
        # Test process a few samples to check for issues
        test_samples = min(5, len(data))
        successful_samples = 0
        
        for i in range(test_samples):
            try:
                formatted = self.prompt_template.format_training_example(data[i])
                if formatted['output'].strip():
                    successful_samples += 1
                else:
                    logger.warning(f"Sample {i} has empty output")
            except Exception as e:
                logger.warning(f"Failed to process test sample {i}: {str(e)}")
        
        logger.info(f"Test processing: {successful_samples}/{test_samples} samples successful")
        
        if successful_samples == 0:
            logger.error("No samples could be processed successfully!")
            raise ValueError("Dataset contains no valid samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            sample_data = self.data[idx]
            formatted = self.prompt_template.format_training_example(sample_data)
            
            if not formatted['output'].strip():
                logger.warning(f"Sample {idx} has empty output, using placeholder")
                formatted['output'] = "# No unit test available for this code"
            
            full_text = formatted['full_text']
            
            # Tokenize
            encoding = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt"
            )
            
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            labels = input_ids.clone()
            
            # Find where the output starts to mask input tokens
            input_text = formatted['input']
            input_encoding = self.tokenizer(
                input_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt"
            )
            input_length = input_encoding['input_ids'].shape[1]
            
            # Mask input tokens in labels (set to -100)
            if input_length < len(labels):
                labels[:input_length] = -100
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
            # Return minimal valid sample
            dummy_text = "def dummy(): pass"
            encoding = self.tokenizer(
                dummy_text,
                truncation=True,
                max_length=32,
                padding=False,
                return_tensors="pt"
            )
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            labels = input_ids.clone()
            labels[:] = -100
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }


class CustomDataCollator:
    """Custom data collator for batching samples"""
    
    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels
        }


def load_processed_data(data_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """Load preprocessed data splits"""
    data_dir = Path(data_dir)
    
    logger.info("Loading preprocessed data splits")
    
    with open(data_dir / 'train_split.pkl', 'rb') as f:
        train_data = pickle.load(f)
        
    with open(data_dir / 'val_split.pkl', 'rb') as f:
        val_data = pickle.load(f)
        
    with open(data_dir / 'test_split.pkl', 'rb') as f:
        test_data = pickle.load(f)
    
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Data loaded - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data, metadata


def create_datasets_and_loaders(
    train_data: List[Dict],
    val_data: List[Dict], 
    test_data: List[Dict],
    tokenizer,
    config: Dict[str, Any]
) -> Tuple[Dataset, Dataset, Dataset, DataLoader, DataLoader, DataLoader]:
    """Create datasets and data loaders"""
    
    prompt_template = PromptTemplate()
    
    # Create datasets with memory limits
    max_train = config.get('max_train_samples', 1000)
    max_val = config.get('max_val_samples', 200)
    max_test = config.get('max_test_samples', 200)
    
    logger.info(f"Creating datasets with limits - Train: {max_train}, Val: {max_val}, Test: {max_test}")
    
    train_dataset = UnitTestDataset(
        data=train_data[:max_train],
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        max_length=config['max_length']
    )
    
    val_dataset = UnitTestDataset(
        data=val_data[:max_val],
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        max_length=config['max_length']
    )
    
    test_dataset = UnitTestDataset(
        data=test_data[:max_test],
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        max_length=config['max_length']
    )
    
    # Create data collator
    data_collator = CustomDataCollator(tokenizer, config['max_length'])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['eval_batch_size'],
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['eval_batch_size'],
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Created data loaders - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024
