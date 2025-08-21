#!/usr/bin/env python3
"""
Test script for validating the training pipeline
Usage: python test_pipeline.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config import get_test_config, validate_config
from data_utils import load_processed_data
from model_utils import load_tokenizer

def test_data_loading():
    """Test data loading functionality"""
    print("🧪 Testing data loading...")
    
    config = get_test_config()
    
    try:
        train_data, val_data, test_data, metadata = load_processed_data(config['data_dir'])
        print(f"✅ Data loaded successfully")
        print(f"   Train: {len(train_data)} samples")
        print(f"   Val: {len(val_data)} samples") 
        print(f"   Test: {len(test_data)} samples")
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")
        return False

def test_tokenizer_loading():
    """Test tokenizer loading"""
    print("\n🧪 Testing tokenizer loading...")
    
    config = get_test_config()
    
    try:
        tokenizer = load_tokenizer(config['model_name'])
        print(f"✅ Tokenizer loaded successfully")
        print(f"   Model: {config['model_name']}")
        print(f"   Vocab size: {tokenizer.vocab_size}")
        return True
    except Exception as e:
        print(f"❌ Tokenizer loading failed: {str(e)}")
        return False

def test_config_validation():
    """Test configuration validation"""
    print("\n🧪 Testing configuration...")
    
    try:
        config = get_test_config()
        validate_config(config)
        print(f"✅ Configuration valid")
        return True
    except Exception as e:
        print(f"❌ Configuration validation failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("="*50)
    print("PIPELINE VALIDATION TESTS")
    print("="*50)
    
    tests = [
        test_config_validation,
        test_data_loading,
        test_tokenizer_loading,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "="*50)
    print(f"TEST RESULTS: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for training.")
        return True
    else:
        print("❌ Some tests failed. Please fix issues before training.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
