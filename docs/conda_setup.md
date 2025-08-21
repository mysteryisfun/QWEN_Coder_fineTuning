# Conda Environment Setup Guide

## Environment Requirements
- **Python Version**: 3.10
- **CUDA Version**: 11.8 (for RTX 3050 compatibility)
- **Environment Name**: `finetune`

## Setup Instructions

### 1. Create Conda Environment
```bash
conda create -n finetune python=3.10 -y
conda activate finetune
```

### 2. Install PyTorch with CUDA Support
```bash
# Install PyTorch with CUDA 11.8 support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 3. Install Other Dependencies
```bash
# Install remaining packages via pip
pip install -r requirements.txt
```

### 4. Verify Installation
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")
```

## Alternative: Single Command Setup
```bash
# Create environment and install everything
conda create -n finetune python=3.10 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda activate finetune
pip install -r requirements.txt
```

## Troubleshooting

### If CUDA is still not detected:
1. **Check NVIDIA drivers**: `nvidia-smi`
2. **Verify conda environment**: `conda list pytorch`
3. **Check CUDA toolkit**: `nvcc --version`
4. **Restart terminal** after installation

### Common Issues:
- **Mixed CPU/CUDA PyTorch**: Uninstall all PyTorch versions and reinstall with conda
- **Version conflicts**: Use conda for PyTorch, pip for everything else
- **Memory issues**: Ensure 4GB+ free space and 16GB+ RAM

## Environment Activation
Always activate the environment before working:
```bash
conda activate finetune
```

## Jupyter Notebook Setup
After environment setup, install Jupyter kernel:
```bash
conda activate finetune
python -m ipykernel install --user --name finetune --display-name "Python (finetune)"
```

Then select the "Python (finetune)" kernel in VS Code/Jupyter.
