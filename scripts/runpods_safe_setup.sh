#!/bin/bash
# Safe setup script for Runpods environment
# Handles pip warnings and environment setup properly

echo "🚀 Runpods Safe Environment Setup"
echo "================================="

# Check if running as root (common in Runpods)
if [ "$EUID" -eq 0 ]; then 
   echo "⚠️ Running as root user detected (common in Runpods)"
   export PIP_ROOT_USER_ACTION=ignore
fi

# Update pip first (suppress warning)
echo "📦 Updating pip..."
python -m pip install --upgrade pip --quiet --root-user-action=ignore 2>/dev/null || python -m pip install --upgrade pip --quiet

# Function to safely install packages
safe_pip_install() {
    local package=$1
    echo "  Installing: $package"
    python -m pip install --quiet --root-user-action=ignore "$package" 2>/dev/null || \
    python -m pip install --quiet "$package"
}

# Install core requirements
echo "📚 Installing core packages..."

# PyTorch (with CUDA support)
safe_pip_install "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"

# Computer Vision packages
safe_pip_install "timm==0.9.16"
safe_pip_install "segmentation-models-pytorch==0.3.3"
safe_pip_install "albumentations==1.3.1"
safe_pip_install "opencv-python-headless==4.8.0.76"

# Utilities
safe_pip_install "pyyaml==6.0.1"
safe_pip_install "omegaconf==2.3.0"
safe_pip_install "torchinfo"
safe_pip_install "kaggle==1.5.16"

# Data science
safe_pip_install "pandas numpy matplotlib seaborn tqdm Pillow"

# Jupyter support
safe_pip_install "ipywidgets notebook jupyterlab"

echo "✅ All packages installed safely"

# Setup Kaggle credentials if found
setup_kaggle() {
    echo ""
    echo "🔍 Checking for Kaggle credentials..."
    
    # Check multiple locations
    KAGGLE_PATHS=(
        "/workspace/kaggle.json"
        "/runpod-volume/kaggle.json"
        "/workspace/.kaggle/kaggle.json"
        "./kaggle.json"
    )
    
    for kaggle_path in "${KAGGLE_PATHS[@]}"; do
        if [ -f "$kaggle_path" ]; then
            echo "  Found: $kaggle_path"
            
            # Setup in home directory
            mkdir -p ~/.kaggle
            cp "$kaggle_path" ~/.kaggle/kaggle.json
            chmod 600 ~/.kaggle/kaggle.json
            
            echo "✅ Kaggle credentials configured"
            return 0
        fi
    done
    
    echo "⚠️ No kaggle.json found. Upload to /workspace/kaggle.json"
    return 1
}

setup_kaggle

# Create workspace directories
echo ""
echo "📁 Setting up workspace directories..."

mkdir -p /workspace/data
mkdir -p /workspace/models  
mkdir -p /workspace/logs
mkdir -p /workspace/checkpoints

echo "  ✅ Created: /workspace/{data,models,logs,checkpoints}"

# Set environment variables
echo ""
echo "🔧 Setting environment variables..."

export VESUVIUS_DATA_PATH="/workspace/data/vesuvius"
export MODEL_CHECKPOINT_PATH="/workspace/models"
export LOG_PATH="/workspace/logs"
export CUDA_HOME="/usr/local/cuda"
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Save to bashrc for persistence
echo "" >> ~/.bashrc
echo "# Vesuvius Challenge Environment" >> ~/.bashrc
echo "export VESUVIUS_DATA_PATH='/workspace/data/vesuvius'" >> ~/.bashrc
echo "export MODEL_CHECKPOINT_PATH='/workspace/models'" >> ~/.bashrc
echo "export LOG_PATH='/workspace/logs'" >> ~/.bashrc
echo "export PIP_ROOT_USER_ACTION=ignore" >> ~/.bashrc

echo "  ✅ Environment variables set"

# Check GPU
echo ""
echo "🎮 Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "  ⚠️ nvidia-smi not found"
fi

# Python verification
echo ""
echo "🐍 Verifying Python environment..."
python -c "
import torch
import sys
print(f'  Python: {sys.version.split()[0]}')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA Version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "✅ Runpods environment setup complete!"
echo ""
echo "📝 Next steps:"
echo "  1. Upload kaggle.json to /workspace/"
echo "  2. Run: jupyter lab --ip=0.0.0.0 --port=8888 --allow-root"
echo "  3. Open setup_kaggle_and_train.ipynb"