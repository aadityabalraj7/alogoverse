#!/bin/bash

# Duality AI Space Station Hackathon - Environment Setup Script (Mac/Linux)
# This script sets up the "EDU" environment with all required dependencies

echo "Setting up Duality AI Space Station Hackathon Environment..."
echo "==========================================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Anaconda/Miniconda is not installed or not in PATH"
    echo "Please install Anaconda from: https://www.anaconda.com/products/distribution"
    exit 1
fi

# Create conda environment
echo "Creating conda environment 'EDU'..."
conda create -n EDU python=3.9 -y

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate EDU

# Install PyTorch with CUDA support (or CPU if no GPU)
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
else
    echo "No CUDA detected, installing CPU-only PyTorch..."
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi

# Install YOLOv8 and dependencies
echo "Installing YOLOv8 and dependencies..."
pip install ultralytics

# Install additional required packages
echo "Installing additional packages..."
pip install opencv-python
pip install matplotlib
pip install seaborn
pip install pandas
pip install numpy
pip install Pillow
pip install tqdm
pip install scikit-learn
pip install jupyter
pip install notebook

# Install visualization and reporting tools
pip install plotly
pip install streamlit

echo ""
echo "============================================"
echo "Environment setup completed successfully!"
echo "============================================"
echo ""
echo "To activate the environment, run:"
echo "conda activate EDU"
echo ""
echo "To deactivate the environment, run:"
echo "conda deactivate"
echo ""
echo "The environment includes:"
echo "- Python 3.9"
echo "- PyTorch with appropriate CUDA support"
echo "- YOLOv8 (ultralytics)"
echo "- OpenCV, Matplotlib, Seaborn"
echo "- Jupyter Notebook"
echo "- Streamlit for app development"
echo "- All required ML libraries" 