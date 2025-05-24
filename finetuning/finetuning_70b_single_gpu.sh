#!/bin/bash
#SBATCH -J finetuning-llama-3.1-70b-single-gpu
#SBATCH -t 24:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem=50G
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH -N 1

# Load required modules 
module load 2024 Python/3.12.3-GCCcore-13.3.0
module load 2024 CUDA/12.6.0

# Define the new virtual environment directory
VENV_DIR="/scratch-shared/mschaffelder/venv_finetune_1"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR"
  python -m venv $VENV_DIR
fi

# Activate virtual environment
echo "Activating virtual environment from $VENV_DIR"
source $VENV_DIR/bin/activate

# Set CUDA environment variables
export CUDA_HOME=${EBROOTCUDA}
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set PyTorch memory options to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:128

# Install required packages
echo "Installing packages from requirements.txt"
pip install -r requirements.txt 

# Run script
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Available GPUs: $(nvidia-smi -L)"

huggingface-cli login --token "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"
# Run single GPU version
python finetuning_70b_single_gpu.py 