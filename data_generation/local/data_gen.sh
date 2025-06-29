#!/bin/bash
#SBATCH -J data-generation-llama-8b
#SBATCH -t 03:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem=20G
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH -N 1


# Load necessary environment modules
module purge

# Try to load available modules - check what's available on the system
module load 2024

# Check available CUDA modules
module avail CUDA 2>/dev/null || echo "CUDA modules not found, will use conda/pip CUDA"

# Check available Python modules  
module avail Python 2>/dev/null || echo "Python modules not found, will use system Python"

# Try to load common modules if they exist
module load CUDA/12.1.1 2>/dev/null || module load CUDA 2>/dev/null || echo "No CUDA module loaded"
module load Python/3.10.4-GCCcore-11.3.0 2>/dev/null || module load Python 2>/dev/null || echo "No Python module loaded"

VENV_NAME="venv_exp_1"
VENV_BASE_DIR="/scratch-shared/mschaffelder" # Or any other persistent shared directory you prefer
VENV_DIR="$VENV_BASE_DIR/$VENV_NAME"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR/bin" ]; then
    echo "Creating Python virtual environment $VENV_NAME at $VENV_DIR using $(python --version)"
    python -m venv $VENV_DIR
else
    echo "Virtual environment $VENV_NAME already exists at $VENV_DIR."
fi

# Activate virtual environment
echo "Activating virtual environment: $VENV_DIR"
source $VENV_DIR/bin/activate

# Run script
cd $SLURM_SUBMIT_DIR
# Make sure CUDA devices are visible

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Available GPUs: $(nvidia-smi -L)"

# Login to Hugging Face using environment variable
if [ -n "$HF_TOKEN" ]; then
    echo "Logging in to Hugging Face using environment variable..."
    huggingface-cli login --token "$HF_TOKEN"
elif [ -f ~/.hf_token ]; then
    echo "Loading Hugging Face token from ~/.hf_token..."
    source ~/.hf_token
    huggingface-cli login --token "$HF_TOKEN"
elif [ -f ~/.cache/huggingface/token ]; then
    echo "Using existing Hugging Face token from cache..."
else
    echo "Warning: No Hugging Face token found. Set HF_TOKEN environment variable or login manually."
fi  

python data_gen_local.py