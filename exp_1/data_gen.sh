#!/bin/bash
#SBATCH -J data-generation-llama-8b-finetuned
#SBATCH -t 08:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem=20G
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH -N 1


VENV_NAME="venv_exp_1"
VENV_BASE_DIR="/scratch-shared/mschaffelder" # Or any other persistent shared directory you prefer
VENV_DIR="$VENV_BASE_DIR/$VENV_NAME"
PYTHON_MODULE="Python/3.10.4-GCCcore-11.3.0"

# Load required modules
module load 2024
module load $PYTHON_MODULE

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR/bin" ]; then
    echo "Creating Python virtual environment $VENV_NAME at $VENV_DIR using $(python --version)"
    python -m venv $VENV_DIR
else
    echo "Virtual environment $VENV_NAME already exists at $VENV_DIR."
fi

# Activate virtual environment
# source /scratch-shared/mschaffelder/venv/bin/activate
echo "Activating virtual environment: $VENV_DIR"
source $VENV_DIR/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt

# Run script
cd $SLURM_SUBMIT_DIR
# Make sure CUDA devices are visible

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Available GPUs: $(nvidia-smi -L)"

huggingface-cli login --token "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"
python data_gen.py
