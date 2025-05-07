#!/bin/bash
#SBATCH -J finetuning-llama-3.1-70b-instant
#SBATCH -t 12:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem=50G
#SBATCH --gpus=3
#SBATCH --partition=gpu_h100
#SBATCH -N 1

# Load required modules 
module load 2024
# module load Python/3.10.4-GCCcore-11.3.0

# Activate virtual environment
source /scratch-shared/mschaffelder/.venv/bin/activate

# Install required packages
pip install -r requirements.txt 

# Run script
cd $SLURM_SUBMIT_DIR
# Make sure CUDA devices are visible
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Available GPUs: $(nvidia-smi -L)"

huggingface-cli login --token "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"
python finetuning_local.py