#!/bin/bash
#SBATCH -J finetuning-llama-3.1-8b-instant
#SBATCH -t 12:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem=10G
#SBATCH --gpus=2
#SBATCH --partition=gpu_h100
#SBATCH -N 1

# Load required modules 
module load 2024 Python/3.12.3-GCCcore-13.3.0
# module load 2023 Python/3.11.3-GCCcore-12.3.0

# Activate virtual environment
# source /scratch-shared/mschaffelder/.venv/bin/activate
source /scratch-shared/mschaffelder/venv/bin/activate

# Install required packages
pip install -r requirements.txt 

# Set PyTorch memory allocation settings to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Add environment variable for distributed training
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# Run script
cd $SLURM_SUBMIT_DIR
# Make sure CUDA devices are visible
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Available GPUs: $(nvidia-smi -L)"

#huggingface-cli login --token "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"
#python finetuning_dr.py
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 finetuning_dr.py