#!/bin/bash
#SBATCH -J finetuning-llama-3.1-70b-single-medium
#SBATCH -t 12:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem=100G
#SBATCH --gpus=4
#SBATCH --partition=gpu_h100
#SBATCH -N 1

# Define variables for script parameters (defaults provided)
TRAIN_PATH="/scratch-shared/mschaffelder/data/finetuning/synthetic/Medium/Llama/dolly_train_all_Llama.jsonl"
VAL_PATH="/scratch-shared/mschaffelder/data/finetuning/synthetic/Medium/Llama/dolly_test_Llama.jsonl"
RESPONSE_KEY="response_model"
OUTPUT_DIR="/scratch-shared/mschaffelder/data/ft_models/lora_llama_70b_single_medium"

# Load required modules 
module load 2024 Python/3.12.3-GCCcore-13.3.0
module load 2024 CUDA/12.6.0

# Define the new virtual environment directory
VENV_DIR="/scratch-shared/mschaffelder/venv_finetune_1"

# Activate virtual environment
echo "Activating virtual environment from $VENV_DIR"
source $VENV_DIR/bin/activate

# Install required packages
pip install -r ../requirements.txt 

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

# Download the model if it doesn't exist
MODEL_DIR="/scratch-shared/mschaffelder/hf_cache/Meta-Llama-3.1-70B-Instruct"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Model not found at $MODEL_DIR. Downloading..."
    tune download meta-llama/Meta-Llama-3.1-70B-Instruct --output-dir $MODEL_DIR --ignore-patterns "original/consolidated*"
else
    echo "Model already exists at $MODEL_DIR. Skipping download."
fi

tune run torchtune.py --nproc_per_node 4 --config 70b_lora.yaml