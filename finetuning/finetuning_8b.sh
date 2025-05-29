#!/bin/bash
#SBATCH -J finetuning-llama-3.1-8b-instant
#SBATCH -t 12:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem=30G
#SBATCH --gpus=2
#SBATCH --partition=gpu_h100
#SBATCH -N 1

# Define variables for script parameters (defaults provided)
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
TRAIN_PATH="/scratch-shared/mschaffelder/data/finetuning/synthetic/Small/Other/dolly_train_all_multi.jsonl"
VAL_PATH="/scratch-shared/mschaffelder/data/finetuning/synthetic/Small/Other/dolly_test_multi.jsonl"
RESPONSE_KEY="response_model"
OUTPUT_DIR="/scratch-shared/mschaffelder/data/ft_models/lora_llama_8b_multi"

# Load required modules 
module load 2024 Python/3.12.3-GCCcore-13.3.0
module load 2024 CUDA/12.6.0

# Define the new virtual environment directory
VENV_DIR="/scratch-shared/mschaffelder/venv_finetune_1"

# Activate virtual environment
echo "Activating virtual environment from $VENV_DIR"
source $VENV_DIR/bin/activate


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

#python finetuning_dr.py
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 finetuning_8b.py \
    --model_name "$MODEL_NAME" \
    --train_path "$TRAIN_PATH" \
    --val_path "$VAL_PATH" \
    --response_key "$RESPONSE_KEY" \
    --output_dir "$OUTPUT_DIR"