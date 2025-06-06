#!/bin/bash
#SBATCH -J finetune_70b_single
#SBATCH -t 12:00:00
#SBATCH --partition=gpu_h100
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8        
#SBATCH --mem=80G
#SBATCH --gpus=4
#SBATCH -N 1

# Load required modules 
module load 2024 Python/3.12.3-GCCcore-13.3.0
module load 2024 CUDA/12.6.0

# Define the new virtual environment directory
VENV_DIR="/scratch-shared/mschaffelder/venv_finetune_1"

# Activate virtual environment
echo "Activating virtual environment from $VENV_DIR"
source $VENV_DIR/bin/activate

# Install dependencies
pip install -r ../requirements.txt

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

# Run training with torchrun on 4 GPUs
torchrun --nproc_per_node=4 finetune_70b.py \
    --model_name meta-llama/Llama-3.1-70B-Instruct \
    --train_file /scratch-shared/mschaffelder/data/finetuning/synthetic/Medium/Llama/dolly_train_all_Llama.jsonl \
    --val_file /scratch-shared/mschaffelder/data/finetuning/synthetic/Medium/Llama/dolly_test_Llama.jsonl \
    --response_key response_model \
    --output_dir /scratch-shared/mschaffelder/data/ft_models/lora_llama_70b_single_medium
