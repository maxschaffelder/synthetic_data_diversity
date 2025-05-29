#!/bin/bash
#SBATCH -J llama-70b-medium-4bit
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

# Run the parameterized training script for medium dataset
python finetuning_llama_lora.py \
    --train_path "/scratch-shared/mschaffelder/Data/Finetuning/synthetic/Medium/Llama/dolly_train_all_Llama.jsonl" \
    --val_path "/scratch-shared/mschaffelder/Data/Finetuning/synthetic/Medium/Llama/dolly_test_Llama.jsonl" \
    --output_dir "/scratch-shared/mschaffelder/Data/ft_models/lora_llama_70b_single_medium_4bit" \
    --run_name "llama_70b_medium_4bit" \
    --model_name "meta-llama/Llama-3.1-70B-Instruct" \
    --max_length 2048 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --num_epochs 2 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --use_4bit 