#!/bin/bash
#SBATCH --job-name=refusalbench_responses_8b_multi
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_h100
#SBATCH --mem=10G
#SBATCH --gpus=1
#SBATCH -N 1


VENV_NAME="venv_exp_2"
VENV_BASE_DIR="/scratch-shared/mschaffelder" 
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
echo "Activating virtual environment: $VENV_DIR"
source $VENV_DIR/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements_exp_2.txt

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

python generate_responses.py \
    --base_model_path "meta-llama/Llama-3.1-8B-Instruct" \
    --lora_model_path "/scratch-shared/mschaffelder/data/ft_models/small/lora_llama_8b_multi_v3" \
    --use_lora \
    --input_file "/scratch-shared/mschaffelder/data/exp_2/inputs/refusalbench_jailbreak.jsonl" \
    --output_file "/scratch-shared/mschaffelder/data/exp_2/outputs/refusalbench_jailbreak_8b_multi_source.jsonl"
