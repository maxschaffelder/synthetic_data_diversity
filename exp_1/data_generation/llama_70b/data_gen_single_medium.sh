#!/bin/bash
#SBATCH -J data-generation-llama-70b-finetuned-single-medium
#SBATCH -t 12:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem=50G
#SBATCH --gpus=2
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
#pip install --upgrade pip
#pip install -r requirements.txt

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

# --- Script Configuration ---
LORA_MODEL_PATH="/scratch-shared/mschaffelder/data/ft_models/medium/llama_70b_single_source_medium"
TEST_DATA_PATH="/scratch-shared/mschaffelder/data/finetuning/dolly/dolly_test.jsonl"
OUTPUT_DIR="/scratch-shared/mschaffelder/data/exp_1/medium/outputs"
# --- End Script Configuration ---

python ../data_gen.py \
    --base_model_path "meta-llama/Llama-3.1-70B-Instruct" \
    --lora_model_path "$LORA_MODEL_PATH" \
    --test_data_path "$TEST_DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 8
