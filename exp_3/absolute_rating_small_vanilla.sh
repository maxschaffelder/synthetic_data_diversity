#!/bin/bash
#SBATCH --job-name=absolute_rating_small_vanilla
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_h100
#SBATCH --mem=10G
#SBATCH --gpus=1
#SBATCH -N 1


VENV_NAME="venv_exp_3"
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
echo "Activating virtual environment: $VENV_DIR"
source $VENV_DIR/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements_exp_3.txt

# Run script
cd $SLURM_SUBMIT_DIR
# Make sure CUDA devices are visible

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Available GPUs: $(nvidia-smi -L)"

huggingface-cli login --token "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"

python absolute_rating.py \
    --base_model_path "meta-llama/Llama-3.1-8B-Instruct" \
    --use_lora False \
    --input_file "/scratch-shared/mschaffelder/Data/exp_3/generated/small/all_summaries_small.jsonl" \
    --output_file "/scratch-shared/mschaffelder/Data/exp_3/absolute_ratings/small_vanilla.jsonl"
