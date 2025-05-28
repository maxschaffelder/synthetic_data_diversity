#!/bin/bash
#SBATCH -J data-generation-llama-70b
#SBATCH -t 10:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem=60G
#SBATCH --gpus=2
#SBATCH --partition=gpu_h100
#SBATCH -N 1

# Load required modules 
module load 2024
# module load Python/3.10.4-GCCcore-11.3.0

# Activate virtual environment
source /scratch-shared/mschaffelder/venv/bin/activate

# Install required packages
# pip install -r requirements.txt

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

python data_gen_local.py