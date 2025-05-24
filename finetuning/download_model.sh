#!/bin/bash
#SBATCH -J download_llama_70b
#SBATCH -t 1:00:00 # 1 hour should be enough, adjust if needed
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 2
#SBATCH --mem=10G
#SBATCH --partition=rome # Or any partition with internet
#SBATCH -N 1

# Load required modules
module load 2024 Python/3.12.3-GCCcore-13.3.0
# No CUDA needed for download

# Activate your new virtual environment (important!)
VENV_DIR="/scratch-shared/mschaffelder/venv_finetune_1"
if [ -d "$VENV_DIR/bin" ]; then
    echo "Activating virtual environment from $VENV_DIR"
    source $VENV_DIR/bin/activate
else
    echo "ERROR: Virtual environment $VENV_DIR not found or not correctly set up."
    echo "Please ensure it was created by the main finetuning script first or create it manually."
    exit 1
fi

# Ensure pip can find the packages in the venv
echo "Python executable: $(which python)"
echo "Pip executable: $(which pip)"

# Install transformers and huggingface_hub if not already in venv_finetune_1
# (They should be from the main requirements.txt, but just in case)
pip install transformers huggingface_hub torch

# Run the download script
huggingface-cli login --token "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"
python download_model.py

echo "Download script finished."