#!/bin/bash

#SBATCH --job-name=manual-consolidate   # Job name
#SBATCH --partition=gpu_h100              # Partition
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH --gpus-per-node=4                 # GPUs per node
#SBATCH --cpus-per-task=64                # CPUs per task
#SBATCH --mem=700G                        # Memory
#SBATCH --time=01:00:00                   # Time limit (1 hour is plenty)

# Load environment modules
module purge
module load 2024
module load CUDA/12.1.1 2>/dev/null || module load CUDA 2>/dev/null || echo "No CUDA module loaded"
module load Python/3.10.4-GCCcore-11.3.0 2>/dev/null || module load Python 2>/dev/null || echo "No Python module loaded"

# Activate virtual environment
VENV_DIR="/scratch-shared/mschaffelder/venv_finetune_llama_70b"
echo "Activating virtual environment: $VENV_DIR"
source $VENV_DIR/bin/activate

# Define paths
CHECKPOINT_TO_CONSOLIDATE="/scratch-shared/mschaffelder/data/ft_models/llama_70b_single_source/checkpoint-318"
CONSOLIDATED_OUTPUT_DIR="/scratch-shared/mschaffelder/data/ft_models/llama_70b_single_source/consolidated_adapter"

# Create output directory if it doesn't exist on the main process
if [ "$SLURM_PROCID" -eq 0 ]; then
    mkdir -p $CONSOLIDATED_OUTPUT_DIR
fi

# We use the same accelerate config as the main training job
# The config is expected to be at ~/.cache/huggingface/accelerate/default_config.yaml

# Set PyTorch CUDA allocation configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use accelerate to run the consolidation script
echo "Starting manual consolidation from checkpoint: $CHECKPOINT_TO_CONSOLIDATE"
accelerate launch manual_consolidate.py \
    --checkpoint_dir $CHECKPOINT_TO_CONSOLIDATE \
    --output_dir $CONSOLIDATED_OUTPUT_DIR \
    --lora_r 16 \
    --lora_alpha 32

echo "Manual consolidation job finished." 