#!/bin/bash

#SBATCH --job-name=consolidate-llama70b   # Job name
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

# Define the checkpoint path to resume from.
CHECKPOINT_TO_RESUME="/scratch-shared/mschaffelder/data/ft_models/llama_70b_single_source/checkpoint-318"
OUTPUT_DIR="/scratch-shared/mschaffelder/data/ft_models/llama_70b_single_source" # Same output directory

# Validate that the checkpoint directory exists
if [ ! -d "$CHECKPOINT_TO_RESUME" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_TO_RESUME"
    exit 1
fi

# We use the same accelerate config as the main training job
# The config is expected to be at ~/.cache/huggingface/accelerate/default_config.yaml

# Set PyTorch CUDA allocation configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Execute the script to resume and save
echo "Resuming from checkpoint: $CHECKPOINT_TO_RESUME"
accelerate launch finetune_llama.py \
    --train_file "/scratch-shared/mschaffelder/data/finetuning/synthetic/Medium/Llama/dolly_train_all_Llama.jsonl" \
    --validation_file "/scratch-shared/mschaffelder/data/finetuning/synthetic/Medium/Llama/dolly_test_Llama.jsonl" \
    --output_dir $OUTPUT_DIR \
    --resume_from_checkpoint $CHECKPOINT_TO_RESUME \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --lora_r 16 \
    --lora_alpha 32 \
    --max_seq_length 1024 \
    --system_prompt "You are a helpful assistant." \
    --response_key "response_model"

echo "Consolidation job completed." 