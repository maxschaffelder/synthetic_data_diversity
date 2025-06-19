#!/bin/bash
#SBATCH -J finetuning-llama-3.1-8b-single
#SBATCH -t 12:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem=30G
#SBATCH --gpus=2
#SBATCH --partition=gpu_h100
#SBATCH -N 1

# Load necessary environment modules
module purge

# Try to load available modules - check what's available on the system
module load 2024

# Check available CUDA modules
module avail CUDA 2>/dev/null || echo "CUDA modules not found, will use conda/pip CUDA"

# Check available Python modules  
module avail Python 2>/dev/null || echo "Python modules not found, will use system Python"

# Try to load common modules if they exist
module load CUDA/12.1.1 2>/dev/null || module load CUDA 2>/dev/null || echo "No CUDA module loaded"
module load Python/3.10.4-GCCcore-11.3.0 2>/dev/null || module load Python 2>/dev/null || echo "No Python module loaded"

VENV_NAME="venv_finetune_llama_8b"
VENV_BASE_DIR="/scratch-shared/mschaffelder" # Or any other persistent shared directory you prefer
VENV_DIR="$VENV_BASE_DIR/$VENV_NAME"

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
# (Optional) Install required packages manually before running jobs
# pip install -r requirements.txt 

# Define paths for data and output
TRAIN_PATH="/scratch-shared/mschaffelder/data/finetuning/synthetic/Small/Llama/dolly_train_all_Llama.jsonl"
OUTPUT_DIR="/scratch-shared/mschaffelder/data/ft_models/lora_llama_8b_single"

# Create output directory and job-specific accelerate config
mkdir -p $OUTPUT_DIR
ACCELERATE_CONFIG_FILE="$OUTPUT_DIR/accelerate_config.yaml"
cat > "$ACCELERATE_CONFIG_FILE" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: DDP
num_processes: 2
machine_rank: 0
main_training_function: main
mixed_precision: bf16
use_cpu: false
EOF

# Login to Hugging Face using environment variable
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN"
fi

# Run script
accelerate launch --config_file "$ACCELERATE_CONFIG_FILE" finetuning_8b.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --train_path "$TRAIN_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --response_key "response_model" \
    --validation_split_percentage 5 \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lora_r 16 \
    --lora_alpha 32 \
    --max_seq_length 1024 \
    --logging_steps 10 \
    --eval_steps 20 \
    --save_steps 100 \
    --early_stopping_patience 5