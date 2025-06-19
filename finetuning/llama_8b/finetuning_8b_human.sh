#!/bin/bash
#SBATCH -J finetuning-llama-3.1-8b-human
#SBATCH -t 12:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem=30G
#SBATCH --gpus=2
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

# Install required packages manually before running jobs
pip install -r requirements.txt 

# Define paths for data and output
TRAIN_PATH="/scratch-shared/mschaffelder/data/finetuning/synthetic/Small/Llama/dolly_train_all_Llama.jsonl"
OUTPUT_DIR="/scratch-shared/mschaffelder/data/ft_models/lora_llama_8b_human"

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
    --response_key "response_human" \
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