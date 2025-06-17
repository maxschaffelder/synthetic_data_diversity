#!/bin/bash

#SBATCH --job-name=llama70b-single-source-finetune   # Assign a descriptive name to the Slurm job
#SBATCH --partition=gpu_h100                 # Specify the target partition: H100 GPU partition on Snellius [8]
#SBATCH --nodes=1                            # Request a single compute node. A full gpu_h100 node has 4 H100 GPUs.[8]
#SBATCH --ntasks-per-node=1                  # Request one task per node. For accelerate launch, typically one main process orchestrates distributed training.
#SBATCH --gpus-per-node=4                    # Explicitly request all 4 H100 GPUs available on the allocated node.[8]
#SBATCH --cpus-per-task=64                   # Request all 64 CPU cores available on an H100 node.[8] This ensures ample CPU resources for data loading, preprocessing, and other CPU-bound operations.
#SBATCH --mem=700G                           # Request a substantial amount of host memory, close to the total 720 GiB available per H100 node.[8] This is a proactive measure to prevent host memory issues.
#SBATCH --time=24:00:00                      # Set the maximum wall clock time for the job (e.g., 24 hours). The gpu_h100 partition allows up to 120 hours (5 days).[8]

# Load necessary environment modules
module purge                                                        
module load CUDA/12.1.1                      # Load the CUDA toolkit version compatible with PyTorch and H100 GPUs

VENV_NAME="venv_finetune_llama_70b"
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
pip install -r requirements.txt

# Define paths for data and output (adjust these as per your setup)
TRAIN_FILE="/scratch-shared/mschaffelder/data/finetuning/synthetic/Medium/Llama/dolly_train_all_Llama.jsonl" # Path to your training data
VALIDATION_FILE="/scratch-shared/mschaffelder/data/finetuning/synthetic/Medium/Llama/dolly_test_Llama.jsonl" # Path to your validation data
OUTPUT_DIR="/scratch-shared/mschaffelder/data/ft_models/llama_70b_single_source" # Output directory for model and logs

# Validate that data files exist
if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: Training file not found: $TRAIN_FILE"
    exit 1
fi

if [ ! -f "$VALIDATION_FILE" ]; then
    echo "Error: Validation file not found: $VALIDATION_FILE"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

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

# Create accelerate config if it doesn't exist
ACCELERATE_CONFIG_FILE="$HOME/.cache/huggingface/accelerate/default_config.yaml"
if [ ! -f "$ACCELERATE_CONFIG_FILE" ]; then
    echo "Creating accelerate config..."
    mkdir -p "$(dirname "$ACCELERATE_CONFIG_FILE")"
    cat > "$ACCELERATE_CONFIG_FILE" << EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
    echo "Accelerate config created at $ACCELERATE_CONFIG_FILE"
fi

# Execute the Python fine-tuning script using accelerate launch
# --num_processes=4 corresponds to the 4 requested GPUs, launching one process per GPU.
# For single-node, multi-GPU setups, accelerate typically handles main_process_ip and main_process_port automatically.
accelerate launch \
    --num_processes=4 \
    --mixed_precision=bf16 \
    finetune_llama.py \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --lora_r 16 \
    --lora_alpha 32 \
    --max_seq_length 1024 \
    --system_prompt "You are a helpful assistant."

echo "Fine-tuning job completed."