#!/bin/bash

#SBATCH --job-name=llama70b-human-source-finetune   # Assign a descriptive name to the Slurm job
#SBATCH --partition=gpu_h100                 # Specify the target partition: H100 GPU partition on Snellius [8]
#SBATCH --nodes=1                            # Request a single compute node. A full gpu_h100 node has 4 H100 GPUs.[8]
#SBATCH --ntasks-per-node=1                  # Request one task per node. For accelerate launch, typically one main process orchestrates distributed training.
#SBATCH --gpus-per-node=4                    # Explicitly request all 4 H100 GPUs available on the allocated node.[8]
#SBATCH --cpus-per-task=64                   # Request all 64 CPU cores available on an H100 node.[8] This ensures ample CPU resources for data loading, preprocessing, and other CPU-bound operations.
#SBATCH --mem=700G                           # Request a substantial amount of host memory, close to the total 720 GiB available per H100 node.[8] This is a proactive measure to prevent host memory issues.
#SBATCH --time=24:00:00                      # Set the maximum wall clock time for the job (e.g., 24 hours). The gpu_h100 partition allows up to 120 hours (5 days).[8]

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

VENV_NAME="venv_finetune_llama_70b"
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

# Install required packages (Assuming this is done once manually before submitting jobs)
# pip install -r requirements.txt

# Define paths for data and output (adjust these as per your setup)
TRAIN_FILE="/scratch-shared/mschaffelder/data/finetuning/dolly/dolly_train_all.jsonl" # Path to your training data
OUTPUT_DIR="/scratch-shared/mschaffelder/data/ft_models/llama_70b_human_source_medium" # Output directory for model and logs

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Create a job-specific accelerate config in the output directory
ACCELERATE_CONFIG_FILE="$OUTPUT_DIR/accelerate_config.yaml"
echo "Creating job-specific accelerate config at: $ACCELERATE_CONFIG_FILE"
cat > "$ACCELERATE_CONFIG_FILE" << EOF
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_offload_params: false
  fsdp_sharding_strategy: 1 # 1 for FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: 'LlamaDecoderLayer'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
use_cpu: false
EOF

# Validate that data files exist
if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: Training file not found: $TRAIN_FILE"
    exit 1
fi

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

# Set PyTorch CUDA allocation configuration to prevent fragmentation, as suggested in the error log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Execute the Python fine-tuning script using accelerate launch
# We rely on the config file for settings and let SLURM manage CUDA devices.
echo "CUDA_VISIBLE_DEVICES as set by SLURM: $CUDA_VISIBLE_DEVICES"
accelerate launch --config_file "$ACCELERATE_CONFIG_FILE" ../finetune_llama.py \
    --train_file $TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --validation_split_percentage 5 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 20 \
    --early_stopping_patience 5 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --lora_r 16 \
    --lora_alpha 32 \
    --max_seq_length 1024 \
    --system_prompt "You are a helpful assistant." \
    --response_key "response_human"

echo "Fine-tuning job completed."