#!/bin/bash
#SBATCH -J exp_1_perplexity_calculation
#SBATCH -t 12:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem=50G
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

VENV_NAME="venv_exp_1"
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



# Define the model to use for perplexity calculation
MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct" 


# Define base path for data
LOG_DIR="/scratch-shared/mschaffelder/logs"
mkdir -p ${LOG_DIR} # Ensure log directory exists




# Get the task ID for array jobs
INPUT_PATH="/scratch-shared/mschaffelder/data/exp_1/medium/outputs/output_llama_70b_multi_source_large.jsonl"
OUTPUT_PATH="/scratch-shared/mschaffelder/data/exp_1/medium/perplexity/ppl_lora_llama_70b_multi_large.jsonl"



echo "Starting perplexity calculation for: ${INPUT_PATH}"
echo "Output will be saved to: ${OUTPUT_PATH}"
echo "Model: ${MODEL_NAME}"

# Run the perplexity calculation script
python ../perplexity.py \
    --model_name "${MODEL_NAME}" \
    --input_path "${INPUT_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --response_key "response_model"

echo "Finished perplexity calculation for: ${INPUT_PATH}" 