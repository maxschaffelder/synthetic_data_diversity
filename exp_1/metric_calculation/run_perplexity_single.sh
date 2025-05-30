#!/bin/bash
#SBATCH -J exp_1_perplexity_calculation
#SBATCH -t 02:00:00
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem=30G
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH -N 1


# Load required modules 
module load 2024 Python/3.12.3-GCCcore-13.3.0
module load 2024 CUDA/12.6.0      

# Define the new virtual environment directory
VENV_DIR="/scratch-shared/mschaffelder/venv_exp_1"

# Activate virtual environment
echo "Activating virtual environment from $VENV_DIR"
source $VENV_DIR/bin/activate



# Define the model to use for perplexity calculation
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" # Changed from Llama-8b-instruct as it's not on HF


# Define base path for data
BASE_DATA_PATH="/scratch-shared/mschaffelder/Data/exp_1/data/small"
LOG_DIR="/scratch-shared/mschaffelder/logs"
mkdir -p ${LOG_DIR} # Ensure log directory exists




# Get the task ID for array jobs
INPUT_FILE_NAME="generation_results_lora_llama_8b_single_v8"
INPUT_PATH="${BASE_DATA_PATH}/${INPUT_FILE_NAME}.jsonl"
OUTPUT_PATH="${BASE_DATA_PATH}/${INPUT_FILE_NAME}_ppl.jsonl"



echo "Starting perplexity calculation for: ${INPUT_PATH}"
echo "Output will be saved to: ${OUTPUT_PATH}"
echo "Model: ${MODEL_NAME}"

# Run the perplexity calculation script
python perplexity.py \
    --model_name "${MODEL_NAME}" \
    --input_path "${INPUT_PATH}" \
    --output_path "${OUTPUT_PATH}" 

echo "Finished perplexity calculation for: ${INPUT_PATH}" 