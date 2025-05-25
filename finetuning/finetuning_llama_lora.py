print("importing libraries")
import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# Set environment variable to avoid tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Set CUDA memory allocation strategy for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("done importing libraries")

# GPU Memory Logging Function
def log_gpu_memory(stage="", device_idx=0):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 2)
        print(f"GPU Memory ({stage}): Allocated={allocated:.2f} MB, Reserved={reserved:.2f} MB")

log_gpu_memory("Initial - After Imports")

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA model with LoRA")
    
    # Data paths
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path to training data JSONL file")
    parser.add_argument("--val_path", type=str, required=True,
                        help="Path to validation data JSONL file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the fine-tuned model")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, 
                        default="meta-llama/Llama-3.1-70B-Instruct",
                        help="Model name or path")
    parser.add_argument("--cache_dir", type=str,
                        default="/scratch-shared/mschaffelder/hf_cache",
                        help="Cache directory for models")
    
    # Training parameters
    parser.add_argument("--response_key", type=str, default="response_model",
                        help="Key for response in dataset")
    parser.add_argument("--run_name", type=str, required=True,
                        help="Name for this training run")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    
    # Quantization options
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization")
    parser.add_argument("--use_8bit", action="store_true", default=False,
                        help="Use 8-bit quantization instead of 4-bit")
    
    return parser.parse_args()

args = parse_args()
print(f"Training arguments: {args}")

# 1. Load the tokenizer and model
specific_model_cache_path = args.cache_dir + "/" + args.model_name.replace("/", "_")

print(f"Attempting to load tokenizer from local path: {specific_model_cache_path}")
tokenizer = AutoTokenizer.from_pretrained(specific_model_cache_path, use_fast=False)

print(f"Attempting to load model from local path: {specific_model_cache_path}")

# Configure quantization based on arguments
if args.use_8bit:
    print("Using 8-bit quantization")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
elif args.use_4bit:
    print("Using 4-bit quantization")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        # Ensure compatibility with training
        bnb_4bit_quant_storage=torch.bfloat16,
    )
else:
    print("No quantization")
    quantization_config = None

# Load model on single GPU
model = AutoModelForCausalLM.from_pretrained(
    specific_model_cache_path,
    torch_dtype=torch.bfloat16,
    use_cache=False,
    attn_implementation="sdpa", 
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    device_map={"": 0},  # Force all layers onto GPU 0
    trust_remote_code=True,
)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Ensure tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    model.resize_token_embeddings(len(tokenizer))
print("done loading tokenizer and model")

log_gpu_memory("After Model and Tokenizer Loaded")

# 2. Configure LoRA for parameter-efficient fine-tuning
lora_config = LoraConfig(
    r=args.lora_rank,                         
    lora_alpha=args.lora_alpha,               
    target_modules=["q_proj", "v_proj"],  
    lora_dropout=0.1,            
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
)

print("getting peft model")
model = get_peft_model(model, lora_config)

# Enable training mode
model.train()

# Explicitly enable gradients for all trainable parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        param.grad = None  # Clear any existing gradients
        
print("done getting peft model")
model.print_trainable_parameters()

# Debug: Check which parameters require gradients
print("Parameters requiring gradients:")
trainable_params = 0
total_params = 0
for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
        print(f"  {name}: {param.shape}")

print(f"Trainable parameters: {trainable_params}")
print(f"Total parameters: {total_params}")
print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

# 3. Load and preprocess the dataset
train_path = args.train_path
val_path = args.val_path
RESPONSE_KEY = args.response_key
OUTPUT_DIR = args.output_dir

print("loading datasets")
train_dataset = load_dataset("json", data_files={"train": train_path})
train_dataset = train_dataset["train"].shuffle(seed=42)
val_dataset = load_dataset("json", data_files={"test": val_path})
val_dataset = val_dataset["test"].shuffle(seed=42)

print("done loading datasets")

# System prompt
system_prompt = "You are a helpful assistant."

# Function to filter out examples exceeding max_length tokens
def filter_long(ex):
    combined = system_prompt + " " + ex["instruction"] + " " + ex[RESPONSE_KEY]
    return len(tokenizer(combined, truncation=False)["input_ids"]) <= args.max_length

print("filtering long examples")
train_dataset = train_dataset.filter(filter_long)
val_dataset = val_dataset.filter(filter_long)
print("done filtering long examples")

# Function to tokenize and prepare model inputs
def tokenize_and_format(ex):
    prompt = system_prompt + " " + ex["instruction"]
    prompt_ids = tokenizer(prompt, truncation=True, add_special_tokens=False, max_length=args.max_length, padding=False).input_ids
    response_ids = tokenizer(" " + ex[RESPONSE_KEY], truncation=True, add_special_tokens=False, max_length=args.max_length, padding=False).input_ids
    input_ids = prompt_ids + response_ids + [tokenizer.eos_token_id]
    labels = [-100] * len(prompt_ids) + response_ids + [tokenizer.eos_token_id]
    return {"input_ids": input_ids, "labels": labels}

print("tokenizing and formatting train dataset")
train_dataset = train_dataset.map(tokenize_and_format, remove_columns=["response_human", "index", "model_name", "category", "response_model", "instruction"])
val_dataset = val_dataset.map(tokenize_and_format, remove_columns=["response_human", "index", "model_name", "category", "response_model", "instruction"])

# Data collator
data_collator = DataCollatorWithPadding(tokenizer)

# Training configuration for single GPU
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=args.batch_size,   
    per_device_eval_batch_size=args.batch_size,    
    gradient_accumulation_steps=args.gradient_accumulation_steps,  
    num_train_epochs=args.num_epochs,
    learning_rate=args.learning_rate,
    weight_decay=0.01,
    fp16=False,                     
    bf16=True,
    eval_strategy="steps",
    eval_steps=200,
    logging_steps=50,
    save_steps=400,
    save_total_limit=3,
    report_to="none",
    run_name=args.run_name,
    logging_dir="/scratch-shared/mschaffelder/Data/ft_models/logs",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    dataloader_num_workers=0,  # Disable multiprocessing to save memory
    gradient_checkpointing=True,
    optim="adamw_torch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    dataloader_pin_memory=False,  # Disable pin memory to save RAM
    # Additional optimizations for quantized training
    remove_unused_columns=False,  # Keep all columns to avoid issues
    max_grad_norm=1.0,  # Gradient clipping
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

print("starting training")
log_gpu_memory("Before Trainer.train()")
trainer.train()
print("finished training")
log_gpu_memory("After Trainer.train()")

print("saving model")
trainer.save_model(OUTPUT_DIR)
print(f"model saved to {OUTPUT_DIR}") 

if torch.cuda.is_available():
    max_allocated = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    max_reserved = torch.cuda.max_memory_reserved(0) / (1024 ** 2)
    print(f"Max GPU Memory during run: Max Allocated={max_allocated:.2f} MB, Max Reserved={max_reserved:.2f} MB")
    # Reset peak stats for next potential runs in the same process (if any)
    torch.cuda.reset_peak_memory_stats(0) 