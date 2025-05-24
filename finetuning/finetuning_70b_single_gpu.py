print("importing libraries")
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model

# Set environment variable to avoid tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Set CUDA memory allocation strategy for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("done importing libraries")

# 1. Load the tokenizer and model
model_name = "meta-llama/Llama-3.1-70B-Instruct"
cache_dir_base = "/scratch-shared/mschaffelder/hf_cache"
specific_model_cache_path = cache_dir_base + "/" + model_name.replace("/", "_")

print(f"Attempting to load tokenizer from local path: {specific_model_cache_path}")
tokenizer = AutoTokenizer.from_pretrained(specific_model_cache_path, use_fast=False)

print(f"Attempting to load model from local path: {specific_model_cache_path}")

# More aggressive quantization for single GPU
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

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

# Ensure tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    model.resize_token_embeddings(len(tokenizer))
print("done loading tokenizer and model")

# 2. Configure LoRA for parameter-efficient fine-tuning
lora_config = LoraConfig(
    r=8,                         # Smaller LoRA rank for memory efficiency
    lora_alpha=16,               
    target_modules=["q_proj", "v_proj"],  
    lora_dropout=0.1,            
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
)

print("getting peft model")
model = get_peft_model(model, lora_config)
print("done getting peft model")
model.print_trainable_parameters()

# 3. Load and preprocess the dataset
train_path = "/scratch-shared/mschaffelder/Data/Finetuning/synthetic/Small/Llama/dolly_train_all_Llama.jsonl" 
val_path = "/scratch-shared/mschaffelder/Data/Finetuning/synthetic/Small/Llama/dolly_test_Llama.jsonl"
RESPONSE_KEY = "response_model"
OUTPUT_DIR = "/scratch-shared/mschaffelder/Data/ft_models/lora_llama_70b_single_gpu"

print("loading datasets")
train_dataset = load_dataset("json", data_files={"train": train_path})
train_dataset = train_dataset["train"].shuffle(seed=42)
val_dataset = load_dataset("json", data_files={"test": val_path})
val_dataset = val_dataset["test"].shuffle(seed=42)

print("done loading datasets")

# System prompt
system_prompt = "You are a helpful assistant."

# Function to filter out examples exceeding 1024 tokens (smaller for memory)
def filter_long(ex):
    combined = system_prompt + " " + ex["instruction"] + " " + ex[RESPONSE_KEY]
    return len(tokenizer(combined, truncation=False)["input_ids"]) <= 2048

print("filtering long examples")
train_dataset = train_dataset.filter(filter_long)
val_dataset = val_dataset.filter(filter_long)
print("done filtering long examples")

# Function to tokenize and prepare model inputs
def tokenize_and_format(ex):
    prompt = system_prompt + " " + ex["instruction"]
    prompt_ids = tokenizer(prompt, truncation=True, add_special_tokens=False, max_length=2048, padding=False).input_ids
    response_ids = tokenizer(" " + ex[RESPONSE_KEY], truncation=True, add_special_tokens=False, max_length=1024, padding=False).input_ids
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
    per_device_train_batch_size=1,   
    per_device_eval_batch_size=1,    
    gradient_accumulation_steps=16,  # Larger accumulation for effective batch size
    num_train_epochs=2,
    learning_rate=1e-5,
    weight_decay=0.01,
    fp16=False,                     
    bf16=True,
    eval_strategy="steps",
    eval_steps=200,
    logging_steps=50,
    save_steps=400,
    save_total_limit=3,
    report_to="none",
    run_name="lora_llama_70b_single_gpu",
    logging_dir="/scratch-shared/mschaffelder/Data/ft_models/logs",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    dataloader_num_workers=0,  # Disable multiprocessing to save memory
    gradient_checkpointing=True,
    optim="adamw_torch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    dataloader_pin_memory=False,  # Disable pin memory to save RAM
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
trainer.train()
print("finished training")

print("saving model")
trainer.save_model(OUTPUT_DIR)
print(f"model saved to {OUTPUT_DIR}") 