print("importing libraries")
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
import flash_attn

# Set environment variable to avoid tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("done importing libraries")
# (Optional) initialize Weights & Biases for experiment tracking
# You can set WANDB_PROJECT and login separately as needed.
#import wandb
#wandb.init(project="llama3_lora_finetuning", name="llama3-8b-run")

# Alternatively, one could use TRL's SFTTrainer for supervised fine-tuning:contentReference[oaicite:0]{index=0}.

# 1. Load the tokenizer and model
# Use the LLaMA 3.1 8B Instruct model from Hugging Face (requires access)

model_name = "meta-llama/Llama-3.1-70B-Instruct"

print("flash_attn.__version__: ", flash_attn.__version__)

print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
print("loading model")

# Remove device_map="auto" as it can cause issues with distributed training
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    use_cache=False,
    attn_implementation="flash_attention_2"
)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Ensure tokenizer has a padding token (LLaMA may not have one by default)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    model.resize_token_embeddings(len(tokenizer))
print("done loading tokenizer and model")

# 2. Configure LoRA for parameter-efficient fine-tuning
# Using Hugging Face PEFT LoRA (low-rank adaptation) example:contentReference[oaicite:1]{index=1}; 
# target LLaMA's attention and MLP layers as recommended:contentReference[oaicite:2]{index=2}.
lora_config = LoraConfig(
    r=16,                         # LoRA rank
    lora_alpha=16,                # LoRA scaling factor (same as rank for now)
    target_modules=["q_proj", "v_proj"],  # Apply Lora only to the attention layers
    lora_dropout=0.1,            # dropout for LoRA layers
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

print("getting peft model")
model = get_peft_model(model, lora_config)
print("done getting peft model")
# Verify the number of trainable parameters (should be much smaller than total)
model.print_trainable_parameters()

# 3. Load and preprocess the dataset
#train_path = "/scratch-shared/mschaffelder/Data/Finetuning/synthetic/Small/Llama/dolly_train_all_Llama.jsonl" 
#val_path = "/scratch-shared/mschaffelder/Data/Finetuning/synthetic/Small/Llama/dolly_test_Llama.jsonl"

train_path = "/scratch-shared/mschaffelder/Data/Finetuning/synthetic/Small/Other/dolly_train_all_multi.jsonl" 
val_path = "/scratch-shared/mschaffelder/Data/Finetuning/synthetic/Small/Other/dolly_test_multi.jsonl"
RESPONSE_KEY = "response_human" # CHANGE THIS TO "response_model" FOR MODEL RESPONSE
OUTPUT_DIR = "/scratch-shared/mschaffelder/Data/ft_models/lora_llama_70b_single"

print("loading datasets")
train_dataset = load_dataset("json", data_files={"train": train_path})
train_dataset = train_dataset["train"].shuffle(seed=42)
val_dataset = load_dataset("json", data_files={"test": val_path}) # TODO: can I use this data for validation?
val_dataset = val_dataset["test"].shuffle(seed=42)


print("done loading datasets")

print("train dataset: ", train_dataset)
print("val dataset: ", val_dataset)




# System prompt to prepend to each instruction
system_prompt = "You are a helpful assistant."

# Function to filter out examples exceeding 2048 tokens (prompt + response)
def filter_long(ex):
    combined = system_prompt + " " + ex["instruction"] + " " + ex[RESPONSE_KEY]
    return len(tokenizer(combined, truncation=False)["input_ids"]) <= 2048

print("filtering long examples")
train_dataset = train_dataset.filter(filter_long)
val_dataset   = val_dataset.filter(filter_long)
print("done filtering long examples")

# Function to tokenize and prepare model inputs
def tokenize_and_format(ex):
    # Combine system prompt and instruction as input prompt
    prompt = system_prompt + " " + ex["instruction"]
    # Tokenize prompt and response separately
    prompt_ids = tokenizer(prompt, truncation=True, add_special_tokens=False, max_length=2048, padding=False).input_ids
    response_ids = tokenizer(" " + ex[RESPONSE_KEY], truncation=True, add_special_tokens=False, max_length=2048, padding=False).input_ids
    # Combine and add end-of-sequence token
    input_ids = prompt_ids + response_ids + [tokenizer.eos_token_id]
    # Labels: mask prompt part with -100 so loss is only computed on the response
    labels = [-100] * len(prompt_ids) + response_ids + [tokenizer.eos_token_id]
    return {"input_ids": input_ids, "labels": labels}

print("tokenizing and formatting train dataset")

# Apply tokenization to the datasets
train_dataset = train_dataset.map(tokenize_and_format, remove_columns=["response_human", "index", "model_name", "category", "response_model", "instruction", "index"])#, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(tokenize_and_format, remove_columns=["response_human", "index", "model_name", "category", "response_model", "instruction", "index"])#, remove_columns=val_dataset.column_names)

# 4. Data collator: pad sequences dynamically
data_collator = DataCollatorWithPadding(tokenizer)

# 5. Training configuration: use mixed precision (AMP) on A100/H100 (fp16)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,   
    per_device_eval_batch_size=1,    
    gradient_accumulation_steps=8,   
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
    #report_to="wandb",             
    report_to="none",
    run_name="lora_llama_70b_single",
    logging_dir="/scratch-shared/mschaffelder/Data/ft_models/logs",
    # Learning rate scheduler settings
    lr_scheduler_type="cosine",     # Use cosine scheduler for smooth decay
    warmup_ratio=0.1,               # Warm up for 10% of training steps
    # DDP settings
    ddp_find_unused_parameters=False,
    dataloader_num_workers=2,
    gradient_checkpointing=True,
    optim="adamw_torch",  # Use PyTorch's AdamW optimizer which is more memory efficient
    # Make DDP more stable
    ddp_backend="nccl",
    local_rank=-1,  # Let torch.distributed handle this
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=tokenizer
)

print("starting training")
# 6. Start training
trainer.train()
print("finished training")

# If needed, save the model after training
print("saving model")
trainer.save_model(OUTPUT_DIR)
print(f"model saved to {OUTPUT_DIR}")

# Clean up distributed process group to avoid resource leaks
if torch.distributed.is_initialized():
    print("Cleaning up distributed process group")
    torch.distributed.destroy_process_group()
    print("Process group destroyed successfully")