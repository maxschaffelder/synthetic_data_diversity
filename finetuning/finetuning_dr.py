print("importing libraries")
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
print("done importing libraries")
# (Optional) initialize Weights & Biases for experiment tracking
# You can set WANDB_PROJECT and login separately as needed.
#import wandb
#wandb.init(project="llama3_lora_finetuning", name="llama3-8b-run")

# Alternatively, one could use TRL's SFTTrainer for supervised fine-tuning:contentReference[oaicite:0]{index=0}.

# 1. Load the tokenizer and model
# Use the LLaMA 3.1 8B Instruct model from Hugging Face (requires access)
print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", use_fast=False)
print("loading model")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Ensure tokenizer has a padding token (LLaMA may not have one by default)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    model.resize_token_embeddings(len(tokenizer))
print("done loading tokenizer and model")

# 2. Configure LoRA for parameter-efficient fine-tuning
# Using Hugging Face PEFT LoRA (low-rank adaptation) example:contentReference[oaicite:1]{index=1}; 
# target LLaMA's attention and MLP layers as recommended:contentReference[oaicite:2]{index=2}.
lora_config = LoraConfig(
    r=32,                         # LoRA rank
    lora_alpha=16,                # LoRA scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # modules to apply LoRA
    lora_dropout=0.05,            # dropout for LoRA layers
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

print("getting peft model")
model = get_peft_model(model, lora_config)
model.to("cuda")
print("done getting peft model")
# Optional: verify the number of trainable parameters (should be much smaller than total)
model.print_trainable_parameters()

# 3. Load and preprocess the dataset
# Data format: JSONL with 'instruction' and 'response_model' fields per line.
data_path = "/scratch-shared/mschaffelder/Data/Finetuning/synthetic/Small/Llama/dolly_train_all_Llama_formatted.jsonl" # TODO: look into validation se

print("loading dataset")
dataset = load_dataset("json", data_files={"train": data_path})
print("done loading dataset")


# Split into training and validation (90% train, 10% eval)
#dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

# System prompt to prepend to each instruction
system_prompt = "You are a helpful assistant."

# Function to filter out examples exceeding 2048 tokens (prompt + response)
def filter_long(ex):
    combined = system_prompt + " " + ex["instruction"] + " " + ex["response_model"]
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
    prompt_ids = tokenizer(prompt, truncation=True, add_special_tokens=False).input_ids
    response_ids = tokenizer(" " + ex["response_model"], truncation=True, add_special_tokens=False).input_ids
    # Combine and add end-of-sequence token
    input_ids = prompt_ids + response_ids + [tokenizer.eos_token_id]
    # Labels: mask prompt part with -100 so loss is only computed on the response
    labels = [-100] * len(prompt_ids) + response_ids + [tokenizer.eos_token_id]
    return {"input_ids": input_ids, "labels": labels}

# Apply tokenization to the datasets
train_dataset = train_dataset.map(tokenize_and_format, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(tokenize_and_format, remove_columns=val_dataset.column_names)

# 4. Data collator: pad sequences dynamically
data_collator = DataCollatorWithPadding(tokenizer)

# 5. Training configuration: use mixed precision (AMP) on A100/H100 (fp16)
training_args = TrainingArguments(
    output_dir="/scratch-shared/mschaffelder/Data/ft_models/llama3_lora_output",
    per_device_train_batch_size=2,    # adjust to fit GPU memory
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,    # for an effective larger batch size
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=False,                        # enable mixed-precision training:contentReference[oaicite:3]{index=3}
    bf16=True,
    evaluation_strategy="steps",
    eval_steps=200,
    logging_steps=50,
    save_steps=500,
    save_total_limit=3,
    report_to="wandb",                # log metrics to Weights & Biases:contentReference[oaicite:4]{index=4}
    run_name="llama3-8b-lora",
    logging_dir="/scratch-shared/mschaffelder/Data/ft_models/logs"
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
# 6. Start training
trainer.train()
print("finished training")