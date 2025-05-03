import os
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from datasets import load_dataset

# This script is written by ChatGPT

# 1. Configuration
MODEL_NAME = "meta-llama/Llama-3-8b-instruct"  # HF repo for Llama 3.1 8B
DATASET_NAME = "../../Data/Finetuning/Augmented/Small/Llama/dolly_train_all_Llama.jsonl"  
OUTPUT_DIR = "../../Data/ft_models/lora_llama3"

# LoRA hyperparameters
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # typically attention proj modules

# Training hyperparameters
BATCH_SIZE = 8
MICRO_BATCH_SIZE = 1  # actual per-GPU batch
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 3
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100

# 2. Load tokenizer and model
print("Loading tokenizer and model...")
tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
# set padding side and token
tokenizer.padding_side = "right"
tokenizer.truncation_side = "right"

# Load model in 8-bit to save memory
model = LlamaForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    device_map="auto"
)

# 3. Prepare model for int8 training and apply LoRA
model = prepare_model_for_int8_training(model)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("LoRA modules added. Trainable params: ", model.get_trainable_params())

# 4. Load and preprocess dataset
raw_dataset = load_dataset("json", data_files=DATASET_NAME)

# system prompt for instruct mode
SYSTEM_PROMPT = "You are a helpful assistant."

# simple tokenize and preprocess
max_length = 2048

def preprocess_fn(examples):
    # combine system prompt, instruction, and model answer (fallback to human answer)
    answers = examples.get("response_llama-3.1-8b-instant", [])
    prompts = [
        f"[INST] <<SYS>> {SYSTEM_PROMPT} <</SYS>>\n\n{ins} [/INST]\n\n{resp}"
        for ins, resp in zip(examples["instruction"], answers)
    ]
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = raw_dataset.map(
    preprocess_fn,
    batched=True,
    remove_columns=raw_dataset["train"].column_names
)

train_dataset = dataset["train"]
eval_dataset = dataset.get("validation", None)

# 5. Setup Trainer
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    per_device_eval_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    fp16=True,
    logging_steps=50,
    save_total_limit=3,
    save_steps=500,
    evaluation_strategy="steps" if eval_dataset else "no",
    eval_steps=500,
    load_best_model_at_end=True if eval_dataset else False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=lambda data: {
        'input_ids': torch.stack([f['input_ids'] for f in data]),
        'attention_mask': torch.stack([f['attention_mask'] for f in data]),
        'labels': torch.stack([f['labels'] for f in data])
    }
)

# 6. Start training
print("Starting training...")
trainer.train()

# 7. Save adapter-only weights
model.save_pretrained(OUTPUT_DIR)
print(f"LoRA-adapted model saved to {OUTPUT_DIR}")
