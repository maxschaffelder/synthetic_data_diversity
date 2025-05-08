import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# 1. Configuration
MODEL_NAME = "gpt2"  # Much smaller model (124M parameters) for local testing
DATASET_NAME = "/home/mschaffeld/data/maxstorage/Data/Finetuning/Augmented/Small/Llama/dolly_train_all_Llama.jsonl"
#TEST_DATASET_NAME = "/Users/maxschaffelder/Desktop/Thesis/Data/Finetuning/Augmented/Small/Llama/dolly_test_Llama.jsonl"
TEST_DATASET_NAME = "/home/mschaffeld/data/maxstorage/Data/Finetuning/Augmented/Small/Llama/dolly_test_Llama.jsonl"
#OUTPUT_DIR = "/Users/maxschaffelder/Desktop/Thesis/Data/ft_models/lora_gpt2_debug"
OUTPUT_DIR = "/home/mschaffeld/data/maxstorage/Data/ft_models"

# LoRA hyperparameters
LORA_R = 4  # Reduced from 8
LORA_ALPHA = 16  # Reduced from 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["c_attn", "c_proj"]  # GPT-2 attention modules

# Training hyperparameters - reduced for local testing
BATCH_SIZE = 32  # Reduced from 128
MICRO_BATCH_SIZE = 4  # Reduced from 16
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 1  # Reduced from 2.5 for faster testing
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 10  # Reduced from 100

# Debug mode flag
DEBUG_MODE = True

# 2. Load tokenizer and model
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# set padding side and token
tokenizer.padding_side = "right"
tokenizer.truncation_side = "right"
# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model loaded successfully!")

# 3. Prepare model for LoRA
print("Preparing LoRA configuration...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("LoRA modules added successfully!")

# 4. Load and preprocess dataset
print("Loading dataset...")
raw_dataset = load_dataset("json", data_files={
    "train": DATASET_NAME,
    "validation": TEST_DATASET_NAME
})
print(f"Dataset loaded. Train size: {len(raw_dataset['train'])}, Validation size: {len(raw_dataset['validation'])}")

# system prompt for instruct mode
SYSTEM_PROMPT = "You are a helpful assistant."

# simple tokenize and preprocess
max_length = 2048

def preprocess_fn(examples):
    # combine system prompt, instruction, and model answer (fallback to human answer)
    # Find the first key that starts with "response_" but is not "response_human"
    response_keys = [key for key in examples.keys() if key.startswith("response_") and key != "response_human"]
    answers = examples.get(response_keys[0], [])

    # Print the first few keys from dataset to debug
    if len(examples) > 0:
        print(f"Available keys in dataset: {list(examples.keys())}")
        if len(examples["instruction"]) > 0:
            print(f"Sample instruction: {examples['instruction'][0][:50]}...")
            print(f"Sample answer: {answers[0][:50]}..." if answers else "No answers found")

    prompts = [
        f"[INST] <<SYS>> {SYSTEM_PROMPT} <</SYS>>\n\n{ins} [/INST]\n\n{resp}"
        for ins, resp in zip(examples["instruction"], answers)
    ]
    
    # First, tokenize without truncation to check lengths
    lengths = [len(tokenizer.encode(prompt)) for prompt in prompts]
    
    # Create mask for samples not exceeding max_length
    valid_mask = [length <= max_length for length in lengths]
    
    # Filter prompts
    filtered_prompts = [prompt for prompt, is_valid in zip(prompts, valid_mask) if is_valid]
    
    # If all samples were filtered out, return empty dict with same structure
    if not filtered_prompts:
        print("Warning: All samples in this batch exceeded max_length and were filtered out")
        return {"input_ids": [], "attention_mask": [], "labels": []}
    
    # Tokenize valid prompts with padding
    tokenized = tokenizer(
        filtered_prompts,
        truncation=False,  # No truncation needed as we filtered
        max_length=max_length,
        padding="max_length"
    )
    
    # Print stats about filtered samples
    filtered_count = len(prompts) - len(filtered_prompts)
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} samples exceeding {max_length} tokens")
    
    # Debug tokenization of [/INST]
    inst_end_str = "[/INST]"
    inst_end_ids = tokenizer.encode(inst_end_str, add_special_tokens=False)
    print(f"[DEBUG] [/INST] token ids: {inst_end_ids}")
    
    # Create labels with -100 for instruction part
    labels = []
    skipped_count = 0
    valid_samples_mask = []  # Track which samples to keep
    
    for i, input_ids in enumerate(tokenized["input_ids"]):
        # Convert to string to find [/INST] substring position
        text = tokenizer.decode(input_ids)
        inst_pos = text.find("[/INST]")
        
        if inst_pos == -1:
            skipped_count += 1
            if skipped_count <= 5:  # Limit debug output
                print(f"Warning: Could not find [/INST] token in sample {i}, skipping this sample")
            valid_samples_mask.append(False)  # Mark this sample to be filtered out
            labels.append([0])  # Placeholder, will be filtered out
        else:
            # Decode up to the [/INST] position + length
            end_pos = inst_pos + len("[/INST]")
            prefix_text = text[:end_pos]
            # Re-encode just this part to get accurate token count
            prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
            response_start_idx = len(prefix_tokens)
            
            # Set labels to -100 for instruction part, actual token IDs for response part
            sample_labels = [-100] * response_start_idx + input_ids[response_start_idx:]
            labels.append(sample_labels)
            valid_samples_mask.append(True)  # Mark this sample to be kept
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} out of {len(tokenized['input_ids'])} samples due to missing [/INST] token")
    
    # Filter out samples where [/INST] wasn't found
    if not all(valid_samples_mask):
        # Keep only samples that have [/INST]
        for key in tokenized.keys():
            tokenized[key] = [item for item, valid in zip(tokenized[key], valid_samples_mask) if valid]
    
    tokenized["labels"] = labels
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
    eval_strategy="steps" if eval_dataset else "no",
    eval_steps=500,
    load_best_model_at_end=True if eval_dataset else False,
    report_to="tensorboard",  # Enable TensorBoard logging
    logging_dir=os.path.join(OUTPUT_DIR, "logs")  # Directory for TensorBoard logs
)

# Use the proper data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # we're doing causal LM, not masked LM
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 6. Start training
print("Starting training...")
trainer.train()

# 7. Save adapter-only weights
model.save_pretrained(OUTPUT_DIR)
print(f"LoRA-adapted model saved to {OUTPUT_DIR}")
