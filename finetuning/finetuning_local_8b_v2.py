import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import default_data_collator

# 1. Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_NAME = "../../Data/Finetuning/Augmented/Small/Llama/dolly_train_all_Llama.jsonl"
TEST_DATASET_NAME = "/scratch-shared/mschaffelder/Data/Finetuning/Augmented/Small/Llama/dolly_test_Llama.jsonl"
OUTPUT_DIR = "../../Data/ft_models/lora_llama_8b_single"

# LoRA hyperparameters
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# Training hyperparameters
BATCH_SIZE = 128
MICRO_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 2.5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
SEED = 42
MAX_LENGTH = 2048
SYSTEM_PROMPT = "You are a helpful assistant."

# 2. Set seed for reproducibility
set_seed(SEED)

# 3. Load tokenizer and model
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "right"
tokenizer.truncation_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4. Load model with 8-bit quantization and prepare for k-bit training
#quantization_config = BitsAndBytesConfig(
#    load_in_8bit=True,
#    llm_int8_threshold=6.0,
#    llm_int8_has_fp16_weight=False
#)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    #quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map="auto"
)
#model = prepare_model_for_kbit_training(model)

# 5. Apply LoRA adapters
guru_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, guru_config)
print("LoRA modules added.")

# 6. Load and preprocess dataset
raw_dataset = load_dataset("json", data_files={"train": DATASET_NAME, "validation": TEST_DATASET_NAME})

# Precompute token ids for inst end marker
INST_END_IDS = tokenizer.encode("[/INST]", add_special_tokens=False)

# Robust function to find sublist

def find_sublist(sequence, sublist):
    for i in range(len(sequence) - len(sublist) + 1):
        if sequence[i : i + len(sublist)] == sublist:
            return i
    return -1


def preprocess_fn(examples):
    # build full prompts
    response_keys = [k for k in examples.keys() if k.startswith("response_") and k != "response_human"]
    answers = examples[response_keys[0]]
    prompts = [
        f"[INST] <<SYS>> {SYSTEM_PROMPT} <</SYS>>\n\n{ins} [/INST]\n\n{resp}"
        for ins, resp in zip(examples["instruction"], answers)
    ]

    # Tokenize with truncation to preserve all examples
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

    input_ids_list = tokenized["input_ids"]
    labels = []
    keep_mask = []
    for input_ids in input_ids_list:
        idx = find_sublist(input_ids, INST_END_IDS)
        if idx == -1:
            # drop sample if marker not found
            keep_mask.append(False)
            labels.append([ -100 ] * len(input_ids))
        else:
            # response starts after the inst end tokens
            response_start = idx + len(INST_END_IDS)
            lbls = [-100] * response_start + input_ids[response_start:]
            keep_mask.append(True)
            labels.append(lbls)

    # filter out invalid samples
    filtered = {k: [v for v, keep in zip(vals, keep_mask) if keep] for k, vals in tokenized.items()}
    filtered['labels'] = [lbl for lbl, keep in zip(labels, keep_mask) if keep]
    return filtered

processed = raw_dataset.map(
    preprocess_fn,
    batched=True,
    remove_columns=raw_dataset['train'].column_names
)
train_dataset = processed['train']
eval_dataset = processed.get('validation', None)

# 7. Setup Trainer with improved save/eval strategy
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
    save_strategy="epoch",
    evaluation_strategy="steps" if eval_dataset else "no",
    eval_steps=500,
    load_best_model_at_end=True if eval_dataset else False,
    seed=SEED,
    report_to="tensorboard",
    logging_dir=os.path.join(OUTPUT_DIR, "logs")
)

#data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
data_collator = default_data_collator

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 8. Start training
print("Starting training...")
trainer.train()

# 9. Save only LoRA adapter weights
model.save_pretrained(
    OUTPUT_DIR,
    safe_serialization=True,
    only_lora_weights=True
)
print(f"LoRA adapter weights saved to {OUTPUT_DIR}")
