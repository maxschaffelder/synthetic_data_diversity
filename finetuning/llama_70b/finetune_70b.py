#!/usr/bin/env python3
import argparse
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3.1-70B-Instruct with LoRA")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct",
                        help="Pre-trained model name or path (default: %(default)s)")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to the training JSONL file")
    parser.add_argument("--val_file", type=str, required=True,
                        help="Path to the validation JSONL file")
    parser.add_argument("--response_key", type=str, default="response_model",
                        help="Key name for the response in JSONL (default: %(default)s)")
    parser.add_argument("--output_dir", type=str, default="lora-ft-output",
                        help="Directory to save fine-tuned model (default: %(default)s)")
    args = parser.parse_args()

    # Load tokenizer and model (with bf16 and multi-GPU support)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"  # automatically split across available GPUs
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Prepare LoRA config: tune only the specified projection layers
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)

    # Load dataset from JSONL (expects fields "instruction" and response key)
    data = load_dataset("json", data_files={"train": args.train_file, "validation": args.val_file})

    system_prompt = "You are a helpful assistant."

    # Token filter: remove too-long examples (>2048 tokens total)
    def filter_too_long(example):
        # Count tokens of system prompt + instruction + response
        text_in = system_prompt + "\n" + example["instruction"]
        text_out = example[args.response_key]
        num_tokens = len(tokenizer(text_in, add_special_tokens=False)["input_ids"]) \
                     + len(tokenizer(text_out, add_special_tokens=False)["input_ids"])
        return num_tokens <= 2048

    data["train"] = data["train"].filter(filter_too_long)
    data["validation"] = data["validation"].filter(filter_too_long)

    # Preprocessing: tokenize and create inputs/labels
    def tokenize_and_format(examples):
        inputs = []
        labels = []
        for instruction, response in zip(examples["instruction"], examples[args.response_key]):
            prompt = system_prompt + "\n" + instruction
            input_ids = tokenizer(prompt, truncation=True, max_length=2048, add_special_tokens=False)["input_ids"]
            response_ids = tokenizer(response, truncation=True, max_length=2048, add_special_tokens=False)["input_ids"]

            # Combine prompt + response
            full_ids = input_ids + response_ids + [tokenizer.eos_token_id]
            # Create labels: mask prompt tokens
            label_ids = [-100] * len(input_ids) + response_ids + [tokenizer.eos_token_id]
            inputs.append(torch.tensor(full_ids, dtype=torch.long))
            labels.append(torch.tensor(label_ids, dtype=torch.long))
        return {"input_ids": inputs, "labels": labels}

    data_enc = data.map(tokenize_and_format, batched=True, remove_columns=data["train"].column_names)

    # Custom data collator for padding
    def collate_fn(batch):
        input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True,
                                 padding_value=tokenizer.pad_token_id)
        label_ids  = pad_sequence([item["labels"]   for item in batch], batch_first=True,
                                 padding_value=-100)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_ids}

    # Training arguments: 2 epochs, batch size 1, bf16 precision
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_enc["train"],
        eval_dataset=data_enc["validation"],
        tokenizer=tokenizer,
        data_collator=collate_fn
    )

    # Train and evaluate
    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
