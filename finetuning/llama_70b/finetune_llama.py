import argparse
import os
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig # Use SFTConfig instead of TrainingArguments
import atexit

# Define command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.1-70B-Instruct with LoRA.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training.jsonl file.")
    parser.add_argument("--validation_file", type=str, required=True, help="Path to the validation.jsonl file.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the model and logs.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Number of gradient accumulation steps.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension (r).")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length for tokenization.")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful AI assistant.", help="Optional system prompt for the chat template.")
    return parser.parse_args()


def format_dataset_for_sft(examples, system_prompt):
    """Format dataset for SFTTrainer - returns raw text, not tokenized"""
    formatted_texts = []
    for i in range(len(examples['instruction'])):
        instruction = examples['instruction'][i]
        response = examples['response_model'][i]

        # Construct the conversation in a simple format that SFTTrainer can handle
        # We'll let SFTTrainer handle the chat template application
        if system_prompt:
            text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
        else:
            text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
        
        formatted_texts.append(text)
    
    return {"text": formatted_texts}

def load_and_preprocess_data(train_file, validation_file, system_prompt):
    """Load and preprocess data for SFTTrainer"""
    # Check if files exist
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(validation_file):
        raise FileNotFoundError(f"Validation file not found: {validation_file}")
    
    print(f"Loading training data from: {train_file}")
    print(f"Loading validation data from: {validation_file}")
    
    train_dataset = load_dataset("json", data_files=train_file, split="train")
    eval_dataset = load_dataset("json", data_files=validation_file, split="train")

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")

    # Apply formatting - convert to text format for SFTTrainer
    train_dataset = train_dataset.map(
        lambda examples: format_dataset_for_sft(examples, system_prompt),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        lambda examples: format_dataset_for_sft(examples, system_prompt),
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    return train_dataset, eval_dataset


def initialize_model_and_tokenizer(model_id):
    """Initialize model and tokenizer without quantization"""
    print(f"Loading model: {model_id}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto", # Automatically distribute model layers across GPUs
            #attn_implementation="flash_attention_2", # Use Flash Attention 2 for speed and memory
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to load with flash_attention_2, falling back to default attention: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    
    model.config.use_cache = False # Disable cache during training
    model.config.pretraining_tp = 1 # Required for Llama 3.1

    # Enable gradient checkpointing for memory optimization
    model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Important for Llama 3.1 and batching

    return model, tokenizer


def configure_lora(model, lora_r, lora_alpha):
    """Configure LoRA for the model"""
    # Define target modules for Llama 3.1
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=0.1, # Dropout probability for LoRA layers
        r=lora_r, # LoRA rank
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def setup_training_arguments(args):
    """Setup SFT training arguments with conservative settings"""
    return SFTConfig(
        # Basic training parameters
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        
        # Logging and evaluation
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3, # Keep only 3 checkpoints to save disk space
        
        # Precision and optimization
        bf16=True, # Enable bfloat16 precision
        tf32=True, # Enable TF32 for faster matmul on Ampere+ GPUs
        optim="paged_adamw_8bit", # Recommended optimizer for LoRA
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        seed=42,
        dataloader_pin_memory=False, # Reduce memory usage
        gradient_checkpointing=True,
        
        # SFT-specific parameters
        max_length=args.max_seq_length, # Maximum sequence length
        dataset_text_field="text", # Field containing the text data
        packing=False, # Disable packing for better control
        # report_to="wandb", # Uncomment to enable Weights & Biases logging
    )


def cleanup_distributed():
    """Clean up distributed training resources"""
    if dist.is_initialized():
        dist.destroy_process_group()

# Register cleanup function
atexit.register(cleanup_distributed)

def main():
    args = parse_args()

    # Validate arguments
    if not os.path.exists(os.path.dirname(args.output_dir)):
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    model_id = "meta-llama/Llama-3.1-70B-Instruct"

    try:
        model, tokenizer = initialize_model_and_tokenizer(model_id)
        model = configure_lora(model, args.lora_r, args.lora_alpha)

        train_dataset, eval_dataset = load_and_preprocess_data(
            args.train_file, args.validation_file, args.system_prompt
        )

        training_args = setup_training_arguments(args)

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Start training
        print("Starting training...")
        trainer.train()

        # Save the final adapter model
        print(f"Saving model to {args.output_dir}")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir) # Save tokenizer as well
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        # Ensure cleanup happens even on error
        cleanup_distributed()
        raise
    finally:
        # Additional cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()