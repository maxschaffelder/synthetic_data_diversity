import argparse
import os
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig 
import atexit

# Define command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.1-8B-Instruct with LoRA.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Name of the model to fine-tune.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training data JSONL file.")
    parser.add_argument("--response_key", type=str, default="response_human", help="The key in the JSONL file that contains the response.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model and logs.")
    parser.add_argument("--validation_split_percentage", type=int, default=5, help="Percentage of the train file to use for validation.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per GPU for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension (r).")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length for tokenization.")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful AI assistant.", help="Optional system prompt for the chat template.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Number of steps between logs.")
    parser.add_argument("--eval_steps", type=int, default=20, help="Number of steps between evaluations.")
    parser.add_argument("--save_steps", type=int, default=100, help="Number of steps between checkpoints.")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Patience for early stopping.")
    return parser.parse_args()


def format_dataset(examples, system_prompt, response_key):
    """Formats the dataset into a conversational format suitable for SFTTrainer."""
    formatted_texts = []
    for i in range(len(examples['instruction'])):
        instruction = examples['instruction'][i]
        response = examples[response_key][i]
        
        # Using the Llama 3.1 chat template structure
        text = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>")
        
        formatted_texts.append(text)
    
    return {"text": formatted_texts}

def load_and_preprocess_data(train_file, validation_split_percentage, system_prompt, response_key):
    """Loads and prepares the dataset, splitting it for training and validation."""
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    dataset = load_dataset("json", data_files=train_file, split="train")
    
    split_dataset = dataset.train_test_split(test_size=(validation_split_percentage / 100.0), shuffle=True, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    print(f"Original dataset size: {len(dataset)}")
    print(f"Training dataset size after split: {len(train_dataset)}")
    print(f"Validation dataset size after split: {len(eval_dataset)}")

    train_dataset = train_dataset.map(lambda x: format_dataset(x, system_prompt, response_key), batched=True, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(lambda x: format_dataset(x, system_prompt, response_key), batched=True, remove_columns=eval_dataset.column_names)

    return train_dataset, eval_dataset


def initialize_model_and_tokenizer(model_id):
    """Initializes the model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    return model, tokenizer


def configure_lora(model, lora_r, lora_alpha):
    """Configures LoRA for the model."""
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def setup_training_arguments(args):
    """Sets up SFT training arguments."""
    return SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        tf32=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        seed=42,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
        ddp_find_unused_parameters=False,
    )

def cleanup_distributed():
    """Clean up distributed training resources to prevent resource leaks."""
    if dist.is_initialized():
        dist.destroy_process_group()

# Register the cleanup function to be called on script exit
atexit.register(cleanup_distributed)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = initialize_model_and_tokenizer(args.model_name)
    model = configure_lora(model, args.lora_r, args.lora_alpha)
    
    train_dataset, eval_dataset = load_and_preprocess_data(
        args.train_path, args.validation_split_percentage, args.system_prompt, args.response_key
    )

    training_args = setup_training_arguments(args)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    print("Starting training...")
    trainer.train()
    print("Training completed.")
    
    print(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()