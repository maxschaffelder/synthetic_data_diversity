import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

def parse_args():
    parser = argparse.ArgumentParser(description="Consolidate a sharded FSDP checkpoint into a single LoRA adapter.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the sharded checkpoint directory (e.g., checkpoint-318).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the consolidated LoRA adapter.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension (r).")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter.")
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator()

    model_id = "meta-llama/Llama-3.1-70B-Instruct"

    # Initialize the model on the 'meta' device to avoid allocating huge memory for weights we will overwrite.
    with accelerator.init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        # Re-create the same LoRA configuration used during training.
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)

    # Use bfloat16 for loading, consistent with the training precision.
    model = model.to(torch.bfloat16)
    
    # Let accelerate prepare the model. This is crucial as it will wrap it with FSDP
    # based on your environment's accelerate config, ensuring the structure matches.
    model = accelerator.prepare(model)
    
    accelerator.print(f"Loading sharded state from {args.checkpoint_dir}")
    
    # Use accelerate's robust load_state function.
    try:
        accelerator.load_state(args.checkpoint_dir)
        accelerator.print("Successfully loaded sharded state.")
    except Exception as e:
        accelerator.print(f"Failed to load state with accelerator.load_state: {e}")
        accelerator.print("This may be due to a mismatch in model structure or library versions.")
        return

    # Unwrap the model to get the underlying PeftModel for saving.
    unwrapped_model = accelerator.unwrap_model(model)
    
    accelerator.print(f"Saving consolidated LoRA adapter to {args.output_dir}")
    
    # Save on the main process to prevent multiple processes from writing to the same files.
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # This peft method saves just the trained LoRA adapter weights and the config.
        unwrapped_model.save_pretrained(args.output_dir)
        
        # Also save the tokenizer for convenience.
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.save_pretrained(args.output_dir)
        
    accelerator.wait_for_everyone()
    accelerator.print("Consolidation complete.")

if __name__ == "__main__":
    main() 