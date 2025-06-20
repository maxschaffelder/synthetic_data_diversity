import argparse
import json
import os
import shutil

def find_best_checkpoint(output_dir):
    """Parses trainer_state.json to find the checkpoint with the best eval_loss."""
    state_file = os.path.join(output_dir, "trainer_state.json")
    if not os.path.exists(state_file):
        raise FileNotFoundError(f"trainer_state.json not found in {output_dir}. Make sure you are pointing to the correct output directory.")

    with open(state_file, 'r') as f:
        state = json.load(f)

    # The trainer logs the path to the best model checkpoint if load_best_model_at_end is True,
    # but since it's False, we parse the log history manually.
    best_checkpoint_path = state.get("best_model_checkpoint")
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        print(f"Found best checkpoint directly from 'best_model_checkpoint' key: {best_checkpoint_path}")
        return best_checkpoint_path

    # Fallback to parsing log history if the key isn't there or the path is broken
    print("Could not find a valid 'best_model_checkpoint'. Parsing log history to find the best step.")
    log_history = state.get('log_history', [])
    eval_logs = [log for log in log_history if 'eval_loss' in log and 'step' in log]
    
    if not eval_logs:
        raise ValueError("Could not find any evaluation logs in trainer_state.json to determine the best checkpoint.")

    best_log = min(eval_logs, key=lambda x: x['eval_loss'])
    best_step = best_log['step']
    best_checkpoint_path = os.path.join(output_dir, f"checkpoint-{best_step}")
    
    print(f"Determined best checkpoint from log history: {best_checkpoint_path} (Step: {best_step}, Eval Loss: {best_log['eval_loss']:.4f})")
    
    return best_checkpoint_path

def main():
    parser = argparse.ArgumentParser(description="Find the best LoRA adapter from training checkpoints and copy it to the root output directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the training checkpoints and trainer_state.json are saved.")
    args = parser.parse_args()

    try:
        best_checkpoint_dir = find_best_checkpoint(args.output_dir)

        if not os.path.isdir(best_checkpoint_dir):
            raise FileNotFoundError(f"Best checkpoint directory not found at the calculated path: {best_checkpoint_dir}")

        # These are the typical files for a LoRA adapter saved by PEFT
        files_to_copy = [
            "adapter_config.json",
            "adapter_model.safetensors",
            "README.md",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model"
        ]

        print(f"\nCopying best adapter files from:\n  '{best_checkpoint_dir}'\nto:\n  '{args.output_dir}'\n")

        for filename in files_to_copy:
            source_file = os.path.join(best_checkpoint_dir, filename)
            destination_file = os.path.join(args.output_dir, filename)
            if os.path.exists(source_file):
                shutil.copy2(source_file, destination_file)
                print(f"  - Copied {filename}")
            else:
                # This is not an error; some files like tokenizer.model might not always exist depending on the tokenizer type.
                print(f"  - Skipping {filename} (not found in source checkpoint)")

        print("\nConsolidation complete. The best adapter is now available in the root of the output directory.")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
        print("Please ensure the output directory path is correct and that training has completed at least one evaluation step.")

if __name__ == "__main__":
    main() 