from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
import gc
from accelerate.utils import get_max_memory
import argparse
from peft import PeftModel

def get_token_probabilities(model_name, model, input_path, output_path, tokenizer, response_key, temperature=1.0, split=None):
    """
    Calculates token probabilities and perplexity for responses in a JSONL file.
    """
    model.config.is_decoder = True
    model.config.use_causal_mask = True

    with open(input_path, "r", encoding="utf-8") as fin:
        all_lines = fin.readlines()

    if split is not None:
        all_lines = all_lines[split[0]:split[1]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as fout:
        for i, line in tqdm(enumerate(all_lines), total=len(all_lines), desc=f"Processing responses for key '{response_key}'"):
            try:
                data = json.loads(line)
                response = data[response_key]
                instruction = data["instruction"]

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response}
                ]

                # Tokenize using the chat template for robustness
                tokenized_output = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt"
                )
                input_ids = tokenized_output.to(model.device)

                with torch.no_grad():
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits

                scaled_logits = logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                log_probs = F.log_softmax(scaled_logits, dim=-1)
                
                input_ids_full = input_ids[0]

                # Robustly find where the assistant's response starts
                # This uses the tokenizer's template to determine the true length of the prompt.
                prompt_tokens = tokenizer.apply_chat_template(
                    messages[:-1], # Messages before the assistant's response
                    add_generation_prompt=True,
                    tokenize=True
                )
                num_prompt_tokens = len(prompt_tokens)

                # Get the token IDs corresponding to the response
                response_ids = input_ids_full[num_prompt_tokens:]

                tokens = []
                token_probs = []
                token_logprobs = []
                token_logits = []

                for j, token_id in enumerate(response_ids):
                    # The logits for the j-th response token (at full index `num_prompt_tokens + j`)
                    # are at logit tensor index `num_prompt_tokens + j - 1`.
                    context_index = num_prompt_tokens + j - 1
                    if context_index < 0:
                        continue # Should not happen with a valid prompt.

                    prob = round(probs[0, context_index, token_id].item(), 5)
                    logprob = round(log_probs[0, context_index, token_id].item(), 5)
                    logit = round(logits[0, context_index, token_id].item(), 5)
                    token = tokenizer.decode(token_id)

                    tokens.append(token)
                    token_probs.append(prob)
                    token_logprobs.append(logprob)
                    token_logits.append(logit)

                avg_neg_logp = -sum(token_logprobs) / len(token_logprobs) if token_logprobs else 0
                perplexity = round(torch.exp(torch.tensor(avg_neg_logp)).item(), 5)
                
                data["token_probabilities"] = {
                    "tokens": tokens,
                    "probs": token_probs,
                    "logprobs": token_logprobs,
                    "logits": token_logits,
                    "perplexity": perplexity
                }

                fout.write(json.dumps(data, ensure_ascii=False) + "\n")

                if i % 10 == 0:
                    fout.flush()

                del outputs, logits, scaled_logits, probs, log_probs, input_ids
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error on line {i} for input containing instruction '{data.get('instruction', 'N/A')}': {e}")
                continue

    print("Done processing inputs.")


def main(): 
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Calculate token probabilities and perplexity for given inputs.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the Hugging Face model to use.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output JSONL file.")
    parser.add_argument("--lora_model_path", type=str, default=None, help="Path to the LoRA adapter if using a fine-tuned model.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for scaling logits.")
    parser.add_argument("--response_key", type=str, default="response_model", help="The key in the JSONL file that contains the response text (e.g., 'response_model', 'response_human').")
    
    args = parser.parse_args()

    # Load model & tokenizer
    torch.set_grad_enabled(False)
    
    # Set environment variable to avoid CUDA memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Count available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    
    # Setup device map to distribute model across GPUs
    max_memory = get_max_memory()
    
    if num_gpus < 1: # Adjusted to allow single GPU operation
        print("Warning: No GPUs available. This script requires at least one GPU.")
        # Depending on the setup, might want to exit or force CPU, but for perplexity, GPU is highly recommended.
        # For now, will proceed, but model loading might fail or be very slow.
    
    # Load model with explicit device map to distribute across GPUs
    print(f"Loading model {args.model_name} with multi-GPU parallelism if available...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,  # use bfloat16 for better performance on H100s
        device_map="auto",  # Auto-distribute across available GPUs
        max_memory=max_memory,
        use_safetensors=True,
        offload_folder="offload",  # Optional: offload to disk if needed
        offload_state_dict=True,  # Optional: offload parameters to CPU if needed
        low_cpu_mem_usage=True#,
        #attn_implementation="flash_attention_2" # flash attention 3 is used if available
    )
    
    if args.lora_model_path:
        print(f"Loading LoRA adapter from {args.lora_model_path}...")
        model = PeftModel.from_pretrained(model, args.lora_model_path)
        print("Merging LoRA adapter into the base model...")
        model = model.merge_and_unload()
        print("LoRA adapter merged successfully.")

    print(f"Model loaded. Device map: {model.hf_device_map}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Calculate token probabilities & perplexity
    print(f"Processing input file: {args.input_path}")
    print(f"Output will be saved to: {args.output_path}")

    get_token_probabilities(
        model_name=args.model_name,
        model=model,
        input_path=args.input_path,
        output_path=args.output_path,
        tokenizer=tokenizer,
        response_key=args.response_key,
        temperature=args.temperature
    )

    # Clearing memory 
    del model  
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("Processing complete.")


if __name__ == "__main__":
    main()
