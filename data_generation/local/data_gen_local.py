from transformers import pipeline
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import gc
import sys
import math
import argparse

########################################################

def create_synthetic_data_local(model_name, model, tokenizer, input_path, output_path, batch_size=8, split=None):
    """
    Generate synthetic data using a locally loaded LLM in batches.
    
    Args:
        model_name: Name of the model from HuggingFace.
        model: The loaded model object.
        tokenizer: The loaded tokenizer object.
        input_path: Path to input JSONL file.
        output_path: Path to output JSONL file.
        batch_size: The number of examples to process in a single batch.
        split: Optional tuple (start, end) to process only a subset of the input.
    """
    
    # Read lines from the input file
    with open(input_path, "r", encoding="utf-8") as fin:
        all_lines = fin.readlines()
    
    # If `split` is set, slice accordingly
    if split is not None and len(split) == 2 and split[0] is not None and split[1] is not None:
        all_lines = all_lines[split[0]:split[1]]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Open output in append mode
    with open(output_path, "a", encoding="utf-8") as fout:
        # Process data in batches
        for i in tqdm(range(0, len(all_lines), batch_size), desc="Generating responses"):
            batch_lines = all_lines[i:i + batch_size]
            
            try:
                # Prepare batch of instructions
                batch_data = [json.loads(line) for line in batch_lines]
                instructions = [data["instruction"] for data in batch_data]
                
                messages_batch = [
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": instruction}
                    ] for instruction in instructions
                ]
                
                text_batch = [
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    ) for messages in messages_batch
                ]
                
                # Tokenize batch with padding
                model_inputs = tokenizer(
                    text_batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=2048 # To avoid excessive memory usage
                ).to(model.device)
                
                # Generate response for the whole batch
                generation_output = model.generate(
                    **model_inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                generated_ids = generation_output.sequences
                scores = generation_output.scores
                
                # Decode and process each item in the batch
                input_length = model_inputs.input_ids.shape[1]
                generated_ids_only = generated_ids[:, input_length:]
                
                generated_texts = tokenizer.batch_decode(generated_ids_only, skip_special_tokens=True)
                
                for j in range(len(generated_texts)):
                    data = batch_data[j]
                    item_generated_ids = generated_ids_only[j]
                    
                    # Stop at the first padding token if any
                    try:
                        pad_token_index = (item_generated_ids == tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0].item()
                        item_generated_ids = item_generated_ids[:pad_token_index]
                    except IndexError:
                        # No padding token found
                        pass

                    generated_tokens = [tokenizer.decode(token_id.item()) for token_id in item_generated_ids]
                    
                    token_probabilities = []
                    token_logprobs = []
                    
                    for step_idx, step_scores in enumerate(scores):
                        if step_idx >= len(item_generated_ids):
                            break
                        
                        token_id = item_generated_ids[step_idx].item()
                        logits = step_scores[j]
                        
                        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                        logp = log_probs[token_id].item()
                        
                        token_logprobs.append(logp)
                        token_probabilities.append(math.exp(logp))

                    perplexity = 0.0
                    if token_logprobs:
                        avg_neg_logp = -sum(token_logprobs) / len(token_logprobs)
                        perplexity = math.exp(avg_neg_logp)

                    data["response_model"] = generated_texts[j]
                    data["model_name"] = model_name
                    data["tokens"] = generated_tokens
                    data["token_probabilities"] = token_probabilities
                    data["token_logprobs"] = token_logprobs 
                    data["perplexity"] = perplexity
                    
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                
                fout.flush()
                
            except Exception as e:
                print(f"Error on batch starting at line {i}: {e}")
                print("Skipping this batch and continuing...")
                continue
    
    print("Done generating synthetic data.")


########################################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate synthetic data using a locally loaded LLM.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model from HuggingFace.")
    parser.add_argument("--model_short", type=str, required=True, help="Short name for the model, used in output file paths.")
    args = parser.parse_args()

    # Check CUDA availability and GPU information
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA is available with {device_count} device(s)")
        
        h100_detected = False
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"Device {i}: {device_name}")
            
            if "H100" in device_name:
                print(f"H100 GPU detected! Using device {i}")
                torch.cuda.set_device(i)
                h100_detected = True
        
        if not h100_detected:
            print("Warning: No H100 GPU detected. Performance may vary.")

        current_device = torch.cuda.current_device()
        print(f"Currently using device {current_device}: {torch.cuda.get_device_name(current_device)}")
    else:
        print("CUDA is not available. GPUs cannot be used.")
        sys.exit(1)

    # Load the model
    torch.set_grad_enabled(False)

    model_name = args.model_name
    model_short = args.model_short
    
    # Explicitly set device_map to use the H100 if detected
    device_map = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map=device_map,
        use_safetensors=True
    )

    # Use torch.compile for a significant speed-up
    print("Compiling the model... (this may take a moment)")
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    print(f"Model is on device: {next(model.parameters()).device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # Configuration
    batch_size = 32  # Adjust based on GPU memory

    for i in range(1, 5):
        dolly_version = i
        from_line = None
        till_line = None

        # Set up paths
        input_path = f"/scratch-shared/mschaffelder/data/finetuning/dolly/dolly_train_{dolly_version}.jsonl"
        output_path = f"/scratch-shared/mschaffelder/data/finetuning/synthetic/Small/Llama/dolly_train_{dolly_version}_{model_short}.jsonl"

        print(f"Generating data with {model_name} for dolly_train_{dolly_version}.jsonl")

        create_synthetic_data_local(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            input_path=input_path,
            output_path=output_path,
            batch_size=batch_size,
            split=(from_line, till_line)
        )

    print(f"Completed generation with {model_name}")

    # No need to clear memory here as the script exits