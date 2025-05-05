from transformers import pipeline
import torch
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import gc
import math

########################################################

def create_synthetic_data_local(model_name, model, tokenizer, input_path, output_path, split=None):
    """
    Generate synthetic data using a locally loaded LLM
    
    Args:
        model_name: Name of the model from HuggingFace
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        split: Optional tuple (start, end) to process only a subset of the input
    """
    
    
    # Read lines from the input file
    with open(input_path, "r", encoding="utf-8") as fin:
        all_lines = fin.readlines()
    
    # If `split` is set, slice accordingly
    if split is not None:
        all_lines = all_lines[split[0]:split[1]]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Open output in append mode
    with open(output_path, "a", encoding="utf-8") as fout:
        for i, line in tqdm(enumerate(all_lines), total=len(all_lines), desc="Generating responses"):
            try:
                # Parse the JSON data
                data = json.loads(line)
                instruction = data["instruction"]
                
                # Prepare input for the model
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": instruction}
                ]
                
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                
                # Generate response with return_dict_in_generate=True to get probabilities
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
                
                # Extract only the generated part (not the input)
                input_length = model_inputs.input_ids.shape[1]
                generated_ids_only = [
                    output_ids[input_length:] for output_ids in generated_ids
                ]
                
                # Decode the generated tokens
                generated_text = tokenizer.batch_decode(generated_ids_only, skip_special_tokens=True)[0]
                
                # Get the token strings for each generated token
                generated_tokens = [tokenizer.decode(token_id.item()) for token_id in generated_ids_only[0]]
                
                # Compute probabilities, log‑probs, and sequence perplexity
                token_probabilities = []
                token_logprobs = []
                for step_idx, logits in enumerate(scores):
                    # Convert logits to log‑probabilities (temperature already applied)
                    log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
                    token_id   = generated_ids_only[0][step_idx].item()
                    logp       = log_probs[token_id].item()
                    token_logprobs.append(logp)
                    token_probabilities.append(math.exp(logp))

                # Perplexity = exp( – average log‑prob )
                avg_neg_logp = -sum(token_logprobs) / len(token_logprobs)
                perplexity   = math.exp(avg_neg_logp)

                # Store everything in the data dict
                model_short = model_name.split("/")[-1]
                data[f"response_{model_short}"]           = generated_text
                data[f"tokens_{model_short}"]             = generated_tokens
                data[f"token_probabilities_{model_short}"]= token_probabilities
                data[f"token_logprobs_{model_short}"]     = token_logprobs
                data[f"perplexity_{model_short}"]         = perplexity
                
                # Write out the updated data immediately
                fout.write(json.dumps(data, ensure_ascii=False))
                fout.write("\n")
                
                # Every 10 lines, explicitly flush to disk
                if i % 10 == 0:
                    fout.flush()
                    
                # Print progress info
                if i % 10 == 0:
                    print(f"Processed {i}/{len(all_lines)} examples.")
                
            except Exception as e:
                print(f"Error on line {i}: {e}")
                break 
    
    print("Done generating synthetic data.")


########################################################


if __name__ == "__main__":

    # Check CUDA availability and GPU information
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA is available with {device_count} device(s)")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"Device {i}: {device_name}")
            
            # Check if this is an H100
            if "H100" in device_name:
                print(f"H100 GPU detected! Using device {i}")
                torch.cuda.set_device(i)  # Explicitly set to use this GPU
            else:
                print(f"Warning: Device {i} is not an H100")
        
        # Show current device
        current_device = torch.cuda.current_device()
        print(f"Currently using device {current_device}: {torch.cuda.get_device_name(current_device)}")
    else:
        print("CUDA is not available. GPUs cannot be used.")
        sys.exit(1)  # Exit if no CUDA available

    # Load the model
    torch.set_grad_enabled(False) # turn off for inference

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Explicitly set device_map to use the H100 if detected
    device_map = "cuda"  # Default behavior uses all available GPUs
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        use_safetensors=True#,
        #attn_implementation="flash_attention_2"  # Enable Flash Attention (first needs to be installed)
    )

    # Print which device the model is actually on
    print(f"Model is on device: {next(model.parameters()).device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)


    # Configuration

    dolly_version = 1
    from_line = None  # Starting line 



    # Set up paths
    #input_path = f"/scratch-shared/mschaffelder/Data/Finetuning/Dolly/dolly_train_{dolly_version}.jsonl"
    input_path = f"/scratch-shared/mschaffelder/Data/Finetuning/Dolly/dolly_test.jsonl" # create synthetic test set



    # Create a model-specific output path
    model_short = model_name.split("/")[-1]
    # output_path = f"/scratch-shared/mschaffelder/Data/Finetuning/Augmented/Small/Llama_with_perplexity/dolly_train_{dolly_version}_{model_short}.jsonl"
    output_path = f"/scratch-shared/mschaffelder/Data/Finetuning/Augmented/Small/Llama/dolly_test_{model_short}.jsonl"


    print(f"Generating data with {model_name}")

    create_synthetic_data_local(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        input_path=input_path,
        output_path=output_path,
        split=(from_line, None)
    )


    print(f"Completed generation with {model_name}")


    # Clearing memory 

    del model  
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()