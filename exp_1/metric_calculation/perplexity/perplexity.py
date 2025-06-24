from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
import gc
from accelerate.utils import get_max_memory
import argparse

def get_token_probabilities(model_name, model, input_path, output_path, tokenizer, temperature=1.0, split=None):

    model.config.is_decoder = True # Model is decoder
    model.config.use_causal_mask = True # Use causal mask to avoid "looking at" future tokens

    with open(input_path, "r", encoding="utf-8") as fin: # read input file
        all_lines = fin.readlines()

    if split is not None: # read from desired line to end of file
        all_lines = all_lines[split[0]:split[1]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    with open(output_path, "a", encoding="utf-8") as fout:

        for i, line in tqdm(enumerate(all_lines), total=len(all_lines), desc="Processing inputs"):

            try:

                data = json.loads(line)
                response_key = "response_model"
                response = data[response_key] # get response
                instruction = data["instruction"] # get instruction belonging to response

                messages = [ # put into chat format
                    {"role": "system", "content": "You are a helpful assistant."}, # system prompt
                    {"role": "user", "content": instruction}, # user prompt
                    {"role": "assistant", "content": response} # model response
                ]

                # Tokenize with chat template but don't add gen prompt
                full_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False # don't want to generate anything
                )

                full_inputs = tokenizer(full_text, return_tensors="pt").to(model.device) # tokenize the text and move to device

                with torch.no_grad():
                    outputs = model(input_ids=full_inputs["input_ids"]) # run the model on the input
                    logits = outputs.logits # get logits
                    

                scaled_logits = logits / temperature # apply temperature
                probs = torch.softmax(scaled_logits, dim=-1) # get probabilities
                log_probs = F.log_softmax(scaled_logits, dim=-1) # get log probabilities
                input_ids = full_inputs["input_ids"][0][:-1] # get input ids

                # Find where the assistant response starts
                system_and_user_text = tokenizer.apply_chat_template(
                    messages[:2],
                    tokenize=False,
                    add_generation_prompt=False
                )
                num_prompt_tokens = len(tokenizer(system_and_user_text)["input_ids"])+4 # might need to change this based on the model

                response_ids = input_ids[num_prompt_tokens:] # get response ids (after prompt tokens)

                tokens = []
                token_probs = []
                token_logprobs = []
                token_logits = []

                
                for j, token_id in enumerate(response_ids):
                    context_index = num_prompt_tokens + j - 1
                    if context_index < 0:
                        continue  # skip if there's no valid context to predict from
                    
                    prob = round(probs[0, context_index, token_id].item(), 5) # get probability of next token from previous 
                    logprob = round(log_probs[0, context_index, token_id].item(), 5) # get log probability of next token from previous 
                    logit = round(logits[0, context_index, token_id].item(), 5) # get logit of next token from previous 
                    token = tokenizer.decode(token_id) # decode token id


                    tokens.append(token) # add token to list
                    token_probs.append(prob) # add probability to list
                    token_logprobs.append(logprob) # add log probability to list
                    token_logits.append(logit) # add logit to list

                # Calculate perplexity
                avg_neg_logp = -sum(token_logprobs) / len(token_logprobs) if token_logprobs else 0
                perplexity = round(torch.exp(torch.tensor(avg_neg_logp)).item(), 5)

                model_short_name = model_name.split("/")[-1]
                data[f"token_probabilities_{model_short_name}"] = {
                    "tokens": tokens,
                    "probs": token_probs,
                    "logprobs": token_logprobs,
                    "logits": token_logits,
                    "perplexity": perplexity
                }

                fout.write(json.dumps(data, ensure_ascii=False) + "\n")

                if i % 10 == 0:
                    fout.flush()

                # Free up memory
                del outputs, logits, scaled_logits, probs, log_probs, input_ids
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error on line {i}: {e}")
                continue

    print("Done processing inputs.")


def get_token_probabilities_human(model_name, model, input_path, output_path, tokenizer, temperature=1.0, split=None):

    model.config.is_decoder = True # Model is decoder
    model.config.use_causal_mask = True # Use causal mask to avoid "looking at" future tokens

    with open(input_path, "r", encoding="utf-8") as fin: # read input file
        all_lines = fin.readlines()

    if split is not None: # read from desired line to end of file
        all_lines = all_lines[split[0]:split[1]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    with open(output_path, "a", encoding="utf-8") as fout:

        for i, line in tqdm(enumerate(all_lines), total=len(all_lines), desc="Processing inputs"):

            try:

                data = json.loads(line)
                response_key = "response_human"
                response = data[response_key] # get response
                instruction = data["instruction"] # get instruction belonging to response

                messages = [ # put into chat format
                    {"role": "system", "content": "You are a helpful assistant."}, # keep system prompt to make more comparable with model data perplexity
                    {"role": "user", "content": instruction}, # user prompt
                    {"role": "assistant", "content": response} # human response
                ]

                # Tokenize with chat template but don't add gen prompt
                full_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False # don't want to generate anything
                )

                full_inputs = tokenizer(full_text, return_tensors="pt").to(model.device) # tokenize the text and move to device

                with torch.no_grad():
                    outputs = model(input_ids=full_inputs["input_ids"]) # run the model on the input
                    logits = outputs.logits # get logits
                    

                scaled_logits = logits / temperature # apply temperature
                probs = torch.softmax(scaled_logits, dim=-1) # get probabilities
                log_probs = F.log_softmax(scaled_logits, dim=-1) # get log probabilities
                input_ids = full_inputs["input_ids"][0][:-1] # get input ids

                # Find where the assistant response starts
                system_and_user_text = tokenizer.apply_chat_template(
                    messages[:2],
                    tokenize=False,
                    add_generation_prompt=False
                )
                num_prompt_tokens = len(tokenizer(system_and_user_text)["input_ids"])+4 # might need to change this based on the model

                response_ids = input_ids[num_prompt_tokens:] # get response ids (after prompt tokens)

                tokens = []
                token_probs = []
                token_logprobs = []
                token_logits = []

                
                for j, token_id in enumerate(response_ids):
                    context_index = num_prompt_tokens + j - 1
                    if context_index < 0:
                        continue  # skip if there's no valid context to predict from
                    
                    prob = round(probs[0, context_index, token_id].item(), 5) # get probability of next token from previous 
                    logprob = round(log_probs[0, context_index, token_id].item(), 5) # get log probability of next token from previous 
                    logit = round(logits[0, context_index, token_id].item(), 5) # get logit of next token from previous 
                    token = tokenizer.decode(token_id) # decode token id


                    tokens.append(token) # add token to list
                    token_probs.append(prob) # add probability to list
                    token_logprobs.append(logprob) # add log probability to list
                    token_logits.append(logit) # add logit to list

                # Calculate perplexity
                avg_neg_logp = -sum(token_logprobs) / len(token_logprobs) if token_logprobs else 0
                perplexity = round(torch.exp(torch.tensor(avg_neg_logp)).item(), 5)


                data[f"token_probabilities_human"] = {
                    "tokens": tokens,
                    "probs": token_probs,
                    "logprobs": token_logprobs,
                    "logits": token_logits,
                    "perplexity": perplexity
                }

                fout.write(json.dumps(data, ensure_ascii=False) + "\n")

                if i % 10 == 0:
                    fout.flush()

                # Free up memory
                del outputs, logits, scaled_logits, probs, log_probs, input_ids
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error on line {i}: {e}")
                continue

    print("Done processing inputs.")



def main(): 
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Calculate token probabilities and perplexity for given inputs.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the Hugging Face model to use.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output JSONL file.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for scaling logits.")
    
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
    
    print(f"Model loaded. Device map: {model.hf_device_map}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Calculate token probabilities & perplexity
    print(f"Processing input file: {args.input_path}")
    print(f"Output will be saved to: {args.output_path}")

    print("Calculating perplexity for model responses.")
    get_token_probabilities(args.model_name, model, args.input_path, args.output_path, tokenizer, temperature=args.temperature)

    # Clearing memory 
    del model  
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("Processing complete.")


if __name__ == "__main__":
    main()
