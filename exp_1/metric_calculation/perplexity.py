import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import os
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
import gc
import pandas as pd
import matplotlib.pyplot as plt
import glob
from accelerate import init_empty_weights
import torch.multiprocessing as mp
from accelerate.utils import get_max_memory

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
                response_key = [k for k in data if k.startswith('response_')][1] # to get model response
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
    # Load model & tokenizer
    torch.set_grad_enabled(False)

    model_name = "meta-llama/Llama-3.1-70B-Instruct"
    
    # Set environment variable to avoid CUDA memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Count available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    
    # Setup device map to distribute model across GPUs
    max_memory = get_max_memory()
    
    if num_gpus < 2:
        print("Warning: Less than 2 GPUs available. This might cause OOM errors.")
    
    # Load model with explicit device map to distribute across GPUs
    print("Loading model with multi-GPU parallelism...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # use bfloat16 for better performance on H100s
        device_map="auto",  # Auto-distribute across available GPUs
        max_memory=max_memory,
        use_safetensors=True,
        offload_folder="offload",  # Optional: offload to disk if needed
        offload_state_dict=True,  # Optional: offload parameters to CPU if needed
        low_cpu_mem_usage=True#,
        #attn_implementation="flash_attention_2" # flash attention 3 is used if available
    )
    
    print(f"Model loaded and distributed across GPUs. Device map: {model.hf_device_map}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Calculate token probabilities & perplexity

    dolly_version = 4
    temperature = 1.0

    input_path = f"/scratch-shared/mschaffelder/Data/Finetuning/Augmented/Medium/Other/dolly_train_{dolly_version}_Command.jsonl"
    output_path = f"/scratch-shared/mschaffelder/Data/analysis/perplexity/llama_medium/dolly_train_{dolly_version}_Command_medium_PPL.jsonl"
    get_token_probabilities(model_name, model, input_path, output_path, tokenizer, temperature=temperature)

    # Calculate PPL scores and token probs for all data in one go
    for dolly_version in range(1, 5):
        #input_path = f"../../Data/Finetuning/Augmented/Medium/Llama/dolly_train_{dolly_version}_Llama.jsonl"
        #output_path = f"../../Data/analysis/perplexity/llama_medium/dolly_train_{dolly_version}_Llama_medium_PPL.jsonl"
        input_path = f"/scratch-shared/mschaffelder/Data/Finetuning/Augmented/Medium/Llama/dolly_train_{dolly_version}_Llama.jsonl"
        output_path = f"/scratch-shared/mschaffelder/Data/analysis/perplexity/llama_medium/dolly_train_{dolly_version}_Llama_medium_PPL.jsonl"
        output_path_human = f"/scratch-shared/mschaffelder/Data/analysis/perplexity/llama_medium/dolly_train_{dolly_version}_Llama_medium_PPL_on_human_data.jsonl"

        #get_token_probabilities(model_name, model, input_path, output_path, tokenizer, temperature=temperature)
        #get_token_probabilities_human(model_name, model, input_path, output_path_human, tokenizer, temperature=temperature)

    # Clearing memory 
    del model  
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
