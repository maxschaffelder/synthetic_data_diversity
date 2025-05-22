import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_tokenizer(base_model_path, lora_model_path):
    logging.info(f"Starting to load tokenizer from: {base_model_path} with padding_side='left'")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        padding_side='left' # Set padding side to left for decoder-only models
    )
    logging.info("Tokenizer loaded.")

    # Ensure tokenizer uses eos_token as pad_token for consistency with Llama generation practices.
    # This also sets tokenizer.pad_token_id to tokenizer.eos_token_id.
    if tokenizer.pad_token_id != tokenizer.eos_token_id:
        logging.warning(
            f"Tokenizer.pad_token_id ({tokenizer.pad_token_id}) is not eos_token_id ({tokenizer.eos_token_id}). "
            f"Forcing pad_token to eos_token for Llama consistency."
        )
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.pad_token is None: # If pad_token is None, but pad_token_id might exist (less likely)
        logging.warning("Tokenizer.pad_token is None. Setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    logging.info(f"Tokenizer configured: pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}, padding_side='{tokenizer.padding_side}'")

    logging.info(f"Starting to load base model from: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    logging.info(f"Base model loaded. Original vocab size: {model.config.vocab_size}")
    
    # Resize token embeddings if necessary
    expected_vocab_size = 128257
    if model.config.vocab_size < expected_vocab_size:
        logging.info(f"Resizing token embeddings from {model.config.vocab_size} to {expected_vocab_size}...")
        model.resize_token_embeddings(expected_vocab_size)
        logging.info(f"Token embeddings resized. New vocab size: {model.config.vocab_size}")
    else:
        logging.info("Token embeddings do not need resizing.")
    
    logging.info(f"Starting to load LoRA weights from: {lora_model_path}")
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, lora_model_path)
    logging.info("LoRA weights loaded and merged with the base model.")
    
    # Align model's pad_token_id with tokenizer's pad_token_id if necessary
    # This is important for the model.generate() method.
    if model.config.pad_token_id is None or model.config.pad_token_id != tokenizer.pad_token_id:
        logging.info(f"Aligning model.config.pad_token_id ({model.config.pad_token_id}) to tokenizer.pad_token_id ({tokenizer.pad_token_id}).")
        model.config.pad_token_id = tokenizer.pad_token_id
        
    return model, tokenizer

def generate_response(model, tokenizer, prompts_batch, max_length=1024):
    if not prompts_batch:
        return []

    logging.info(f"Generating responses for batch of {len(prompts_batch)} prompts. First prompt (50 chars): '{prompts_batch[0][:50]}...'")
    logging.info(f"Input device: {model.device}, preparing to move inputs to device.")
    
    # Tokenize the batch of prompts.
    # padding_side='left' was set when tokenizer was loaded.
    # tokenizer.pad_token was also set (to eos_token if None) when tokenizer was loaded.
    inputs = tokenizer(
        prompts_batch, 
        return_tensors="pt", 
        padding=True, # Pad to the longest sequence in the batch
        truncation=True, # Truncate sequences longer than model max length 
        max_length=model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 2048 
    ).to(model.device)
    logging.info("Batch of inputs tokenized, padded, truncated, and moved to model device.")
    
    with torch.no_grad():
        logging.info("Starting batched model.generate()...")
        # model.generate() will use model.config.pad_token_id
        outputs = model.generate(
            **inputs,
            max_length=max_length, 
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            # pad_token_id=tokenizer.eos_token_id # Use model.config.pad_token_id which should be aligned
        )
        logging.info("Batched model.generate() completed.")
    
    logging.info("Decoding batch of responses...")
    raw_decoded_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    logging.info("Batch of responses decoded (raw).")

    # Remove the prompt from the beginning of each response
    cleaned_responses = []
    for prompt, raw_response in zip(prompts_batch, raw_decoded_responses):
        # Check if the raw_response starts with the prompt. 
        # Need to be a bit careful if tokenization differences make them not exactly match at string level,
        # but usually for simple continuation it's fine.
        # A more robust way might involve comparing token IDs if there are issues, 
        # but string prefix removal is usually sufficient.
        if raw_response.startswith(prompt):
            cleaned_response = raw_response[len(prompt):].lstrip() # lstrip to remove leading whitespace
            cleaned_responses.append(cleaned_response)
            logging.debug(f"Cleaned response. Original: '{raw_response[:100]}...', Cleaned: '{cleaned_response[:100]}...'")
        else:
            # If the prompt isn't at the start (e.g., model generated something completely different or empty),
            # keep the raw response but log a warning, as this might indicate an issue.
            logging.warning(f"Prompt not found at the beginning of the raw response. Prompt: '{prompt[:50]}...', Raw Response: '{raw_response[:50]}...'")
            cleaned_responses.append(raw_response) 
    
    logging.info("Prompts removed from responses.")
    return cleaned_responses

def main():
    # Configuration
    BATCH_SIZE = 8 
    base_model_path = "meta-llama/Llama-3.1-8B-Instruct"
    lora_model_path = "/scratch-shared/mschaffelder/Data/ft_models/lora_llama_8b_single_v6/checkpoint-1684"
    test_data_path = "/scratch-shared/mschaffelder/Data/Finetuning/Dolly/dolly_test.jsonl"
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(base_model_path, lora_model_path)
    
    # Create output directory if it doesn't exist
    output_dir = "/scratch-shared/mschaffelder/Data/exp_1/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'generation_results_8b_single_v6.jsonl')

    # Clear/Create the output file at the beginning of this run
    with open(output_path, 'w') as f:
        logging.info(f"Output file {output_path} created/cleared.")
        # Optionally, write a header if your JSONL has a specific schema, but for simple line-by-line JSON, it's not needed.
    
    # Process test data
    logging.info("Starting to process test data...")
    prompts_batch = []
    data_batch_info = [] 
    
    with open(test_data_path, 'r') as f:
        logging.info(f"Opened test data file: {test_data_path}")
        for i, line in enumerate(f):
            logging.info(f"Reading line {i+1} from test data...")
            try:
                data = json.loads(line)
                prompt = data['instruction']
                prompts_batch.append(prompt)
                data_batch_info.append(data) 
                logging.info(f"Added prompt from line {i+1} to batch.")
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from line {i+1}: {e}")
                continue 
            except KeyError as e:
                logging.error(f"Missing key 'instruction' in line {i+1}: {e}")
                continue
            
            if len(prompts_batch) == BATCH_SIZE:
                logging.info(f"Processing batch of {len(prompts_batch)} prompts (up to line {i+1})...")
                try:
                    generated_responses_batch = generate_response(model, tokenizer, prompts_batch)
                    logging.info(f"Batch of responses generated.")
                    for idx, (original_data, gen_response) in enumerate(zip(data_batch_info, generated_responses_batch)):
                        # Prepare result item for this entry
                        result_item = {
                            'instruction': original_data['instruction'],
                            'response_model': gen_response,
                            'response_human': original_data.get('response', '')
                        }
                        # Append this single result to the file
                        with open(output_path, 'a') as f:
                            f.write(json.dumps(result_item) + '\n')
                    logging.info(f"Appended {len(generated_responses_batch)} results from batch to {output_path}")
                except Exception as e:
                    logging.error(f"Error during batch generate_response (lines around {i+1}): {e}")
                    # Store error for all items in this failed batch by writing them out
                    with open(output_path, 'a') as f:
                        for original_data in data_batch_info:
                            error_result_item = {
                                'instruction': original_data['instruction'],
                                'response_model': f"ERROR_BATCH: {e}",
                                'response_human': original_data.get('response', '')
                            }
                            f.write(json.dumps(error_result_item) + '\n')
                    logging.info(f"Appended {len(data_batch_info)} error results from batch to {output_path}")
                finally:
                    prompts_batch = [] 
                    data_batch_info = []

    if prompts_batch: 
        logging.info(f"Processing final batch of {len(prompts_batch)} prompts...")
        try:
            generated_responses_batch = generate_response(model, tokenizer, prompts_batch)
            logging.info(f"Final batch of responses generated.")
            for idx, (original_data, gen_response) in enumerate(zip(data_batch_info, generated_responses_batch)):
                # Prepare result item for this entry
                result_item = {
                    'instruction': original_data['instruction'],
                    'response_model': gen_response,
                    'response_human': original_data.get('response', '')
                }
                # Append this single result to the file
                with open(output_path, 'a') as f:
                    f.write(json.dumps(result_item) + '\n')
            logging.info(f"Appended {len(generated_responses_batch)} results from final batch to {output_path}")
        except Exception as e:
            logging.error(f"Error during final batch generate_response: {e}")
            # Store error for all items in this failed batch by writing them out
            with open(output_path, 'a') as f:
                for original_data in data_batch_info:
                    error_result_item = {
                        'instruction': original_data['instruction'],
                        'generated_response': f"ERROR_FINAL_BATCH: {e}",
                        'ground_truth': original_data.get('response', '')
                    }
                    f.write(json.dumps(error_result_item) + '\n')
            logging.info(f"Appended {len(data_batch_info)} error results from final batch to {output_path}")
            
    
    print(f"Processing complete. Results saved incrementally to {output_path}")

if __name__ == "__main__":
    main()