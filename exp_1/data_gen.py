import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_tokenizer(base_model_path, lora_model_path):
    logging.info(f"Starting to load tokenizer from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    logging.info("Tokenizer loaded.")

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
    return model, tokenizer

def generate_response(model, tokenizer, prompts_batch, max_length=1024):
    if not prompts_batch:
        return []

    logging.info(f"Generating responses for batch of {len(prompts_batch)} prompts. First prompt (50 chars): '{prompts_batch[0][:50]}...'")
    logging.info(f"Input device: {model.device}, preparing to move inputs to device.")
    
    # Tokenize the batch of prompts. Use padding to handle sequences of different lengths.
    if tokenizer.pad_token is None:
        logging.warning("Tokenizer does not have a pad_token_id. Using eos_token_id as pad_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(
        prompts_batch, 
        return_tensors="pt", 
        padding=True, # Pad to the longest sequence in the batch
        truncation=True, # Truncate sequences longer than model max length 
        max_length=model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 2048 # Use model's max length for truncation
    ).to(model.device)
    logging.info("Batch of inputs tokenized, padded, truncated, and moved to model device.")
    
    with torch.no_grad():
        logging.info("Starting batched model.generate()...")
        outputs = model.generate(
            **inputs,
            max_length=max_length, # This is the max_length for the *generation* part
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id # Important for generation to know when to stop/ignore padding
        )
        logging.info("Batched model.generate() completed.")
    
    logging.info("Decoding batch of responses...")
    
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    logging.info("Batch of responses decoded.")
    return responses

def main():
    # Configuration
    BATCH_SIZE = 8 # Define batch size, tune based on GPU memory
    base_model_path = "meta-llama/Llama-3.1-8B-Instruct"
    lora_model_path = "/scratch-shared/mschaffelder/Data/ft_models/lora_llama_8b_single_v6/checkpoint-1684"
    test_data_path = "/scratch-shared/mschaffelder/Data/Finetuning/Dolly/dolly_test.jsonl"
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(base_model_path, lora_model_path)
    
    # Create output directory if it doesn't exist
    output_dir = "/scratch-shared/mschaffelder/Data/exp_1/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process test data
    logging.info("Starting to process test data...")
    results = []
    prompts_batch = []
    data_batch_info = [] # To store original data items for associating responses
    
    with open(test_data_path, 'r') as f:
        logging.info(f"Opened test data file: {test_data_path}")
        for i, line in enumerate(f):
            logging.info(f"Reading line {i+1} from test data...")
            try:
                data = json.loads(line)
                prompt = data['instruction']
                prompts_batch.append(prompt)
                data_batch_info.append(data) # Store the whole data item
                logging.info(f"Added prompt from line {i+1} to batch.")
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from line {i+1}: {e}")
                continue 
            except KeyError as e:
                logging.error(f"Missing key 'instruction' in line {i+1}: {e}")
                continue
            
            # Process batch when it's full or it's the last line
            if len(prompts_batch) == BATCH_SIZE:
                logging.info(f"Processing batch of {len(prompts_batch)} prompts (up to line {i+1})...")
                try:
                    generated_responses_batch = generate_response(model, tokenizer, prompts_batch)
                    logging.info(f"Batch of responses generated.")
                    for idx, (original_data, gen_response) in enumerate(zip(data_batch_info, generated_responses_batch)):
                        results.append({
                            'instruction': original_data['instruction'],
                            'generated_response': gen_response,
                            'ground_truth': original_data.get('response', '')
                        })
                except Exception as e:
                    logging.error(f"Error during batch generate_response (lines around {i+1}): {e}")
                    # Store error for all items in this failed batch
                    for original_data in data_batch_info:
                         results.append({
                            'instruction': original_data['instruction'],
                            'generated_response': f"ERROR_BATCH: {e}",
                            'ground_truth': original_data.get('response', '')
                        })
                finally:
                    prompts_batch = [] # Clear batch
                    data_batch_info = []

    # Process any remaining prompts in the last batch (if not perfectly divisible by BATCH_SIZE)
    if prompts_batch: # Check if there are any prompts left
        logging.info(f"Processing final batch of {len(prompts_batch)} prompts...")
        try:
            generated_responses_batch = generate_response(model, tokenizer, prompts_batch)
            logging.info(f"Final batch of responses generated.")
            for idx, (original_data, gen_response) in enumerate(zip(data_batch_info, generated_responses_batch)):
                results.append({
                    'instruction': original_data['instruction'],
                    'generated_response': gen_response,
                    'ground_truth': original_data.get('response', '')
                })
        except Exception as e:
            logging.error(f"Error during final batch generate_response: {e}")
            for original_data in data_batch_info:
                 results.append({
                    'instruction': original_data['instruction'],
                    'generated_response': f"ERROR_FINAL_BATCH: {e}",
                    'ground_truth': original_data.get('response', '')
                })
        # No finally needed to clear here as it's the end
            
    # Save results
    output_path = os.path.join(output_dir, 'generation_results_8b_single_v6.jsonl')
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()