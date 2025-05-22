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
        torch_dtype=torch.float16,
        device_map="auto"
    )
    logging.info(f"Base model loaded. Original vocab size: {model.config.vocab_size}")
    
    # Resize token embeddings if necessary
    expected_vocab_size = 128257
    if model.config.vocab_size < expected_vocab_size:
        logging.info(f"Resizing token embeddings from {model.config.vocab_size} to {expected_vocab_size}...")
        model.resize_token_embeddings(expected_vocab_size)
        logging.info(f"Token embeddings resized. New vocab size: {model.config.vocab_size}")
        # Note: The messages "The new embeddings will be initialized..." usually appear during the call above.
    else:
        logging.info("Token embeddings do not need resizing.")
    
    logging.info(f"Starting to load LoRA weights from: {lora_model_path}")
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, lora_model_path)
    logging.info("LoRA weights loaded and merged with the base model.")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=2048):
    logging.info(f"Generating response for prompt (first 50 chars): '{prompt[:50]}...'")
    logging.info(f"Input device: {model.device}, preparing to move inputs to device.")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    logging.info("Inputs tokenized and moved to model device.")
    
    with torch.no_grad():
        logging.info("Starting model.generate()...")
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        logging.info("model.generate() completed.")
    
    logging.info("Decoding response...")
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.info("Response decoded.")
    return response

def main():
    # Model paths
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
    
    with open(test_data_path, 'r') as f:
        logging.info(f"Opened test data file: {test_data_path}")
        for i, line in enumerate(f):
            logging.info(f"Processing line {i+1} from test data...")
            try:
                data = json.loads(line)
                prompt = data['instruction']
                logging.info(f"Instruction (prompt) extracted from line {i+1}.")
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from line {i+1}: {e}")
                continue # Skip to the next line
            except KeyError as e:
                logging.error(f"Missing key 'instruction' in line {i+1}: {e}")
                continue # Skip to the next line
            
            # Generate response
            logging.info(f"Calling generate_response for line {i+1}...")
            try:
                response = generate_response(model, tokenizer, prompt)
                logging.info(f"Response generated for line {i+1}.")
            except Exception as e:
                logging.error(f"Error during generate_response for line {i+1}, prompt: '{prompt[:100]}...': {e}")
                # Optionally, decide if you want to store a placeholder or skip
                results.append({
                    'instruction': prompt,
                    'generated_response': f"ERROR: {e}",
                    'ground_truth': data.get('response', '')
                })
                continue # Skip to next item
            
            # Store results
            result = {
                'instruction': prompt,
                'generated_response': response,
                'ground_truth': data.get('response', '')
            }
            results.append(result)
    
    # Save results
    output_path = os.path.join(output_dir, 'generation_results_8b_single_v6.jsonl')
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()