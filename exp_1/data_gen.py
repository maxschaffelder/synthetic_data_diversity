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
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
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
    print("Processing test data...")
    results = []
    
    with open(test_data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            prompt = data['instruction']
            
            # Generate response
            response = generate_response(model, tokenizer, prompt)
            
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