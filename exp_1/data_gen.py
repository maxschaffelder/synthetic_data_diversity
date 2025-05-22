import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def load_model_and_tokenizer(base_model_path, lora_model_path):
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Resize token embeddings if necessary
    if model.config.vocab_size < 128257:
        model.resize_token_embeddings(128257)
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, lora_model_path)
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
