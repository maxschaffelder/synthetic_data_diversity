import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import logging
import argparse

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

SYSTEM_PROMPT = "You are a helpful assistant."

def generate_response(model, tokenizer, prompts_batch, max_new_tokens=1024):
    if not prompts_batch:
        return []

    logging.info(f"Generating responses for batch of {len(prompts_batch)} prompts. System prompt: '{SYSTEM_PROMPT}'")

    # Apply chat template to each prompt in the batch
    formatted_prompts_for_tokenizer = []
    for user_prompt in prompts_batch:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        # tokenize=False to get the string, add_generation_prompt=True to prepare for assistant generation
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        formatted_prompts_for_tokenizer.append(formatted_prompt)
    
    logging.info(f"First formatted prompt for tokenizer (first 100 chars): '{formatted_prompts_for_tokenizer[0][:100]}...'")
    logging.info(f"Input device: {model.device}, preparing to move inputs to device.")
    
    inputs = tokenizer(
        formatted_prompts_for_tokenizer, # Tokenize the list of formatted strings 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 2048 
    ).to(model.device)
    logging.info("Batch of inputs tokenized, padded, truncated, and moved to model device.")
    
    with torch.no_grad():
        logging.info("Starting batched model.generate()...")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id  # Explicitly set EOS token
        )
        logging.info("Batched model.generate() completed.")
    
    logging.info("Decoding batch of responses by stripping the prompt string...")
    cleaned_responses = []
    # Decode the full output sequence and then remove the original prompt string.
    # This is more robust than token-level manipulation.
    for i, (full_output, prompt_str) in enumerate(zip(outputs, formatted_prompts_for_tokenizer)):
        # Decode the entire generated sequence
        decoded_output = tokenizer.decode(full_output, skip_special_tokens=True)
        
        # The prompt string as it was fed to the tokenizer
        prompt_as_decoded = tokenizer.decode(tokenizer.encode(prompt_str), skip_special_tokens=True)

        # Check if the decoded output starts with the prompt
        if decoded_output.startswith(prompt_as_decoded):
            # Strip the prompt part to get only the generated response
            response = decoded_output[len(prompt_as_decoded):].lstrip()
            cleaned_responses.append(response)
            logging.debug(f"Cleaned response (string-based stripping). Prompt: '{prompts_batch[i][:50]}...', Generated: '{response[:100]}...'")
        else:
            logging.warning(f"Could not strip prompt for item {i}. The decoded output did not start with the expected prompt. Fallback to token-based stripping.")
            # Fallback to the original, less reliable token-based method if string stripping fails
            input_token_len = len(inputs.input_ids[i])
            generated_token_ids = full_output[input_token_len:]
            response = tokenizer.decode(generated_token_ids, skip_special_tokens=True).lstrip()
            cleaned_responses.append(response)

    logging.info("Prompt strings stripped from responses.")
    return cleaned_responses

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate responses using a finetuned Llama model.")
    parser.add_argument("--lora_model_path", type=str, required=True, help="Path to the LoRA model checkpoint.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test data JSONL file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generation results.")
    parser.add_argument("--base_model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Path or HuggingFace name of the base model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation.")
    
    args = parser.parse_args()
    # --- End Argument Parsing ---

    # Configuration
    BATCH_SIZE = args.batch_size
    base_model_path = args.base_model_path
    lora_model_path = args.lora_model_path
    test_data_path = args.test_data_path
    output_dir = args.output_dir
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Construct output path using the output_dir and a fixed filename for now
    output_filename = f"output_{os.path.basename(lora_model_path)}.jsonl"
    output_path = os.path.join(output_dir, output_filename)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(base_model_path, lora_model_path)

    # Clear/Create the output file at the beginning of this run
    with open(output_path, 'w') as f:
        logging.info(f"Output file {output_path} created/cleared.")
    
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

                    results_to_write = []
                    for original_data, gen_response in zip(data_batch_info, generated_responses_batch):
                        result_item = {
                            'instruction': original_data['instruction'],
                            'response_model': gen_response,
                            'response_human': original_data.get('response_human', '')
                        }
                        results_to_write.append(json.dumps(result_item))
                    
                    # Write all results for the batch in one go
                    with open(output_path, 'a') as f:
                        f.write('\n'.join(results_to_write) + '\n')
                    logging.info(f"Appended {len(results_to_write)} results from batch to {output_path}")

                except Exception as e:
                    logging.error(f"Error during batch generate_response (lines around {i+1}): {e}")
                    # Store error for all items in this failed batch by writing them out
                    error_results_to_write = []
                    for original_data in data_batch_info:
                        error_result_item = {
                            'instruction': original_data['instruction'],
                            'response_model': f"ERROR_BATCH: {e}",
                            'response_human': original_data.get('response_human', '')
                        }
                        error_results_to_write.append(json.dumps(error_result_item))
                    with open(output_path, 'a') as f:
                        f.write('\n'.join(error_results_to_write) + '\n')
                    logging.info(f"Appended {len(data_batch_info)} error results from batch to {output_path}")
                finally:
                    prompts_batch = [] 
                    data_batch_info = []

    if prompts_batch: 
        logging.info(f"Processing final batch of {len(prompts_batch)} prompts...")
        try:
            generated_responses_batch = generate_response(model, tokenizer, prompts_batch)
            logging.info(f"Final batch of responses generated.")

            results_to_write = []
            for original_data, gen_response in zip(data_batch_info, generated_responses_batch):
                result_item = {
                    'instruction': original_data['instruction'],
                    'response_model': gen_response,
                    'response_human': original_data.get('response_human', '')
                }
                results_to_write.append(json.dumps(result_item))

            # Write all results for the final batch in one go
            with open(output_path, 'a') as f:
                f.write('\n'.join(results_to_write) + '\n')
            logging.info(f"Appended {len(results_to_write)} results from final batch to {output_path}")
        except Exception as e:
            logging.error(f"Error during final batch generate_response: {e}")
            # Store error for all items in this failed batch by writing them out
            error_results_to_write = []
            for original_data in data_batch_info:
                error_result_item = {
                    'instruction': original_data['instruction'],
                    'response_model': f"ERROR_FINAL_BATCH: {e}",
                    'response_human': original_data.get('response_human', '')
                }
                error_results_to_write.append(json.dumps(error_result_item))
            with open(output_path, 'a') as f:
                f.write('\n'.join(error_results_to_write) + '\n')
            logging.info(f"Appended {len(data_batch_info)} error results from final batch to {output_path}")
            
    
    print(f"Processing complete. Results saved incrementally to {output_path}")

if __name__ == "__main__":
    main()