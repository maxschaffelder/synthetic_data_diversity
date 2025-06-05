import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import logging
import argparse

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_tokenizer(base_model_path):
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
    
    # Align model's pad_token_id with tokenizer's pad_token_id if necessary
    # This is important for the model.generate() method.
    if model.config.pad_token_id is None or model.config.pad_token_id != tokenizer.pad_token_id:
        logging.info(f"Aligning model.config.pad_token_id ({model.config.pad_token_id}) to tokenizer.pad_token_id ({tokenizer.pad_token_id}).")
        model.config.pad_token_id = tokenizer.pad_token_id
        
    return model, tokenizer

SYSTEM_PROMPT = "You are a helpful assistant."

def generate_response(model, tokenizer, prompts_batch, max_length=2048):
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
            max_length=inputs.input_ids.shape[1] + max_length, # max_length is for *new* tokens
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        logging.info("Batched model.generate() completed.")
    
    logging.info("Decoding batch of responses by stripping input tokens...")
    cleaned_responses = []
    # `outputs` contains the full sequence (input_ids + generated_ids)
    # `inputs.input_ids` are the tokenized inputs we sent to the model.
    for i in range(len(prompts_batch)):
        input_token_len = inputs.input_ids[i].shape[0]
        # The output tokens for the i-th item in the batch
        output_tokens_for_item = outputs[i]
        
        # Assuming the input prompt tokens are at the beginning of the output tokens:
        generated_token_ids = output_tokens_for_item[input_token_len:]

        if generated_token_ids.nelement() == 0:
            logging.warning(f"No new tokens generated for prompt: '{prompts_batch[i][:50]}...'. Input length: {input_token_len}, Output length: {output_tokens_for_item.shape[0]}")
            cleaned_responses.append("") # Append empty string for no new generation
        else:
            # Decode only the generated tokens
            cleaned_response = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            cleaned_responses.append(cleaned_response.lstrip()) # lstrip to remove leading whitespace from model's actual output
            logging.debug(f"Cleaned response (token-based stripping). Prompt: '{prompts_batch[i][:50]}...', Generated: '{cleaned_response[:100]}...'")

    logging.info("Input tokens stripped from responses.")
    return cleaned_responses


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate responses using a finetuned Llama model.")
    parser.add_argument("--input_data_path", type=str, required=True, help="Path to the input data JSONL file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generation results.")
    parser.add_argument("--base_model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Path or HuggingFace name of the base model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation.")
    parser.add_argument("--output_filename", type=str, default="judgement_results.jsonl", help="Output filename.")
    
    args = parser.parse_args()
    # --- End Argument Parsing ---

    # Configuration
    BATCH_SIZE = args.batch_size
    base_model_path = args.base_model_path
    input_data_path = args.input_data_path
    output_path = args.output_path
    output_filename = args.output_filename
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Construct output path using the output_dir and a fixed filename for now
    output_path = os.path.join(output_path, output_filename)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(base_model_path)

    # Clear/Create the output file at the beginning of this run
    with open(output_path, 'w') as f:
        logging.info(f"Output file {output_path} created/cleared.")
    
    # Process test data
    logging.info("Starting to process test data...")
    prompts_batch = []
    data_batch_info = [] 
    
    with open(input_data_path, 'r') as f:
        logging.info(f"Opened test data file: {input_data_path}")
        for i, line in enumerate(f):
            logging.info(f"Reading line {i+1} from test data...")
            try:
                data = json.loads(line)
                prompt = data['judge_input']
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
                            'judge_input': original_data['judge_input'],
                            'judgement': gen_response,
                            "instruction": original_data['instruction'],
                            "response_model": original_data['response_model'],
                            "category": original_data['category'],
                            "judge_model": base_model_path.split("/")[-1]
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
                                'judge_input': original_data['judge_input'],
                                'judgement': f"ERROR_BATCH: {e}",
                                "instruction": original_data['instruction'],
                                "response_model": original_data['response_model'],
                                "category": original_data['category'],
                                "judge_model": base_model_path.split("/")[-1]
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
                    'judge_input': original_data['judge_input'],
                    'judgement': gen_response,
                    "instruction": original_data['instruction'],
                    "response_model": original_data['response_model'],
                    "category": original_data['category'],
                    "judge_model": base_model_path.split("/")[-1]
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
                        'judge_input': original_data['judge_input'],
                        'judgement': f"ERROR_FINAL_BATCH: {e}",
                        "instruction": original_data['instruction'],
                        "response_model": original_data['response_model'],
                        "category": original_data['category'],
                        "judge_model": base_model_path.split("/")[-1]
                    }
                    f.write(json.dumps(error_result_item) + '\n')
            logging.info(f"Appended {len(data_batch_info)} error results from final batch to {output_path}")
            
    
    print(f"Processing complete. Results saved incrementally to {output_path}")

if __name__ == "__main__":
    main()