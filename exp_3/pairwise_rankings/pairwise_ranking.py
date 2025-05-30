import json
import os
import logging
import sys
sys.path.append('/scratch-shared/mschaffelder')
from code.exp_3.helper_functions_judge import load_model_and_tokenizer, generate_pairwise_ranking_response
import argparse


def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Argument parsing
    parser = argparse.ArgumentParser(description="Run pairwise ranking generation.")
    parser.add_argument("--base_model_path", type=str, help="Path to the base model.")
    parser.add_argument("--lora_model_path", type=str, help="Path to the LoRA model.")
    parser.add_argument("--use_lora", type=bool, help="Whether to use the LoRA model.")
    parser.add_argument("--input_file", type=str, help="Path to the input data.")
    parser.add_argument("--output_file", type=str, help="Path to the output file.")
    
    args = parser.parse_args()  

    # Use parsed arguments
    base_model_path = args.base_model_path
    lora_model_path = args.lora_model_path
    use_lora = args.use_lora
    input_file = args.input_file
    output_file = args.output_file
    
    logging.info(f"Starting script with output file: {output_file}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Configuration
    BATCH_SIZE = 8 
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(base_model_path, use_lora, lora_model_path)


    # Clear/Create the output file at the beginning of this run
    with open(output_file, 'w') as f:
        logging.info(f"Output file {output_file} created/cleared.")
    
    # Process data
    logging.info("Starting to process data...")
    prompts_batch = []
    data_batch_info = [] 
    
    with open(input_file, 'r') as f:
        logging.info(f"Opened input file: {input_file}")
        for i, line in enumerate(f):
            logging.info(f"Reading line {i+1} from input file...")
            try:
                data = json.loads(line)
                prompt = data['summary']
                prompts_batch.append(prompt)
                data_batch_info.append(data) 
                logging.info(f"Added prompt from line {i+1} to batch.")
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from line {i+1}: {e}")
                continue 
            except KeyError as e:
                logging.error(f"Missing key 'summary' in line {i+1}: {e}")
                continue
            
            if len(prompts_batch) == BATCH_SIZE:
                logging.info(f"Processing batch of {len(prompts_batch)} prompts (up to line {i+1})...")
                try:
                    generated_responses_batch, token_probabilities_batch = generate_pairwise_ranking_response(model, tokenizer, prompts_batch)
                    logging.info(f"Batch of responses generated.")
                    for idx, (original_data, gen_response, token_probabilities) in enumerate(zip(data_batch_info, generated_responses_batch, token_probabilities_batch)):
                        # Prepare result item for this entry
                        result_item = {
                            'summary': original_data['summary'],
                            'ranking_output': gen_response,
                            'token_probabilities': token_probabilities,
                            'model_a': original_data['model_a'],
                            'model_b': original_data['model_b']
                        }
                        # Append this single result to the file
                        with open(output_file, 'a') as f:
                            f.write(json.dumps(result_item) + '\n')
                    logging.info(f"Appended {len(generated_responses_batch)} results from batch to {output_file}")
                except Exception as e:
                    logging.error(f"Error during batch generate_response (lines around {i+1}): {e}")
                    # Store error for all items in this failed batch by writing them out
                    with open(output_file, 'a') as f:
                        for original_data in data_batch_info:
                            error_result_item = {
                                'summary': original_data['summary'],
                                'ranking_output': f"ERROR_BATCH: {e}", 
                                'token_probabilities': [],
                                'model_a': original_data['model_a'],
                                'model_b': original_data['model_b']
                            }
                            f.write(json.dumps(error_result_item) + '\n')
                    logging.info(f"Appended {len(data_batch_info)} error results from batch to {output_file}")
                finally:
                    prompts_batch = [] 
                    data_batch_info = []

    if prompts_batch: 
        logging.info(f"Processing final batch of {len(prompts_batch)} prompts...")
        try:
            generated_responses_batch, token_probabilities_batch = generate_pairwise_ranking_response(model, tokenizer, prompts_batch)
            logging.info(f"Final batch of responses generated.")
            for idx, (original_data, gen_response, token_probabilities) in enumerate(zip(data_batch_info, generated_responses_batch, token_probabilities_batch)):
                # Prepare result item for this entry
                result_item = {
                    'summary': original_data['summary'],
                    'ranking_output': gen_response,
                    'token_probabilities': token_probabilities,
                    'model_a': original_data['model_a'],
                    'model_b': original_data['model_b']
                }
                # Append this single result to the file
                with open(output_file, 'a') as f:
                    f.write(json.dumps(result_item) + '\n')
            logging.info(f"Appended {len(generated_responses_batch)} results from final batch to {output_file}")
        except Exception as e:
            logging.error(f"Error during final batch generate_response: {e}")
            # Store error for all items in this failed batch by writing them out
            with open(output_file, 'a') as f:
                for original_data in data_batch_info:
                    error_result_item = {
                        'summary': original_data['summary'],
                        'ranking_output': f"ERROR_FINAL_BATCH: {e}",
                        'token_probabilities': [],
                        'model_a': original_data['model_a'],
                        'model_b': original_data['model_b']
                    }
                    f.write(json.dumps(error_result_item) + '\n')
            logging.info(f"Appended {len(data_batch_info)} error results from final batch to {output_file}")
            
    
    print(f"Processing complete. Results saved incrementally to {output_file}")

if __name__ == "__main__":
    main()