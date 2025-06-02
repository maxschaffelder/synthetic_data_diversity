import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

#SYSTEM_PROMPT_PAIRWISE_RANKING = "You are a helpful assistant. Your task is to rank the two provided texts, which are marked with 'A' and 'B'. Please explicitly write which of the two texts is of higher quality by writing 'A' or 'B' in the output, and nothing else."
#SYSTEM_PROMPT_PAIRWISE_RANKING = "You are a helpful assistant. Your task is to rank the two provided texts. Please explicitly write which of the two texts is of higher quality by writing the corresponding number in the output, and nothing else."
#SYSTEM_PROMPT_PAIRWISE_RANKING = "You are a helpful assistant. Your task is to rank the two provided texts. Please explicitly write which of the two texts is of higher quality by writing the corresponding letter in the output, and nothing else."

# Load model and tokenizer
def load_model_and_tokenizer(base_model_path, use_lora=False, lora_model_path=None):
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
    
    if use_lora:
        if lora_model_path is None:
            raise ValueError("lora_model_path must be provided when use_lora is True")
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


# Generate pairwise ranking response
def generate_pairwise_ranking_response(model, tokenizer, prompts_batch, system_prompt, ranking_markers_list, max_length=1024):
    if not prompts_batch:
        return [], [], []

    logging.info(f"Generating rankings for batch of {len(prompts_batch)} pairs of prompts. System prompt: '{system_prompt}'")

    # Tokenize markers once, assuming they are single tokens
    marker1_str = ranking_markers_list[0].upper()
    marker2_str = ranking_markers_list[1].upper()
    actual_token_id_marker1 = -1
    actual_token_id_marker2 = -1

    token_ids_marker1 = tokenizer.encode(marker1_str, add_special_tokens=False)
    if len(token_ids_marker1) == 1:
        actual_token_id_marker1 = token_ids_marker1[0]
    else:
        logging.warning(f"Marker '{marker1_str}' is not a single token (token IDs: {token_ids_marker1}). Cannot reliably calculate its choice probability.")

    token_ids_marker2 = tokenizer.encode(marker2_str, add_special_tokens=False)
    if len(token_ids_marker2) == 1:
        actual_token_id_marker2 = token_ids_marker2[0]
    else:
        logging.warning(f"Marker '{marker2_str}' is not a single token (token IDs: {token_ids_marker2}). Cannot reliably calculate its choice probability.")

    # Apply chat template to each prompt in the batch
    formatted_prompts_for_tokenizer = []
    for user_prompt in prompts_batch:
        messages = [
            {"role": "system", "content": system_prompt},
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
            output_scores=True,
            return_dict_in_generate=True
        )
        logging.info("Batched model.generate() completed.")
    
    
    logging.info("Decoding batch of responses and extracting token probabilities...")
    cleaned_responses = []
    all_token_probabilities = [] # Stores probabilities of the chosen tokens in the generated sequence
    marker_choice_probabilities = [] # Stores probabilities of the predefined markers at the first step

    # outputs.sequences contains the full sequence (input_ids + generated_ids) of shape (batch_size, sequence_length)
    # outputs.scores is a tuple of tensors of logits for each generated token. Len is num_generated_tokens.
    # Each element of outputs.scores has shape (batch_size, vocab_size).
    
    for i in range(len(prompts_batch)):
        input_token_len = inputs.input_ids[i].shape[0]
        
        # Get generated token IDs for the current item from outputs.sequences
        generated_token_ids = outputs.sequences[i][input_token_len:]

        # Calculate probabilities for the specified markers at the first decision point
        current_item_marker_choice_probs = {marker1_str: None, marker2_str: None}
        if outputs.scores and actual_token_id_marker1 != -1 and actual_token_id_marker2 != -1:
            # scores[0] corresponds to the logits of the first generated token
            first_token_logits = outputs.scores[0][i, :]
            first_token_softmax_probs = torch.softmax(first_token_logits, dim=-1)
            
            prob_marker1 = first_token_softmax_probs[actual_token_id_marker1].item()
            prob_marker2 = first_token_softmax_probs[actual_token_id_marker2].item()
            current_item_marker_choice_probs = {marker1_str: prob_marker1, marker2_str: prob_marker2}
        elif not outputs.scores:
            logging.warning(f"No scores available for item {i}, cannot calculate marker choice probabilities.")
        else: # One or both markers were not single tokens
            logging.warning(f"Cannot calculate marker choice probabilities for item {i} due to multi-token marker(s).")
        marker_choice_probabilities.append(current_item_marker_choice_probs)

        if generated_token_ids.nelement() == 0:
            logging.warning(f"No new tokens generated for prompt: '{prompts_batch[i][:50]}...'. Input length: {input_token_len}, Output length: {outputs.sequences[i].shape[0]}")
            cleaned_responses.append("") 
            all_token_probabilities.append([]) # Add empty list for no generation
        else:
            # Decode only the generated tokens
            cleaned_response = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            cleaned_responses.append(cleaned_response.lstrip()) 
            logging.debug(f"Cleaned response (token-based stripping). Prompt: '{prompts_batch[i][:50]}...', Generated: '{cleaned_response[:100]}...'")

            # Calculate token probabilities for this response
            current_response_token_probs = []
            # outputs.scores[k_step] are logits for (k_step+1)-th generated token for the ENTIRE BATCH.
            for k_step, token_id_at_step_k in enumerate(generated_token_ids):
                # Logits for the current item 'i' at generation step 'k_step'
                logits_for_item_i_at_step_k = outputs.scores[k_step][i, :] 
                probabilities_for_item_i_at_step_k = torch.softmax(logits_for_item_i_at_step_k, dim=-1)
                prob_of_chosen_token = probabilities_for_item_i_at_step_k[token_id_at_step_k].item()
                current_response_token_probs.append(prob_of_chosen_token)
            all_token_probabilities.append(current_response_token_probs)

    logging.info("Input tokens stripped from responses and probabilities extracted.")
    return cleaned_responses, all_token_probabilities, marker_choice_probabilities
