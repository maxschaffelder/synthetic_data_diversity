import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

SYSTEM_PROMPT_RELATIVE_RANKING = "You are a helpful assistant. Your task is to rank the two provided texts, which are marked with 'A' and 'B'. Please explicitly write which of the two texts is of higher quality by writing 'A' or 'B' in the output."
SYSTEM_PROMPT_ABSOLUTE_RATING = "You are a helpful assistant. Your task is to rate the provided text in terms of overall quality. The rating should be from one to five, with one being the lowest quality and five being the highest quality. Please write the number only, no other text."

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


# Rank responses
def generate_relative_ranking_response(model, tokenizer, prompts_batch, max_length=1024):
    if not prompts_batch:
        return []

    logging.info(f"Generating rankings for batch of {len(prompts_batch)} pairs of prompts. System prompt: '{SYSTEM_PROMPT_RANKING}'")

    # Apply chat template to each prompt in the batch
    formatted_prompts_for_tokenizer = []
    for user_prompt in prompts_batch:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_RANKING},
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



# Generate ranking response
def generate_ranking_response(model, tokenizer, prompts_batch, max_length=1024):
    if not prompts_batch:
        return []

    logging.info(f"Generating rankings for batch of {len(prompts_batch)} pairs of prompts. System prompt: '{SYSTEM_PROMPT_RANKING}'")

    # Apply chat template to each prompt in the batch
    formatted_prompts_for_tokenizer = []
    for user_prompt in prompts_batch:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_RANKING},
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



# Generate absolute rating response
def generate_absolute_rating_response(model, tokenizer, prompts_batch, max_length=1024):
    if not prompts_batch:
        return []

    logging.info(f"Generating absolute ratings for batch of {len(prompts_batch)} prompts. System prompt: '{SYSTEM_PROMPT_ABSOLUTE_RATING}'")

    # Apply chat template to each prompt in the batch
    formatted_prompts_for_tokenizer = []
    for user_prompt in prompts_batch:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_ABSOLUTE_RATING},
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



# Generate ranking response
def generate_ranking_response(model, tokenizer, prompts_batch, max_length=1024):
    if not prompts_batch:
        return []

    logging.info(f"Generating rankings for batch of {len(prompts_batch)} pairs of prompts. System prompt: '{SYSTEM_PROMPT_RANKING}'")

    # Apply chat template to each prompt in the batch
    formatted_prompts_for_tokenizer = []
    for user_prompt in prompts_batch:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_RANKING},
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