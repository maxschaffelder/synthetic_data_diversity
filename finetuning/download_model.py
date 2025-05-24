from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.1-70B-Instruct"
# It's good practice to make the cache directory more specific if multiple users or projects use it
# Or if you might cache different versions/types of models.
# Using the model name in the path can help.
cache_dir_base = "/scratch-shared/mschaffelder/hf_cache"
specific_model_cache_path = cache_dir_base + "/" + model_name.replace("/", "_")

print(f"Ensuring cache directory exists: {specific_model_cache_path}")
import os
os.makedirs(specific_model_cache_path, exist_ok=True)

print(f"Downloading and caching tokenizer for {model_name} to {specific_model_cache_path}")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir_base) # Download to general cache
    tokenizer.save_pretrained(specific_model_cache_path) # Save to specific model path
    print(f"Tokenizer saved to {specific_model_cache_path}")
except Exception as e:
    print(f"Error downloading/saving tokenizer: {e}")

print(f"Downloading and caching model {model_name} to {specific_model_cache_path}")
try:
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir_base) # Download to general cache
    model.save_pretrained(specific_model_cache_path) # Save to specific model path
    print(f"Model saved to {specific_model_cache_path}")
except Exception as e:
    print(f"Error downloading/saving model: {e}")

print("Download and caching attempt complete.") 