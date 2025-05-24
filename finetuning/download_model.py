from huggingface_hub import snapshot_download
import os

model_name = "meta-llama/Llama-3.1-70B-Instruct"
cache_dir_base = "/scratch-shared/mschaffelder/hf_cache" # General Hugging Face cache
specific_model_download_path = os.path.join(cache_dir_base, model_name.replace("/", "_"))

print(f"Ensuring download directory exists: {specific_model_download_path}")
os.makedirs(specific_model_download_path, exist_ok=True)

print(f"Downloading model and tokenizer files for {model_name} to {specific_model_download_path}")
try:
    snapshot_download(
        repo_id=model_name,
        local_dir=specific_model_download_path,
        local_dir_use_symlinks=False,  # Important for shared filesystems, store actual files
        resume_download=True,
        # token=os.environ.get("HF_TOKEN") # Assuming token is set by huggingface-cli login
    )
    print(f"Files for {model_name} downloaded to {specific_model_download_path}")
except Exception as e:
    print(f"Error downloading files: {e}")

print("Download script attempt complete.") 