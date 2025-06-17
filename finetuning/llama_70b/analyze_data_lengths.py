import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from collections import Counter
import argparse

def format_text_like_training(instruction, response, system_prompt=None):
    """Format text exactly like the training script does"""
    if system_prompt:
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
    else:
        text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
    return text

def load_and_analyze_data(file_path, system_prompt="You are a helpful assistant."):
    """Load JSONL data and analyze token lengths"""
    
    print(f"Loading data from: {file_path}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")
    except Exception as e:
        print(f"Failed to load Llama tokenizer, using GPT-2 as fallback: {e}")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Load data
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed line {line_num}: {e}")
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return
    
    print(f"Loaded {len(data)} examples")
    
    if len(data) == 0:
        print("No data found!")
        return
    
    # Analyze lengths
    lengths = []
    instruction_lengths = []
    response_lengths = []
    total_lengths = []
    
    print("Analyzing token lengths...")
    
    for i, item in enumerate(data):
        if i % 100 == 0 and i > 0:
            print(f"Processed {i}/{len(data)} examples...")
        
        instruction = item.get('instruction', '')
        response = item.get('response_model', '')
        
        # Format like training script
        formatted_text = format_text_like_training(instruction, response, system_prompt)
        
        # Tokenize
        tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
        instruction_tokens = tokenizer.encode(instruction, add_special_tokens=False)
        response_tokens = tokenizer.encode(response, add_special_tokens=False)
        
        lengths.append(len(tokens))
        instruction_lengths.append(len(instruction_tokens))
        response_lengths.append(len(response_tokens))
        total_lengths.append(len(instruction_tokens) + len(response_tokens))
    
    # Calculate statistics
    lengths = np.array(lengths)
    instruction_lengths = np.array(instruction_lengths)
    response_lengths = np.array(response_lengths)
    total_lengths = np.array(total_lengths)
    
    print("\n" + "="*60)
    print("TOKEN LENGTH ANALYSIS")
    print("="*60)
    
    print(f"\nFULL FORMATTED TEXT (with special tokens):")
    print(f"  Mean length: {lengths.mean():.1f} tokens")
    print(f"  Median length: {np.median(lengths):.1f} tokens")
    print(f"  Min length: {lengths.min()} tokens")
    print(f"  Max length: {lengths.max()} tokens")
    print(f"  Std dev: {lengths.std():.1f} tokens")
    
    print(f"\nINSTRUCTIONS ONLY:")
    print(f"  Mean length: {instruction_lengths.mean():.1f} tokens")
    print(f"  Median length: {np.median(instruction_lengths):.1f} tokens")
    print(f"  Max length: {instruction_lengths.max()} tokens")
    
    print(f"\nRESPONSES ONLY:")
    print(f"  Mean length: {response_lengths.mean():.1f} tokens")
    print(f"  Median length: {np.median(response_lengths):.1f} tokens")
    print(f"  Max length: {response_lengths.max()} tokens")
    
    # Percentile analysis
    print(f"\nPERCENTILE ANALYSIS (Full formatted text):")
    percentiles = [50, 75, 90, 95, 99, 99.5]
    for p in percentiles:
        value = np.percentile(lengths, p)
        percentage_fit = (lengths <= value).mean() * 100
        print(f"  {p:4.1f}th percentile: {value:4.0f} tokens ({percentage_fit:5.1f}% of data fits)")
    
    # Truncation analysis
    print(f"\nTRUNCATION ANALYSIS:")
    for max_len in [512, 1024, 1536, 2048, 3072, 4096]:
        percentage_fit = (lengths <= max_len).mean() * 100
        num_truncated = (lengths > max_len).sum()
        print(f"  max_seq_length={max_len:4d}: {percentage_fit:5.1f}% fit, {num_truncated:4d} truncated")
    
    # Find examples that would be truncated at different lengths
    print(f"\nEXAMPLES OF LONG SEQUENCES:")
    long_indices = np.argsort(lengths)[-5:]  # Top 5 longest
    for idx in reversed(long_indices):
        instruction = data[idx].get('instruction', '')[:100]
        response = data[idx].get('response_model', '')[:100]
        print(f"  Length {lengths[idx]:4d}: Instr='{instruction}...' Resp='{response}...'")
    
    # Recommendations
    print(f"\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if np.percentile(lengths, 95) <= 1024:
        print("✅ max_seq_length=1024 should work well (covers 95%+ of your data)")
    elif np.percentile(lengths, 90) <= 1024:
        print("⚠️  max_seq_length=1024 covers 90%+ but consider 1536 or 2048")
    elif np.percentile(lengths, 95) <= 2048:
        print("📈 Recommend max_seq_length=2048 for good coverage")
    else:
        print("🚨 Your data has very long sequences. Consider max_seq_length=4096 or data preprocessing")
    
    # Memory impact
    print(f"\nMEMORY IMPACT ESTIMATES:")
    print(f"  1024 tokens: ~Low memory usage")
    print(f"  2048 tokens: ~2x memory usage vs 1024")
    print(f"  4096 tokens: ~4x memory usage vs 1024")
    
    return lengths, instruction_lengths, response_lengths

def create_plots(lengths, instruction_lengths, response_lengths, output_dir="."):
    """Create visualization plots"""
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Token Length Analysis', fontsize=16)
        
        # Full length histogram
        axes[0,0].hist(lengths, bins=50, alpha=0.7, color='blue')
        axes[0,0].set_title('Full Formatted Text Lengths')
        axes[0,0].set_xlabel('Tokens')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].axvline(1024, color='red', linestyle='--', label='1024 tokens')
        axes[0,0].axvline(2048, color='orange', linestyle='--', label='2048 tokens')
        axes[0,0].legend()
        
        # Instruction vs Response scatter
        axes[0,1].scatter(instruction_lengths, response_lengths, alpha=0.5, s=1)
        axes[0,1].set_title('Instructions vs Response Lengths')
        axes[0,1].set_xlabel('Instruction Tokens')
        axes[0,1].set_ylabel('Response Tokens')
        
        # Box plot comparison
        axes[1,0].boxplot([instruction_lengths, response_lengths, lengths], 
                         labels=['Instructions', 'Responses', 'Full Text'])
        axes[1,0].set_title('Length Distribution Comparison')
        axes[1,0].set_ylabel('Tokens')
        
        # Cumulative distribution
        sorted_lengths = np.sort(lengths)
        cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
        axes[1,1].plot(sorted_lengths, cumulative * 100)
        axes[1,1].set_title('Cumulative Distribution')
        axes[1,1].set_xlabel('Tokens')
        axes[1,1].set_ylabel('Percentage of Data')
        axes[1,1].axvline(1024, color='red', linestyle='--', label='1024 tokens')
        axes[1,1].axvline(2048, color='orange', linestyle='--', label='2048 tokens')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = f"{output_dir}/token_length_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Plots saved to: {plot_path}")
        
        # Try to show plot if running interactively
        try:
            plt.show()
        except:
            print("   (Plot display not available in this environment)")
            
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze token lengths in finetuning data")
    parser.add_argument("--file", type=str, required=True, help="Path to JSONL file")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.", 
                       help="System prompt to use (should match training)")
    parser.add_argument("--plot", action="store_true", help="Create visualization plots")
    
    args = parser.parse_args()
    
    lengths, instruction_lengths, response_lengths = load_and_analyze_data(
        args.file, args.system_prompt
    )
    
    if args.plot and lengths is not None:
        create_plots(lengths, instruction_lengths, response_lengths)

if __name__ == "__main__":
    main() 