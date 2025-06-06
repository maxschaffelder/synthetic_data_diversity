import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str, required=True, help="Path to the training data JSONL file.")
parser.add_argument("--val_path", type=str, required=True, help="Path to the validation data JSONL file.")
parser.add_argument("--response_key", type=str, default="response_model", help="The key in the JSONL file that contains the response.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model and logs.")
args = parser.parse_args()

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-70B",
    max_seq_length=max_seq_length,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"], 
    use_rslora=True,
    use_gradient_checkpointing="unsloth"
)


tokenizer = get_chat_template(
    tokenizer,
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    chat_template="chatml",
)

def apply_template(examples):
    messages = examples["conversations"]
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": text}

dataset = load_dataset("mlabonne/FineTome-100k", split="train")
dataset = dataset.map(apply_template, batched=True)


trainer=SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        seed=0,
    ),
)

print("Starting training...")
trainer.train()
print("Training finished.")

print("Running inference test...")

model = FastLanguageModel.for_inference(model)

messages = [
    {"from": "human", "value": "What is the capital of Germany?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True)


print("\n\nSaving model...")
model.save_pretrained_merged(args.output_dir, tokenizer, save_method="merged_16bit")
print(f"Model saved to '{args.output_dir}'")
