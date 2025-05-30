import json
import cohere
import google.generativeai as genai
from groq import Groq
from openai import OpenAI
from mistralai import Mistral

SYSTEM_PROMPT_SUMMARY = "You are a helpful assistant. Your task is to summarize the provided text. The summary should be approximately 50 words long."

# Data generation function using API

def create_synthetic_data(model, input_path, output_path, split=None):

    if model == "gemini-2.0-flash":
        # Gemini API
        with open("../../Keys/gemini_key.txt", "r") as f:
            key = f.read().strip()

        # Configure Gemini API
        genai.configure(api_key=key)

    elif model == "llama-3.1-8b-instant" or model == "gemma2-9b-it":
        # Groq API
        with open("../../Keys/groq_key.txt", "r") as f:
            key = f.read().strip()

        client = Groq(api_key=key)

    elif model == "meta-llama/Meta-Llama-3.1-70B-Instruct" or model == "Qwen/Qwen2.5-72B-Instruct" or model == "deepseek-ai/DeepSeek-V3" or model == "meta-llama/Meta-Llama-3.1-405B-Instruct":
        # Deepinfra API
        with open("../../Keys/deepinfra_key.txt", "r") as f:
            key = f.read().strip()

        client = OpenAI(
            api_key=key,
            base_url="https://api.deepinfra.com/v1/openai",
        )

    elif model == "command-r-plus":
        # Cohere API
        with open("../../Keys/cohere_key_paid.txt", "r") as f:
            key = f.read().strip()
        
        client = cohere.ClientV2(key)

    elif model == "mistral-large-latest":
        # Mistral API
        with open("../../Keys/mistral_key.txt", "r") as f:
            key = f.read().strip()

        client = Mistral(api_key=key)

    elif model == "gpt-4o":
        # OpenAI API
        with open("../../Keys/openai_key.txt", "r") as f:
            key = f.read().strip()

        client = OpenAI(
            api_key=key,
        )



    # Read lines from the input file
    with open(input_path, "r", encoding="utf-8") as fin:
        all_lines = fin.readlines()

    # If `split` is set, slice accordingly
    if split is not None:
        all_lines = all_lines[split[0]:split[1]] 

    # Open output in append mode
    with open(output_path, "a", encoding="utf-8") as fout:
        for i, line in enumerate(all_lines):
            try:
                data = json.loads(line)
                instruction = data["article"]


                if model == "gemini-2.0-flash":

                    # Call Gemini API
                    generation_config = {"max_output_tokens": 1024, "temperature": 0.7, "top_p": 0.9}
                    model_instance = genai.GenerativeModel(model, system_instruction=SYSTEM_PROMPT_SUMMARY)
                    response = model_instance.generate_content(instruction, generation_config=generation_config)


                    if response.candidates:
                        generated_text = response.candidates[0].content.parts[0].text
                    else:
                        generated_text = "Error: No response received"


                elif model == "command-r-plus":

                    response = client.chat(
                        model=model, 
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT_SUMMARY},
                            {"role": "user", "content": instruction}],
                        max_tokens=1024,
                        temperature=0.7,
                        p=0.9
                    )

                    generated_text = response.message.content[0].text

                elif model == "mistral-large-latest":
                    response = client.chat.complete(
                        model = model,
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT_SUMMARY},
                            {"role": "user", "content": instruction}],
                            max_tokens=1024,
                            temperature=0.7,
                            top_p=0.9
                    )
                    generated_text = response.choices[0].message.content

                else: 
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT_SUMMARY},
                            {"role": "user", "content": instruction}],
                        model=model,
                        temperature=0.7,
                        top_p=0.9,
                        max_tokens=1024
                    )
                    generated_text = chat_completion.choices[0].message.content

                data["response_model"] = generated_text
                data[f"model_name"] = model
                data["system_prompt"] = SYSTEM_PROMPT_SUMMARY



                # Write out the updated data immediately
                fout.write(json.dumps(data, ensure_ascii=False))
                fout.write("\n")

                # Every 100 lines, explicitly flush to disk
                if i % 100 == 0:
                    fout.flush()

            except Exception as e:
                print(f"Error on line {i}: {e}")
                break  # Adjust behavior as needed

    print("Done generating data.")


