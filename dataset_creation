import re
import json
import random
import openai
import tiktoken
import os
# import prompts
import pdb

from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from transformers import pipeline


model_name = "meta-llama/Llama-3.3-70B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
paraphraser = pipeline("text-generation", model = model, tokenizer = tokenizer)



import pdb
import prompts
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_variants(text, max_variants=1):
    """Generates paraphrased variants of the input text."""
    prompt = prompts.VARIANCE_PROMPT.format(text=text)
    response = paraphraser(
        prompt,
        max_length=250,
        do_sample=True,
        num_return_sequences=max_variants,
        temperature=1.0,         # Increased for creative language
        top_p=0.8,               # Diverse phrasing and richer details
        repetition_penalty=1.5,  # Breaks repetitive phrasing
        length_penalty=3.0
    )
    breakpoint()
    return response



def split_into_chunks(text):
    """Splits text into smaller chunks while preserving sentence structure."""
    encoding = tiktoken.get_encoding(os.getenv("TOKEN_ENCODING"))
    max_chunk_size=100

    # Encode the entire text once
    tokens = encoding.encode(text)
    chunks = []

    # Process tokens into chunks
    for i in range(0, len(tokens), max_chunk_size):
        chunk = tokens[i:i + max_chunk_size]
        chunks.append(encoding.decode(chunk))

    return chunks

def clean_text(text):
    return re.sub(r'[\s]+|[-\\*]', ' ', text.strip())


with open("all_lovecraft_stories.txt", 'r', encoding='ISO-8859-1') as file:
    text = file.read()

# Dataset collection
dataset = []

story_splits = re.split(r'(?=\n[A-Z][A-Z ]+\n)', text)


    # Generate dataset entries
for story in story_splits:
# Split story into paragraphs
    paragraphs = story.strip().split('\n\n')  # Split into paragraphs

    # Split large paragraphs into manageable chunks
    split_paragraphs = [
        chunk
        for para in paragraphs
        for chunk in (split_into_chunks(para) if len(para) > 1000 else [para])
    ]

    # Dataset generation logic
    for idx in range(len(split_paragraphs) - 1):  # Iterate up to the second-to-last paragraph
        prompt_temp = split_paragraphs[idx]
        if len(prompt_temp) < 200:
            continue  # Skip short prompts

        prompt = clean_text(prompt_temp)

        # Combine short completions with the next paragraph
        comp_temp = split_paragraphs[idx + 1]
        if len(comp_temp) < 200 and idx + 2 < len(split_paragraphs):
            comp_temp += " " + split_paragraphs[idx + 2]

        completion = clean_text(comp_temp)

        # Metadata for fine-tuning optimization
        metadata = {
            "genre": "horror",
            "emotion": random.choice(["dread", "fear", "mystery"]),
            "style": random.choice(["descriptive", "poetic", "minimalistic"]),
            "length": "short"
        }

        # Original Record
        dataset.append({
            "prompt": prompt,
            "completion": completion,
            "metadata": metadata
        })

        # Add paraphrased variants for dataset expansion
        prompt_variants = generate_variants(prompt, max_variants=1)
        completion_variants = generate_variants(completion, max_variants=1)


        for p in prompt_variants:
            for c in completion_variants:
                dataset.append({
                    "prompt": clean_text(p),
                    "completion": clean_text(c),
                    "metadata": metadata
                })
        counter = 0
        if(len(dataset) % 170 == 0):
            with open("dataset.jsonl", 'a', encoding='utf-8') as outfile:
                # Write the dataset to a JSONL file
                counter += len(dataset)
                for entry in dataset:
                    outfile.write(json.dumps(entry) + "\n")
                outfile.close()
                dataset = []  # Clear the dataset after writing

            print(f"Generated {counter} entries so far.")


print(f"Dataset successfully created with {len(dataset)} entries.")
