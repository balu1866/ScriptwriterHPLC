
import re
import json
import random
import tiktoken
import os
import prompts
import pdb

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from dotenv import load_dotenv
from transformers import pipeline


model_name = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
paraphraser = pipeline("text-generation", model = model, tokenizer = tokenizer)

import pdb
import prompts
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
load_dotenv()

def clean_text(text):
    return re.sub(r'[\s]+|[-\\*]', ' ', text.strip())


def generate_variants(text, max_variants=1):
    """Generates paraphrased variants of the input text."""
    prompt = prompts.VARIANCE_PROMPT.format(text=text)
    response = paraphraser(
        prompt,
        max_new_tokens=400,
        do_sample=True,
        num_return_sequences=max_variants,
        temperature=0.8,         # Increased for creative language
        top_p=0.8,               # Diverse phrasing and richer details
        repetition_penalty=1.5,  # Breaks repetitive phrasing
        length_penalty=3.0,
        return_full_text=False

    )
    breakpoint()
    texts = [(clean_text(text["generated_text"])) for text in response]
    return texts


def split_into_chunks(text):
    """Splits text into smaller chunks while preserving sentence structure."""
    encoding = tiktoken.get_encoding(os.getenv("TOKEN_ENCODING"))
    max_chunk_size=250

    # Encode the entire text once
    tokens = encoding.encode(text)
    chunks = []

    # Process tokens into chunks
    for i in range(0, len(tokens), max_chunk_size):
        chunk = tokens[i:i + max_chunk_size]
        chunks.append(encoding.decode(chunk))

    return chunks



with open("az.txt", 'r', encoding='ISO-8859-1') as file:
    text = file.read()

# Dataset collection
dataset = []

story_splits = re.split(r'(?=\n[A-Z][A-Z ]+\n)', text)


counter = 0
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
                    "prompt": f"Complete the following text:{clean_text(p)}",
                    "completion": clean_text(c),
                    "metadata": metadata
                })
        if(len(dataset) % 50 == 0):
            counter += len(dataset)
            with open("dataset.jsonl", 'a', encoding='utf-8') as outfile:
                # Write the dataset to a JSONL file
                for entry in dataset:
                    outfile.write(json.dumps(entry) + "\n")
                outfile.close()
                dataset = []  # Clear the dataset after writing

            print(f"Generated {counter} entries so far.")


print(f"Dataset successfully created with {len(dataset)} entries.")
