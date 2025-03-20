import openai
import os
import faiss

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
openai_client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
model = SentenceTransformer(os.getenv("EMBEDDING_MODEL"))
index = faiss.read_index(os.getenv("VECTOR_STORE"))

mappings = {}

with open(os.getenv("VECTOR_MAPPING"), "r") as f:
    for line in f:
        idx, text = line.split("||")
        mappings[int(idx)] = text

def get_script(idea):
    query_vector = model.encode([idea])
    D, I = index.search(query_vector, k=5)
    context = [mappings[i] for i in I[0]]

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a successful horror script writer inspired by H. P. Lovecraft."},
            {"role": "user", "content": f"Based on the following information, draw inspiration from these short clippings of text from the author H. P. Lovecraft: {context} and give a 2000 worded short horror story from the idea: {idea}"}
        ]
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    idea = "4 college students go on a camping trip to a remote forest and encounter a supernatural entity."
    script = get_script(idea)
    try:
        with open("script.txt", "w") as f:
            f.write(script)
        print("Script written to script.txt")
    except Exception as e:
        print(f"Error writing to file: {e}")


