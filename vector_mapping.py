import os
import faiss
import openai
import numpy as np

from dotenv import load_dotenv
from data_ingestion import read_docments, chunkify

load_dotenv()

model = os.getenv("OPENAI_EMBEDDING_MODEL")


doc = read_docments()
chunks = chunkify(doc)


texts = [chunk for chunk in chunks]

embeddings = []
count = 0
n = len(texts)
print("length of texts: ", n)
for text in texts:
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    embeddings.append(response.data[0].embedding)
    count += 1
    if count % 50 == 0:
        print(f"loops left: {n - count}")

index =  faiss.IndexFlatL2(len(embeddings[0]))
index.add(np.array(embeddings))

faiss.write_index(index, os.getenv("VECTOR_STORE"))

with open(os.getenv("VECTOR_MAPPING"), "w") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"{i}||{chunk}\n")
   
print("Vector store and mapping created successfully.")




