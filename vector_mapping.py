import os
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from data_ingestion import read_docments, chunkify

load_dotenv()

model = SentenceTransformer(os.getenv("EMBEDDING_MODEL"))

doc = read_docments()
chunks = chunkify(doc)


texts = [chunk.page_content for chunk in chunks]
embeddings = model.encode(texts, show_progress_bar=True)

index =  faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

faiss.write_index(index, os.getenv("VECTOR_STORE"))

with open(os.getenv("VECTOR_MAPPING"), "w") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"{i}||{chunk.page_content}\n")
   
print("Vector store and mapping created successfully.")




