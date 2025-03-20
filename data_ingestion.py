import os
import glob
import tiktoken

from dotenv import load_dotenv
from langchain.schema import Document



load_dotenv()

def read_docments(data_dir = "resources"):
    document = []
    file = glob.glob(os.path.join(data_dir, "all_lovecraft_stories.txt"))
    with open(file[0], "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        content = content.replace("\n", " ")
        document = [Document(page_content=content, metadata={"source": "all_lovecraft_stories.txt"})]
    print("Document read successfully.")
    return document

def chunkify(document):
    encoding = tiktoken.get_encoding(os.getenv("TOKEN_ENCODING"))
    tokens = encoding.encode(document[0].page_content)

    chunk_size = 1000
    chunk_overlap = 200
    chunks = []
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(encoding.decode(chunk))
    print("Document chunked successfully.")
    return chunks
