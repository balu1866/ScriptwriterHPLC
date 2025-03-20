import os
import glob

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter




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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(document)
    print("Document chunked successfully.")
    return chunks
