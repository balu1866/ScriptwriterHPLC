# ScriptwriterHPLC

ScriptwriterHPLC is a Python-based tool that generates horror scripts inspired by H. P. Lovecraft. It uses OpenAI's GPT-3.5-turbo model, FAISS for vector search, and Sentence Transformers for text embeddings.

## Features

- Generates horror scripts based on user-provided ideas.
- Leverages H. P. Lovecraft's text clippings for inspiration.
- Uses FAISS for efficient vector-based similarity search.
- Supports custom embedding models and vector stores.

## Requirements

- Python 3.9 or higher
- OpenAI API key
- Pre-trained Sentence Transformer model
- FAISS index and vector mapping file

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ScriptwriterHPLC.git
   cd ScriptwriterHPLC
   ```

2. Set up a virtual environment:
   ```bash
   python3 -m venv .hplcRAG
   source .hplcRAG/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with the following variables:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   EMBEDDING_MODEL=your_embedding_model_name
   VECTOR_STORE=path_to_your_faiss_index
   VECTOR_MAPPING=path_to_your_vector_mapping_file
   ```

## Usage

1. Prepare your FAISS index and vector mapping file:
   - The FAISS index should be stored at the path specified in the `VECTOR_STORE` environment variable.
   - The vector mapping file should contain lines in the format `index||text`.

2. Run the script:
   ```bash
   python script_getter.py
   ```

3. Provide an idea for the script in the `get_script` function or modify the `idea` variable in the `__main__` block.

4. The generated script will be saved to `script.txt`.

## Example

```python
idea = "4 college students go on a camping trip to a remote forest and encounter a supernatural entity."
script = get_script(idea)
```

## File Structure

```
.env
.gitignore
data_ingestion.py
README.md
script_getter.py
script.txt
vector_mapping.py
resources/
    all_lovecraft_stories.txt
faiss/
    index_mapping.json
    index.faiss
```

## Acknowledgments

- [OpenAI](https://openai.com) for GPT-3.5-turbo.
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search.
- [Sentence Transformers](https://www.sbert.net/) for embedding generation.
- H. P. Lovecraft for the inspiration.