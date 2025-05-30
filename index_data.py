import os
from typing import List, Dict
import json
from elasticsearch import Elasticsearch
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import sys
import argparse

# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables
ES_URL = os.getenv("ES_URL")
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_CERT_FINGERPRINT = os.getenv("ES_CERT_FINGERPRINT")
ES_INDEX = os.getenv("ES_INDEX_NAME", "valley_air_documents")
OUTPUT_DIR = "output"

IBM_CLOUD_API_KEY = os.getenv("IBM_CLOUD_API_KEY")
IBM_CLOUD_ENDPOINT = os.getenv("IBM_CLOUD_ENDPOINT")
IBM_CLOUD_PROJECT_ID = os.getenv("IBM_CLOUD_PROJECT_ID")

print("DEBUG: ES_URL =", repr(ES_URL))
print("DEBUG: ES_USER =", repr(ES_USER))
print("DEBUG: ES_PASSWORD =", repr(ES_PASSWORD))
print("DEBUG: ES_CERT_FINGERPRINT =", repr(ES_CERT_FINGERPRINT))
print("DEBUG: ES_INDEX =", repr(ES_INDEX))

def initialize_elasticsearch():
    """Initialize Elasticsearch client and create index if it doesn't exist."""
    es = Elasticsearch(
        ES_URL,
        basic_auth=(ES_USER, ES_PASSWORD),
        verify_certs=True,
        ssl_assert_fingerprint=ES_CERT_FINGERPRINT
    )
    print("Elasticsearch client info:", es.info())
    try:
        print("Pinging Elasticsearch...")
        if not es.ping():
            print("Ping failed! Check your credentials, URL, and network.")
        else:
            print("Ping successful.")
        print("Checking if index exists...")
        if not es.indices.exists(index=ES_INDEX):
            print(f"Index '{ES_INDEX}' does not exist. Creating...")
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "url": {"type": "keyword"},
                        "title": {"type": "text"},
                        "chunk_index": {"type": "integer"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 768
                        }
                    }
                }
            }
            # NOTE: If you change the mapping, you must delete and recreate the index for changes to take effect.
            try:
                es.indices.create(index=ES_INDEX, body=mapping)
                print(f"Index '{ES_INDEX}' created successfully.")
            except Exception as e:
                print(f"Error creating index '{ES_INDEX}': {e}")
                if hasattr(e, 'info'):
                    print(f"Elasticsearch error info: {e.info}")
                raise
        else:
            print(f"Index '{ES_INDEX}' already exists.")
    except Exception as e:
        print(f"Error during index existence check or creation: {e}")
        if hasattr(e, 'info'):
            print(f"Elasticsearch error info: {e.info}")
        import traceback
        traceback.print_exc()
        raise
    return es

def initialize_watsonx():
    """Initialize IBM watsonx.ai embedding model."""
    model_id = "ibm/slate-125m-english-rtrvr-v2"
    embed_params = {
        EmbedParams.TRUNCATE_INPUT_TOKENS: 500,  # Restrict to 500 tokens
        EmbedParams.RETURN_OPTIONS: {
            'input_text': True
        }
    }
    credentials = {
        "url": IBM_CLOUD_ENDPOINT,
        "apikey": IBM_CLOUD_API_KEY,
    }
    embedding = Embeddings(
        model_id=model_id,
        credentials=credentials,
        params=embed_params,
        project_id=IBM_CLOUD_PROJECT_ID
    )
    return embedding

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks of approximately chunk_size characters."""
    chunks = []
    current_chunk = []
    current_size = 0
    # Split by sentences (simple approach)
    sentences = text.split('. ')
    for sentence in sentences:
        if current_size + len(sentence) > chunk_size and current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_size = len(sentence)
        else:
            current_chunk.append(sentence)
            current_size += len(sentence)
    if current_chunk:
        chunks.append('. '.join(current_chunk))
    return chunks

def is_crap_line(line: str) -> bool:
    """Return True if the line is considered unwanted/crap content."""
    CRAP_LINES = [
        '*   *   *   *   *   *   *',
        'You can search for the page or document that you are looking for here:',
        'For assistance or if you have any questions, please feel free to .',
        'Your feedback will be used to help improve Google Translate',
        '',  # Also treat empty lines as crap for filtering
    ]
    # Remove leading/trailing whitespace for comparison
    line_stripped = line.strip()
    if not line_stripped:
        return True
    if line_stripped in CRAP_LINES:
        return True
    return False

def process_file(file_path: str, embedding_model, es_client) -> None:
    """Process a single markdown file and store its vectors in Elasticsearch."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Extract URL from first line
    lines = content.split('\n')
    url = lines[0].strip()
    # Get the rest of the content, filter out crap lines
    filtered_lines = [line for line in lines[1:] if not is_crap_line(line)]
    text_content = '\n'.join(filtered_lines).strip()
    # Skip if content is empty after filtering
    if not text_content:
        print(f"Skipping file {file_path}: only URL and empty or unwanted content.")
        return
    # Chunk the content
    chunks = chunk_text(text_content)
    # Generate embeddings for each chunk
    for i, chunk in enumerate(chunks):
        try:
            # Generate embedding
            embedding_result = embedding_model.embed_documents(texts=[chunk])
            vector = embedding_result[0]
            # Prepare document for Elasticsearch
            doc = {
                "content": chunk,
                "url": url,
                "title": os.path.basename(file_path),
                "chunk_index": i,
                "embedding": vector
            }
            # Index the document
            es_client.index(
                index=ES_INDEX,
                document=doc
            )
        except Exception as e:
            print(f"Error processing chunk {i} in file {file_path}: {str(e)}")

def check_elasticsearch_connection(es_client):
    try:
        if es_client.ping():
            print("Successfully connected to Elasticsearch.")
            return True
        else:
            print("Failed to connect to Elasticsearch.")
            return False
    except Exception as e:
        print(f"Error connecting to Elasticsearch: {e}")
        return False

def check_watsonx_connection(embedding_model):
    try:
        # Try a dummy embedding call
        _ = embedding_model.embed_documents(texts=["connection test"])
        print("Successfully connected to IBM watsonx.ai.")
        return True
    except Exception as e:
        print(f"Error connecting to IBM watsonx.ai: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Index markdown files or delete Elasticsearch index.")
    parser.add_argument('--delete-index', action='store_true', help='Delete the Elasticsearch index and exit')
    args = parser.parse_args()

    # Initialize Elasticsearch client
    es_client = initialize_elasticsearch()

    if args.delete_index:
        try:
            if es_client.indices.exists(index=ES_INDEX):
                es_client.indices.delete(index=ES_INDEX)
                print(f"Index '{ES_INDEX}' deleted successfully.")
            else:
                print(f"Index '{ES_INDEX}' does not exist.")
        except Exception as e:
            print(f"Error deleting index '{ES_INDEX}': {e}")
        return

    # Initialize embedding model
    embedding_model = initialize_watsonx()

    # Check connections before proceeding
    if not check_elasticsearch_connection(es_client):
        print("Exiting due to Elasticsearch connection failure.")
        sys.exit(1)
    if not check_watsonx_connection(embedding_model):
        print("Exiting due to IBM watsonx.ai connection failure.")
        sys.exit(1)

    # Process all markdown files
    markdown_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.md')]
    for file_name in tqdm(markdown_files, desc="Processing files"):
        file_path = os.path.join(OUTPUT_DIR, file_name)
        process_file(file_path, embedding_model, es_client)

if __name__ == "__main__":
    main()