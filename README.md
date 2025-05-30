# Valley Air RAG Pipeline

## Getting Started: Python Virtual Environment Setup

It is highly recommended to use a Python 3.11 virtual environment for this project. Follow these steps before running any scripts:

1. **Create a virtual environment (Python 3.11):**
   ```bash
   python3.11 -m venv venv
   ```
2. **Activate the virtual environment:**
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

This project implements a full pipeline for building a Retrieval-Augmented Generation (RAG) chatbot for the San Joaquin Valley Air Pollution Control District (Valley Air). The pipeline consists of three main stages:

1. **Web Crawling & Markdown Extraction** (`crawl_data.py`)
2. **Embedding & Indexing in Elasticsearch** (`index_data.py`)
3. **Conversational Chatbot (RAG) Web App** (`chat_app.py`)

---

## 1. Web Crawling & Markdown Extraction (`crawl_data.py`)

### **Purpose**
Fetches all URLs from the Valley Air sitemap, crawls each page, extracts the main content, and saves it as Markdown files (one per page) in the `output/` directory.

### **How it works**
- **Fetches the sitemap** (`https://www.valleyair.org/sitemap.xml`) and parses all URLs.
- **Crawls each URL** using `crawl4ai.AsyncWebCrawler` with content pruning and tag exclusion (removes nav, footer, header).
- **Extracts and cleans content**, generating Markdown for each page.
- **Saves each page** as a Markdown file in the `output/` directory, with a sanitized filename based on the page title or URL.

### **Key Functions**
- `sanitize_filename(name)`: Cleans a string for safe filenames.
- `main()`: Orchestrates the crawling, extraction, and saving process.

### **How to run**
```bash
python crawl_data.py
```
- Output: Markdown files in the `output/` directory.

---

## 2. Embedding & Indexing in Elasticsearch (`index_data.py`)

### **Purpose**
Reads the Markdown files from `output/`, generates embeddings for each chunk using IBM watsonx.ai, and indexes them into an Elasticsearch vector database.

### **How it works**
- **Loads environment variables** for Elasticsearch and IBM watsonx.ai credentials.
- **Initializes Elasticsearch** (creates index if needed, with dense vector mapping).
- **Initializes IBM watsonx.ai embedding model**.
- **Processes each Markdown file**:
  - Extracts the URL and content.
  - Cleans unwanted lines.
  - Splits content into chunks (~1000 characters).
  - Generates embeddings for each chunk.
  - Indexes each chunk (with embedding, content, URL, title, chunk index) into Elasticsearch.

### **Key Functions**
- `initialize_elasticsearch()`: Connects to ES, creates index if missing.
- `initialize_watsonx()`: Sets up the IBM embedding model.
- `chunk_text(text, chunk_size)`: Splits text into manageable chunks.
- `process_file(file_path, embedding_model, es_client)`: Embeds and indexes a single file.
- `main()`: Handles argument parsing, connection checks, and batch processing.

### **How to run**
```bash
python index_data.py
```
- Output: Chunks indexed in Elasticsearch.

**To delete the Elasticsearch index:**
```bash
python index_data.py --delete-index
```

---

## 3. Conversational Chatbot Web App (`chat_app.py`)

### **Purpose**
Provides a web-based (Flask) and CLI chatbot that answers user questions using RAG (Retrieval-Augmented Generation) over the indexed Valley Air content.

### **How it works**
- **Loads environment variables** for all credentials.
- **Initializes the LLM** (Meta Llama 3 via IBM watsonx).
- **Initializes the embedding model** (IBM watsonx).
- **Connects to Elasticsearch** and sets up a custom vector store.
- **Defines a prompt template** for the assistant, with guidelines for tone, structure, and edge cases.
- **Retrieves relevant chunks** from Elasticsearch using vector search.
- **Runs the LLM with context** to generate a concise, helpful answer.
- **Web UI**: Flask app with a modern chat interface, showing sources for each answer.
- **CLI**: Simple command-line chat loop.

### **Key Functions**
- `get_ai_response(user_input, min_score)`: Runs retrieval and LLM, returns answer and sources.
- `run_tests()`: Diagnostic tests for ES and embedding.
- Flask routes: `/` (chat UI), `/chat` (API endpoint).
- `main()`: CLI chat loop.

### **How to run**
**Web app:**
```bash
python chat_app.py --web
```
- Visit [http://localhost:5001](http://localhost:5001) in your browser.

**CLI:**
```bash
python chat_app.py
```

**Test connections:**
```bash
python chat_app.py --test
```

---

## Environment Variables

All scripts require a `.env` file in the project root with the following variables:

```env
# Elasticsearch
ES_URL=https://your-elasticsearch-url:port
ES_USER=your_es_username
ES_PASSWORD=your_es_password
ES_CERT_FINGERPRINT=your_es_ssl_fingerprint
ES_INDEX_NAME=valley_air_documents

# IBM watsonx.ai
IBM_CLOUD_API_KEY=your_ibm_cloud_api_key
IBM_CLOUD_ENDPOINT=https://your-ibm-cloud-endpoint
IBM_CLOUD_PROJECT_ID=your_ibm_project_id

# For LLM (chat_app.py only)
WATSONX_URL=https://your-ibm-watsonx-url
WATSONX_PROJECT_ID=your_watsonx_project_id
```

**Note:**  
- The `WATSONX_URL` and `WATSONX_PROJECT_ID` are only required for the chatbot (LLM) in `chat_app.py`.
- All credentials must be valid for IBM watsonx.ai and Elasticsearch with vector search enabled.

---

## Typical Pipeline Flow

1. **Crawl and extract content:**
   - `python crawl_data.py`
2. **Embed and index content:**
   - `python index_data.py`
3. **Run the chatbot:**
   - `python chat_app.py --web` (for web UI) or `python chat_app.py` (for CLI)

---

## Dependencies

Install all required packages (Python 3.8+ recommended):

```bash
pip install -r requirements.txt
```

**Key packages:**
- `requests`, `beautifulsoup4`, `crawl4ai`
- `elasticsearch`, `ibm-watsonx-ai`
- `langchain`, `flask`, `tqdm`, `python-dotenv`

---

## Directory Structure

```
.
├── crawl_data.py
├── index_data.py
├── chat_app.py
├── output/                # Markdown files (created by crawl_data.py)
├── .env                   # Your environment variables
└── requirements.txt
```

---

## Troubleshooting

- Ensure your `.env` is correct and all services (Elasticsearch, IBM watsonx.ai) are accessible.
- If you change the ES index mapping, delete and recreate the index (`python index_data.py --delete-index`).
- For SSL issues with Elasticsearch, check your `ES_CERT_FINGERPRINT`.

---

## Contact

For questions about Valley Air, visit [valleyair.org](https://www.valleyair.org) or call 559-230-5800.

--- 