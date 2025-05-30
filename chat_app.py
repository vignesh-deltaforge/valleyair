import os
import sys
import argparse
from dotenv import load_dotenv
from langchain_ibm import WatsonxLLM
from langchain_elasticsearch import ElasticsearchStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from elasticsearch import Elasticsearch
from langchain_core.documents import Document
from flask import Flask, request, jsonify, render_template_string

# Load environment variables
load_dotenv()

ES_URL = os.getenv("ES_URL")
ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_INDEX = os.getenv("ES_INDEX_NAME", "valley_air_documents")
ES_CERT_FINGERPRINT = os.getenv("ES_CERT_FINGERPRINT")

IBM_CLOUD_API_KEY = os.getenv("IBM_CLOUD_API_KEY")
IBM_CLOUD_ENDPOINT = os.getenv("IBM_CLOUD_ENDPOINT")
IBM_CLOUD_PROJECT_ID = os.getenv("IBM_CLOUD_PROJECT_ID")

# Set up the LLM (meta-llama/llama-3-3-70b-instruct)
llm = WatsonxLLM(
    model_id="meta-llama/llama-3-3-70b-instruct",
    url=os.getenv("WATSONX_URL"),
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    params={
        "decoding_method": "sample",
        "max_new_tokens": 256,
        "min_new_tokens": 1,
        "temperature": 0.5,
        "top_k": 50,
        "top_p": 0.9,
    }
)

# Set up the embedding model (ibm/slate-125m-english-rtrvr-v2)
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams

embed_params = {
    EmbedParams.TRUNCATE_INPUT_TOKENS: 500,
    EmbedParams.RETURN_OPTIONS: {'input_text': True}
}
ibm_embedding = Embeddings(
    model_id="ibm/slate-125m-english-rtrvr-v2",
    credentials={"url": IBM_CLOUD_ENDPOINT, "apikey": IBM_CLOUD_API_KEY},
    params=embed_params,
    project_id=IBM_CLOUD_PROJECT_ID
)

class IBMEmbeddingWrapper:
    def embed_documents(self, texts):
        return ibm_embedding.embed_documents(texts=texts)
    def embed_query(self, text):
        return ibm_embedding.embed_documents(texts=[text])[0]

# Create the Elasticsearch connection with ssl_assert_fingerprint
es_connection = Elasticsearch(
    ES_URL,
    basic_auth=(ES_USER, ES_PASSWORD),
    verify_certs=True,
    ssl_assert_fingerprint=ES_CERT_FINGERPRINT
)

class CustomElasticsearchStore(ElasticsearchStore):
    def __init__(self, *args, **kwargs):
        # print("[DEBUG] CustomElasticsearchStore __init__ called")
        super().__init__(*args, **kwargs)

    def _create_documents(self, hits):
        # print("[DEBUG] CustomElasticsearchStore _create_documents called")
        documents = []
        for hit in hits:
            source = hit.get("_source", {})
            metadata = {
                "url": source.get("url", ""),
                "title": source.get("title", ""),
                "chunk_index": source.get("chunk_index", 0),
                "score": hit.get("_score", 0.0),
                **{k: v for k, v in source.items() if k not in [self.query_field, self.vector_query_field]}
            }
            content = source.get(self.query_field, "")
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

# Initialize the custom vector store
vectorstore = CustomElasticsearchStore(
    es_connection=es_connection,
    index_name=ES_INDEX,
    embedding=IBMEmbeddingWrapper(),
    vector_query_field="embedding",
    query_field="content",
)

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4,
        "source_fields": ["content", "url", "title", "chunk_index"]
    }
)


# Prompt template for RAG
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
<|begin_of_text|><|start_header_id|>system<|end_header_id>
You are an AI assistant for the San Joaquin Valley Air Pollution Control District (Valley Air), dedicated to improving air quality in California's Central Valley. Your goal is to provide accurate, concise, and helpful answers based on valleyair.org content and the provided context. Follow these guidelines:

1. **Tone**: Use a friendly, professional tone. Explain technical terms (e.g., AQI, PM2.5) in simple language for residents, businesses, and community members.
2. **Answer Structure**:
   - Answer directly in 1-2 sentences.
   - List specific details or benefits in bullet points if the question asks for them (e.g., "benefits" include financial, environmental, or operational advantages).
   - Suggest a follow-up action (e.g., visit valleyair.org/grants, call 559-230-5800).
3. **Context**:
   - Use the context to answer. If insufficient, say: "I don't have enough details to answer fully. Visit valleyair.org or call 559-230-5800."
4. **Edge Cases**:
   - Vague questions: Ask for clarification (e.g., "Can you specify what you mean?").
   - Off-topic: Redirect politely (e.g., "I focus on air quality and Valley Air services. How can I help?").
   - Sensitive topics: Respond empathetically and suggest contact (e.g., "I'm sorry for your concern. Contact Valley Air at 559-230-5800.").
5. **Real-Time Data**: For current AQI or events, direct to valleyair.org/air-quality.
6. **Output**: Generate only the answer text, excluding any structural markers, headers, or prior questions. Do not include tokens like <|eot_id|> or <|start_header_id|> in the response.

Context from valleyair.org:
{context}
<|eot_id|><|start_header_id|>user<|end_header_id>
{question}
<|eot_id|>
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

def run_tests():
    print("\n--- Elasticsearch Index Test ---")
    try:
        res = es_connection.search(index=ES_INDEX, size=1)
        doc = res['hits']['hits'][0]['_source']
        print("Sample document from index:")
        for k, v in doc.items():
            print(f"  {k}: {str(v)[:100]}{'...' if len(str(v)) > 100 else ''}")
        if 'embedding' in doc:
            print(f"Embedding length: {len(doc['embedding'])}")
        else:
            print("No 'embedding' field found in document!")
    except Exception as e:
        print(f"Error fetching document from index: {e}")

    print("\n--- Manual Vector Search Test ---")
    try:
        test_query = "What grants does valley air provide?"
        query_vector = ibm_embedding.embed_documents(texts=[test_query])[0]
        res = es_connection.search(
            index=ES_INDEX,
            knn={
                "field": "embedding",
                "query_vector": query_vector,
                "k": 4,
                "num_candidates": 100
            }
        )
        print(f"Found {len(res['hits']['hits'])} hits for test query.")
        for hit in res['hits']['hits']:
            src = hit['_source']
            print(f"Score: {hit.get('_score', 'N/A')}")
            print(f"Content: {src.get('content', '')[:100]}...")
            print(f"URL: {src.get('url', 'No URL')}")
            print(f"Full metadata: {src}")
            print("---")
    except Exception as e:
        print(f"Error running manual vector search: {e}")

    print("\n--- Retriever Test ---")
    try:
        # Directly use the vectorstore's search and document creation
        query = "What grants does valley air provide?"
        query_vector = ibm_embedding.embed_documents(texts=[query])[0]
        res = es_connection.search(
            index=ES_INDEX,
            knn={
                "field": "embedding",
                "query_vector": query_vector,
                "k": 4,
                "num_candidates": 100
            }
        )
        hits = res["hits"]["hits"]
        docs = vectorstore._create_documents(hits)
        print(f"Retriever returned {len(docs)} documents.")
        for doc in docs:
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
            print("---")
    except Exception as e:
        print(f"Error running retriever: {e}")

def get_ai_response(user_input, min_score=0.2):
    result = qa_chain.invoke({"query": user_input})
    ai_answer = result["result"]
    # Custom vector search for sources with full metadata
    query_vector = ibm_embedding.embed_documents(texts=[user_input])[0]
    res = es_connection.search(
        index=ES_INDEX,
        knn={
            "field": "embedding",
            "query_vector": query_vector,
            "k": 4,
            "num_candidates": 100
        }
    )
    hits = res["hits"]["hits"]
    # Only keep unique sources by URL and above min_score
    seen_urls = set()
    sources = []
    for hit in hits:
        doc = vectorstore._create_documents([hit])[0]
        url = doc.metadata.get("url", "No URL")
        score = doc.metadata.get("score", hit.get("_score", 0.0))
        if url not in seen_urls and score >= min_score:
            sources.append({
                "url": url,
                "title": doc.metadata.get("title", "Untitled"),
                "score": score,
                "metadata": doc.metadata
            })
            seen_urls.add(url)
    return ai_answer, sources

# --- Flask Web App ---
app = Flask(__name__)

CHAT_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Valley Air RAG Chatbot</title>
    <style>
        html, body {
            height: 100%;
            margin: 0;
            padding: 0;
            width: 100vw;
            background: #f4f7fa;
            overflow: hidden;
        }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            height: 100vh;
            width: 100vw;
        }
        .chat-container {
            width: 100vw;
            height: 100vh;
            min-height: 0;
            min-width: 0;
            background: #fff;
            display: flex;
            flex-direction: column;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            box-shadow: none;
            border-radius: 0;
        }
        .chat-header {
            text-align: center;
            color: #1976d2;
            padding: 18px 0 8px 0;
            font-size: 1.3em;
            font-weight: 600;
            border-bottom: 1px solid #e0e0e0;
            background: #fff;
        }
        .chat-history {
            flex: 1 1 auto;
            overflow-y: auto;
            padding: 18px 18px 8px 18px;
            display: flex;
            flex-direction: column;
            min-height: 0;
        }
        .bubble { padding: 14px 18px; border-radius: 18px; margin-bottom: 10px; max-width: 80%; word-break: break-word; }
        .user { background: #e0f7fa; align-self: flex-end; margin-left: 20%; text-align: right; }
        .ai { background: #e8eaf6; align-self: flex-start; margin-right: 20%; }
        .sources { font-size: 0.95em; margin-top: 8px; margin-bottom: 8px; }
        .sources a { color: #1976d2; text-decoration: none; }
        .sources a:hover { text-decoration: underline; }
        .input-row {
            display: flex;
            gap: 8px;
            padding: 16px 18px 18px 18px;
            background: #fff;
            border-top: 1px solid #e0e0e0;
            position: sticky;
            bottom: 0;
            z-index: 2;
        }
        .input-row input {
            flex: 1;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #bdbdbd;
            font-size: 1em;
        }
        .input-row button {
            padding: 12px 20px;
            border-radius: 8px;
            border: none;
            background: #1976d2;
            color: #fff;
            font-size: 1em;
            cursor: pointer;
        }
        .input-row button:active { background: #1565c0; }
        @media (max-width: 700px) {
            .chat-container { width: 100vw; height: 100vh; border-radius: 0; }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">Valley Air RAG Chatbot</div>
        <div class="chat-history" id="chat-history"></div>
        <form class="input-row" id="chat-form" autocomplete="off">
            <input type="text" id="user-input" placeholder="Type your question..." autofocus required />
            <button type="submit">Send</button>
        </form>
    </div>
    <script>
        const chatHistory = document.getElementById('chat-history');
        const chatForm = document.getElementById('chat-form');
        const userInput = document.getElementById('user-input');

        function addBubble(text, sender, sources=[]) {
            const bubble = document.createElement('div');
            bubble.className = 'bubble ' + sender;
            bubble.innerText = text;
            chatHistory.appendChild(bubble);
            if (sender === 'ai' && sources.length > 0) {
                const srcDiv = document.createElement('div');
                srcDiv.className = 'sources';
                srcDiv.innerHTML = '<b>Sources:</b><ul style="padding-left:18px; margin:4px 0;">' +
                    sources.map(s => `<li><a href="${s.url}" target="_blank">${s.url}</a></li>`).join('') + '</ul>';
                chatHistory.appendChild(srcDiv);
            }
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }

        chatForm.onsubmit = async (e) => {
            e.preventDefault();
            const text = userInput.value.trim();
            if (!text) return;
            addBubble(text, 'user');
            userInput.value = '';
            addBubble('Thinking...', 'ai');
            try {
                const resp = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text })
                });
                const data = await resp.json();
                // Remove the 'Thinking...' bubble
                chatHistory.removeChild(chatHistory.lastChild);
                addBubble(data.answer, 'ai', data.sources);
            } catch (err) {
                chatHistory.removeChild(chatHistory.lastChild);
                addBubble('Sorry, there was an error. Please try again.', 'ai');
            }
        };

        // Always scroll to bottom on page load
        window.onload = () => {
            chatHistory.scrollTop = chatHistory.scrollHeight;
        };
    </script>
</body>
</html>
'''

@app.route("/", methods=["GET"])
def index():
    return render_template_string(CHAT_TEMPLATE)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"answer": "Please provide a message.", "sources": []})
    # You can adjust min_score here if needed
    answer, sources = get_ai_response(user_message, min_score=0.4)
    # Only send url and title for each source
    return jsonify({
        "answer": answer,
        "sources": [{"url": s["url"], "title": s["title"]} for s in sources]
    })

def main():
    print("Welcome to the Valley Air RAG Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        result = qa_chain.invoke({"query": user_input})
        print("AI:", result["result"])
        print("Sources:")
        # Custom vector search for sources with full metadata
        query_vector = ibm_embedding.embed_documents(texts=[user_input])[0]
        res = es_connection.search(
            index=ES_INDEX,
            knn={
                "field": "embedding",
                "query_vector": query_vector,
                "k": 4,
                "num_candidates": 100
            }
        )
        hits = res["hits"]["hits"]
        docs = vectorstore._create_documents(hits)
        for doc in docs:
            print("-", doc.metadata.get("url", "No URL"))
            print("  Metadata:", doc.metadata)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run ES and embedding tests')
    parser.add_argument('--web', action='store_true', help='Run the Flask web chat UI')
    args = parser.parse_args()
    if args.test:
        run_tests()
    elif args.web:
        app.run(host="0.0.0.0", port=5001, debug=True)
    else:
        main() 