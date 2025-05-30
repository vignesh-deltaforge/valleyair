from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document
from config import ES_URL, ES_USER, ES_PASSWORD, ES_INDEX, ES_CERT_FINGERPRINT
from llm import IBMEmbeddingWrapper

# Create the Elasticsearch connection with ssl_assert_fingerprint
es_connection = Elasticsearch(
    ES_URL,
    basic_auth=(ES_USER, ES_PASSWORD),
    verify_certs=True,
    ssl_assert_fingerprint=ES_CERT_FINGERPRINT
)

class CustomElasticsearchStore(ElasticsearchStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_documents(self, hits):
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

vector_retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 10,
        "source_fields": ["content", "url", "title", "chunk_index"]
    }
) 