from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from vectorstore import vectorstore, vector_retriever, es_connection, ES_INDEX
from llm import ibm_embedding

class SpecializedRetrievalAgent:
    def __init__(self, vectorstore, es_connection, es_index, embedding_model, docs_corpus):
        self.vectorstore = vectorstore
        self.es_connection = es_connection
        self.es_index = es_index
        self.embedding_model = embedding_model
        self.docs_corpus = docs_corpus
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    def __call__(self, state):
        rewrites = state.get("rewrites", [])
        keywords = state.get("keywords", [])
        # BM25 retrieval
        bm25 = BM25Okapi([doc["content"].split() for doc in self.docs_corpus])
        bm25_scores = bm25.get_scores(keywords)
        bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:10]
        bm25_docs = [self.docs_corpus[i] for i in bm25_top_indices]
        # Vector retrieval (using retriever to ensure metadata)
        vector_results = vector_retriever.invoke(rewrites[0])
        # Combine and deduplicate, always prefer Document with metadata over dict
        all_docs = {}
        for i, doc in enumerate(vector_results):
            url = getattr(doc, 'metadata', {}).get("url", None)
            if url:
                all_docs[url] = doc
            else:
                all_docs[f"vector_{i}"] = doc
        for doc in bm25_docs:
            url = doc.get("url", None)
            if url and url not in all_docs:
                all_docs[url] = doc
            elif not url:
                all_docs[f"bm25_{id(doc)}"] = doc
        combined_docs = list(all_docs.values())
        # Rerank using cross-encoder
        query = state.get("user_query", rewrites[0] if rewrites else "")
        pairs = [(query, doc.page_content if hasattr(doc, 'page_content') else doc["content"]) for doc in combined_docs]
        scores = self.cross_encoder.predict(pairs)
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:4]
        top_docs = [combined_docs[i] for i in top_indices]
        # --- Ensure all top_docs have full metadata by fetching from ES ---
        enriched_docs = []
        for doc in top_docs:
            url = None
            chunk_index = None
            if hasattr(doc, 'metadata') and doc.metadata:
                url = doc.metadata.get("url")
                chunk_index = doc.metadata.get("chunk_index")
            elif isinstance(doc, dict):
                url = doc.get("url")
                chunk_index = doc.get("chunk_index")
            es_query = None
            if url is not None and chunk_index is not None:
                es_query = {
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"url": url}},
                                {"term": {"chunk_index": chunk_index}}
                            ]
                        }
                    }
                }
            elif url is not None:
                es_query = {
                    "query": {"term": {"url": url}}
                }
            else:
                content = doc.page_content if hasattr(doc, 'page_content') else doc.get("content", "")
                es_query = {
                    "query": {"match": {"content": content}}
                }
            es_res = self.es_connection.search(index=self.es_index, body=es_query, size=1)
            hits = es_res["hits"]["hits"]
            if hits:
                enriched_doc = self.vectorstore._create_documents([hits[0]])[0]
                enriched_docs.append(enriched_doc)
            else:
                enriched_docs.append(doc)
        # print("DEBUG: RetrievalAgent top_docs:", top_docs)
        # print("DEBUG: RetrievalAgent enriched_docs:", enriched_docs)
        return {**state, "retrieved_docs": enriched_docs} 