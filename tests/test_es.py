from vectorstore import es_connection, ES_INDEX, vectorstore
from llm import ibm_embedding

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