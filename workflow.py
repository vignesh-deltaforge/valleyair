from langgraph.graph import StateGraph, END, START
from agents.query_context import QueryContextAgent, AgentState
from agents.retrieval import SpecializedRetrievalAgent
from agents.synthesis import ResponseSynthesisAgent, StreamingResponseSynthesisAgent
from llm import llm, ibm_embedding
from vectorstore import vectorstore, es_connection, ES_INDEX

# Prepare docs_corpus for BM25 (load from ES index)
def load_docs_corpus(es_connection, es_index):
    res = es_connection.search(index=es_index, size=1000)
    docs = []
    for hit in res["hits"]["hits"]:
        src = hit["_source"]
        docs.append({"content": src.get("content", ""), "url": src.get("url", ""), "title": src.get("title", "")})
    return docs

docs_corpus = load_docs_corpus(es_connection, ES_INDEX)

query_agent = QueryContextAgent(llm)
retrieval_agent = SpecializedRetrievalAgent(vectorstore, es_connection, ES_INDEX, ibm_embedding, docs_corpus)
synthesis_agent = ResponseSynthesisAgent(llm)
streaming_synth_agent = StreamingResponseSynthesisAgent(llm)

graph_builder = StateGraph(AgentState)
graph_builder.add_node("query_context", query_agent)
graph_builder.add_node("specialized_retrieval", retrieval_agent)
graph_builder.add_node("synthesis", synthesis_agent)
graph_builder.add_edge(START, "query_context")
graph_builder.add_edge("query_context", "specialized_retrieval")
graph_builder.add_edge("specialized_retrieval", "synthesis")
graph_builder.add_edge("synthesis", END)
graph = graph_builder.compile()

def run_multiagent_workflow(user_query):
    state = {"user_query": user_query, "messages": []}
    result = graph.invoke(state)
    return result.get("answer", ""), result.get("sources", [])

def run_multiagent_workflow_streaming(user_query):
    state = {"user_query": user_query, "messages": []}
    query_stream = query_agent.stream(state)
    for event in query_stream:
        if event["type"] == "tool":
            yield event
            state["rewrites"] = event["rewrites"]
            state["keywords"] = event["keywords"]
        elif event["type"] == "query_context_done":
            state["rewrites"] = event["rewrites"]
            state["keywords"] = event["keywords"]
    state = retrieval_agent(state)
    yield {"type": "tool", "tool": "SpecializedRetrievalAgent", "description": "Retrieved relevant documents from Elasticsearch and BM25."}
    for event in streaming_synth_agent.stream(state):
        yield event 