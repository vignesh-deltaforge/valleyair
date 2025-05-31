from langgraph.graph import StateGraph, END, START
from agents.query_context import QueryContextAgent, AgentState
from agents.retrieval import SpecializedRetrievalAgent
from agents.synthesis import ResponseSynthesisAgent, StreamingResponseSynthesisAgent
from agents.air_quality_agent import AirQualityAgent
from llm import llm, ibm_embedding
from vectorstore import vectorstore, es_connection, ES_INDEX

# --- Query Classifier Tool ---
class QueryClassifierTool:
    def __init__(self, llm):
        self.llm = llm
    def __call__(self, state):
        user_query = state.get("user_query", "")
        print(f"===CLASSIFIER INPUT=== {user_query!r}")
        prompt = f"""
<|start_of_role|>system<|end_of_role|>You are a classifier for the Valley Air chatbot. Your job is to classify the user's query as either "air_quality" or "general".

Instructions:
- Output ONLY one of these two labels: air_quality or general.
- Output the label as the first line, with no explanation, no extra text, and no formatting.
- If the query asks about AQI, air quality, air pollution, PM2.5, PM10, ozone, NO2, SO2, CO, smoke, wildfire smoke, burn days, air quality advisories, or pollutant concentrations, output air_quality.
- If the query is about Valley Air rules, grants, permits, enforcement, regulations, board meetings, sponsorships, rulemaking, appeals, inspections, or any other topic not directly about current air quality or pollutant levels, output general.
- If the query is ambiguous, output general.
<|end_of_text|>
<|start_of_role|>user<|end_of_role|>Query: {user_query}<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
"""
        response = self.llm.invoke(prompt)
        label = response.strip().splitlines()[0].strip().lower()
        print(f"===CLASSIFIER RAW OUTPUT=== {response!r} for query: {user_query!r}")
        if label not in ["air_quality", "general"]:
            print(f"===CLASSIFIER FINAL LABEL=== Unexpected label {label!r}, defaulting to 'general'. Full output: {response!r}")
            label = "general"
        else:
            print(f"===CLASSIFIER FINAL LABEL=== Classified as {label!r}")
        state = dict(state)
        state["query_type"] = label
        return state

def route_query(state):
    if state.get("query_type") == "air_quality":
        return "air_quality_agent"
    return "query_context"

def merge_parallel_results(state1, state2):
    merged = dict(state1)
    merged.update(state2)
    docs1 = state1.get("retrieved_docs", [])
    docs2 = state2.get("retrieved_docs", [])
    merged["retrieved_docs"] = docs1 + docs2
    if state1.get("air_quality_data"):
        merged["air_quality_data"] = state1["air_quality_data"]
    if state2.get("air_quality_data"):
        merged["air_quality_data"] = state2["air_quality_data"]
    if state1.get("location"):
        merged["location"] = state1["location"]
    if state2.get("location"):
        merged["location"] = state2["location"]
    return merged

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
air_quality_agent = AirQualityAgent(llm)
query_classifier = QueryClassifierTool(llm)

graph_builder = StateGraph(AgentState)
graph_builder.add_node("classifier", query_classifier)
graph_builder.add_node("air_quality_agent", air_quality_agent)
graph_builder.add_node("query_context", query_agent)
graph_builder.add_node("specialized_retrieval", retrieval_agent)
graph_builder.add_node("synthesis", synthesis_agent)
graph_builder.add_edge(START, "classifier")
graph_builder.add_conditional_edges(
    "classifier",
    route_query,
    {
        "air_quality_agent": "air_quality_agent",
        "query_context": "query_context"
    }
)
graph_builder.add_edge("air_quality_agent", "synthesis")
graph_builder.add_edge("query_context", "specialized_retrieval")
graph_builder.add_edge("specialized_retrieval", "synthesis")
graph_builder.add_edge("synthesis", END)
graph = graph_builder.compile()

def run_multiagent_workflow(user_query):
    state = {"user_query": user_query, "messages": []}
    state = query_classifier(state)
    if state.get("query_type") == "air_quality":
        air_state = air_quality_agent(state)
        result = synthesis_agent(air_state)
    elif state.get("query_type") == "general":
        rewrites = None
        keywords = None
        for event in query_agent.stream(state):
            if event.get("rewrites") is not None:
                rewrites = event["rewrites"]
            if event.get("keywords") is not None:
                keywords = event["keywords"]
        if rewrites is not None:
            state["rewrites"] = rewrites
        if keywords is not None:
            state["keywords"] = keywords
        state = retrieval_agent(state)
        result = synthesis_agent(state)
    else:
        result = {"answer": "Sorry, I could not classify your query.", "sources": []}
    return result.get("answer", ""), result.get("sources", [])

def run_multiagent_workflow_streaming(user_query, callback_handler=None):
    state = {"user_query": user_query, "messages": []}
    state = query_classifier(state)
    print(f"DEBUG: Entering workflow with query_type: {state.get('query_type')}")
    if state.get("query_type") == "air_quality":
        print("DEBUG: Calling air_quality_agent.stream")
        for event in air_quality_agent.stream(state, callback_handler=callback_handler):
            print(f"DEBUG: AirQualityAgent event: {event}")
            yield event
    elif state.get("query_type") == "general":
        rewrites = None
        keywords = None
        for event in query_agent.stream(state, callback_handler=callback_handler):
            yield event
            if event.get("rewrites") is not None:
                rewrites = event["rewrites"]
            if event.get("keywords") is not None:
                keywords = event["keywords"]
        if rewrites is not None:
            state["rewrites"] = rewrites
        if keywords is not None:
            state["keywords"] = keywords
        if callback_handler is not None:
            callback_handler.on_tool_start("SpecializedRetrievalAgent", "Retrieving relevant documents from Elasticsearch and BM25.")
        state = retrieval_agent(state)
        for event in streaming_synth_agent.stream(state, callback_handler=callback_handler):
            yield event
    else:
        yield {"type": "done", "sources": [], "answer": "Sorry, I could not classify your query."} 