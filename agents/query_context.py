from langchain.prompts import PromptTemplate
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph.message import add_messages

# --- Improved Query & Context Agent Prompts ---
REWRITE_AND_KEYWORDS_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""
<|start_of_role|>system<|end_of_role|>You are an expert search assistant for an air quality and public services knowledge base. Your job is to help users find the most relevant information by rewriting their queries for semantic search and generating effective BM25-style keywords.

**Instructions**:
1. For the user query, generate three rewritten queries, each capturing a unique intent or phrasing relevant to air quality, grants, permits, or Valley Air services.
2. Each rewrite should be clear, concise, and optimized for semantic search.
3. Produce a list of 5-7 BM25-style keywords or short phrases to improve document retrieval.
4. Output a JSON object with two keys: `"rewrites"` (list of 3 rewritten queries) and `"keywords"` (list of 5-7 keywords/phrases).
5. Exclude explanations or extra formatting; return only the JSON text.

**Example**:
User Query: "What grants does Valley Air provide?"
Output:
{{
  "rewrites": [
    "Available grants from Valley Air District",
    "Financial assistance programs at Valley Air",
    "Funding opportunities for businesses and residents from Valley Air"
  ],
  "keywords": [
    "Valley Air grants",
    "financial assistance",
    "funding programs",
    "incentives",
    "business grants"
  ]
}}
<|end_of_text|>
<|start_of_role|>user<|end_of_role|>{query}<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
"""
)

class AgentState(TypedDict):
    user_query: str
    rewrites: List[str]
    keywords: List[str]
    retrieved_docs: List[Any]
    answer: str
    sources: List[Dict[str, Any]]
    messages: Annotated[list, add_messages]

class QueryContextAgent:
    def __init__(self, llm):
        self.llm = llm
    def __call__(self, state: AgentState) -> Dict:
        user_query = state.get("user_query")
        prompt = REWRITE_AND_KEYWORDS_PROMPT.format(query=user_query)
        response = self.llm.invoke(prompt)
        import json
        try:
            data = json.loads(response)
            rewrites = data.get("rewrites", [])
            keywords = data.get("keywords", [])
        except Exception:
            rewrites = [user_query]
            keywords = user_query.split()
        print(f"Rewrites: {rewrites}")
        print(f"Keywords: {keywords}")
        return {"user_query": user_query, "rewrites": rewrites, "keywords": keywords, "messages": state.get("messages", [])}
    def stream(self, state: AgentState, callback_handler=None):
        user_query = state.get("user_query")
        prompt = REWRITE_AND_KEYWORDS_PROMPT.format(query=user_query)
        response = self.llm.invoke(prompt)
        import json
        try:
            data = json.loads(response)
            rewrites = data.get("rewrites", [])
            keywords = data.get("keywords", [])
        except Exception:
            rewrites = [user_query]
            keywords = user_query.split()
        event = {"type": "tool", "tool": "QueryContextAgent", "description": "Generated rewrites and keywords.", "rewrites": rewrites, "keywords": keywords}
        if callback_handler is not None:
            callback_handler.on_tool_start("QueryContextAgent", "Generated rewrites and keywords.")
        yield event
        done_event = {"type": "query_context_done", "rewrites": rewrites, "keywords": keywords}
        yield done_event 