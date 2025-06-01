from langchain.prompts import PromptTemplate
from typing import Dict, Any

SYTHESIS_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
<|start_of_role|>system<|end_of_role|>You are Granite, developed by xAI, acting as an AI assistant for the San Joaquin Valley Air Pollution Control District (Valley Air), dedicated to improving air quality in California's Central Valley. Your goal is to provide accurate, concise, and helpful answers based on valleyair.org content and the provided context. Today's date: May 30, 2025.

**Instructions**:
1. Use the provided context from valleyair.org and any real-time air quality data to answer the user's question in 1-2 sentences.
2. Adopt a friendly, professional tone, explaining technical terms (e.g., AQI, PM2.5) in simple language for residents, businesses, and community members.
3. If the question seeks details (e.g., "benefits"), include a short bulleted list of specific points (e.g., financial, environmental benefits).
4. Suggest a follow-up action (e.g., visit valleyair.org/grants, call 559-230-5800).
5. If context is insufficient, state: "I don't have enough details to answer fully. Visit valleyair.org or call 559-230-5800."
6. For vague questions, suggest clarification (e.g., "Can you specify what you mean?").
7. For off-topic queries, redirect politely (e.g., "I focus on air quality and Valley Air services. How can I help?").
8. For sensitive topics, respond empathetically and suggest contact (e.g., "I'm sorry for your concern. Contact Valley Air at 559-230-5800.").
9. For real-time data (e.g., AQI), direct to valleyair.org/air-quality.
10. Output only the answer text, excluding structural markers or tokens.

**Example**:
Context: Valley Air offers grants for clean vehicles and equipment.
User Question: What grants does Valley Air provide?
Output: Valley Air provides grants for clean vehicles and equipment to reduce emissions. Visit valleyair.org/grants for details.
<|end_of_text|>
<|start_of_role|>user<|end_of_role|>Context:
{context}

User question: {question}<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
"""
)

class ResponseSynthesisAgent:
    def __init__(self, llm):
        self.llm = llm
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        docs = state.get("retrieved_docs", [])
        air_quality = state.get("air_quality_data")
        context = "\n\n".join([doc.page_content if hasattr(doc, 'page_content') else doc["content"] for doc in docs])
        if air_quality:
            aq_context = f"\n\n[Real-time Air Quality]\nAQI: {air_quality.get('aqi')} ({air_quality.get('aqi_category')})\nPM2.5: {air_quality.get('pm2_5')} µg/m³\nOzone: {air_quality.get('ozone')} ppb\nSource: https://open-meteo.com/en/docs/air-quality-api"
            context = aq_context + "\n" + context
        user_query = state.get("user_query", "")
        prompt = SYTHESIS_PROMPT.format(context=context, question=user_query)
        answer = self.llm.invoke(prompt)
        sources = []
        seen_urls = set()
        for doc in docs:
            meta = {}
            if hasattr(doc, 'metadata') and doc.metadata:
                meta = doc.metadata
            elif isinstance(doc, dict):
                meta = {"url": doc.get("url", ""), "title": doc.get("title", "Untitled")}
            if not meta.get("url") and hasattr(doc, 'url'):
                meta["url"] = getattr(doc, 'url', "")
            if not meta.get("title") and hasattr(doc, 'title'):
                meta["title"] = getattr(doc, 'title', "Untitled")
            if not meta.get("url"):
                meta["url"] = "No URL"
            if not meta.get("title"):
                meta["title"] = "Untitled"
            url = meta["url"]
            if url not in seen_urls:
                sources.append(meta)
                seen_urls.add(url)
        # Only add air quality API source if it is present in the state
        if "sources" in state and state["sources"]:
            sources.extend(state["sources"])
        return {**state, "answer": answer, "sources": sources}

class StreamingResponseSynthesisAgent:
    def __init__(self, llm):
        self.llm = llm
    def stream(self, state: Dict[str, Any], callback_handler=None):
        docs = state.get("retrieved_docs", [])
        air_quality = state.get("air_quality_data")
        context = "\n\n".join([doc.page_content if hasattr(doc, 'page_content') else doc["content"] for doc in docs])
        if air_quality:
            aq_context = f"\n\n[Real-time Air Quality]\nAQI: {air_quality.get('aqi')} ({air_quality.get('aqi_category')})\nPM2.5: {air_quality.get('pm2_5')} µg/m³\nOzone: {air_quality.get('ozone')} ppb\nSource: https://open-meteo.com/en/docs/air-quality-api"
            context = aq_context + "\n" + context
        user_query = state.get("user_query", "")
        prompt = SYTHESIS_PROMPT.format(context=context, question=user_query)
        sources = []
        seen_urls = set()
        for doc in docs:
            meta = {}
            if hasattr(doc, 'metadata') and doc.metadata:
                meta = doc.metadata
            elif isinstance(doc, dict):
                meta = {"url": doc.get("url", ""), "title": doc.get("title", "Untitled")}
            if not meta.get("url") and hasattr(doc, 'url'):
                meta["url"] = getattr(doc, 'url', "")
            if not meta.get("title") and hasattr(doc, 'title'):
                meta["title"] = getattr(doc, 'title', "Untitled")
            if not meta.get("url"):
                meta["url"] = "No URL"
            if not meta.get("title"):
                meta["title"] = "Untitled"
            url = meta["url"]
            if url not in seen_urls:
                sources.append(meta)
                seen_urls.add(url)
        # Only add air quality API source if it is present in the state
        if "sources" in state and state["sources"]:
            for extra in state["sources"]:
                url = extra.get("url", "No URL")
                if url not in seen_urls:
                    sources.append(extra)
                    seen_urls.add(url)
        answer = ""
        for chunk in self.llm.stream(prompt):
            if callback_handler is not None:
                callback_handler.on_llm_new_token(chunk)
            else:
                yield {"type": "token", "token": chunk}
            answer += chunk
        # Yield the synthesized answer before done
        yield {"type": "answer", "content": answer, "sources": sources}
        yield {"type": "done", "sources": sources} 