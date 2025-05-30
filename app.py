import streamlit as st
from workflow import run_multiagent_workflow_streaming
import time

st.set_page_config(page_title="Valley Air RAG Chatbot", page_icon="💨", layout="wide")

# --- Session State Initialization ---
if "history" not in st.session_state:
    st.session_state["history"] = []  # List of dicts: {role, content, sources, tool_events}
if "tool_events" not in st.session_state:
    st.session_state["tool_events"] = []  # List of tool call event dicts

st.title("Valley Air RAG Chatbot (Streamlit)")

# --- Display Chat History ---
for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "ai" and msg.get("sources"):
            st.markdown("**Sources:**")
            for src in msg["sources"]:
                st.markdown(f"- [{src['url']}]({src['url']})")
        if msg["role"] == "tool":
            st.info(msg["content"])
        if msg["role"] == "query_context":
            st.success(msg["content"])

# --- Display Tool Call Events (if any) ---
if st.session_state["tool_events"]:
    with st.expander("Show Tool Call Events", expanded=False):
        for event in st.session_state["tool_events"]:
            st.info(f"Tool: {event.get('tool', 'Unknown')} | {event.get('description', '')}")

# --- User Input ---
user_input = st.chat_input("Type your question...")
if user_input:
    # 1. Append user message
    st.session_state["history"].append({"role": "user", "content": user_input})
    # 2. Display chat history up to this point (user message only)
    for msg in st.session_state["history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    # 3. Run streaming workflow
    query_context_content = None
    # Wait for QueryContextAgent event
    for event in run_multiagent_workflow_streaming(user_input):
        if event["type"] == "tool" and event["tool"] == "QueryContextAgent":
            rewrites = event.get("rewrites", [])
            keywords = event.get("keywords", [])
            query_context_content = (
                "**Expanded Queries:**\n" + "\n".join([f"- {r}" for r in rewrites]) +
                "\n\n**BM25 Keywords:**\n" + ", ".join(keywords)
            )
            # Show the query context as a green box below the user message
            with st.chat_message("query_context"):
                st.success(query_context_content)
            break
    # 4. Stream the synthesis answer below the query context
    streamed_answer = ""
    sources = []
    with st.chat_message("ai"):
        answer_placeholder = st.empty()
        for event in run_multiagent_workflow_streaming(user_input):
            if event["type"] == "token":
                streamed_answer += event["token"]
                answer_placeholder.markdown(streamed_answer + "▌")
            elif event["type"] == "done":
                sources = event.get("sources", [])
        answer_placeholder.markdown(streamed_answer)
        if sources:
            st.markdown("**Sources:**")
            for src in sources:
                st.markdown(f"- [{src['url']}]({src['url']})")
    # 5. Append query context and answer to history for future display
    st.session_state["history"].append({"role": "query_context", "content": query_context_content})
    st.session_state["history"].append({"role": "ai", "content": streamed_answer, "sources": sources})
    st.rerun()

# --- Session Reset Button ---
if st.button("Reset Chat"):
    st.session_state["history"] = []
    st.session_state["tool_events"] = []
    st.rerun()

# --- Footer ---
st.markdown("<hr style='margin-top:2em;margin-bottom:1em;'>", unsafe_allow_html=True)
st.markdown("<small>Powered by Streamlit, LangGraph, and IBM Watsonx. <a href='https://valleyair.org' target='_blank'>valleyair.org</a></small>", unsafe_allow_html=True) 