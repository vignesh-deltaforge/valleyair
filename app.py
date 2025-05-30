import streamlit as st
from workflow import run_multiagent_workflow_streaming
from datetime import datetime
import torch

torch.classes.__path__ = []

st.set_page_config(page_title="Valley Air RAG Chatbot", page_icon="\U0001F4A8", layout="wide")

# --- Session State Initialization ---
if "history" not in st.session_state:
    st.session_state["history"] = []  # List of dicts: {role, content, sources, timestamp}
if "tool_events" not in st.session_state:
    st.session_state["tool_events"] = []
if "pending_ai" not in st.session_state:
    st.session_state["pending_ai"] = False
if "last_user_message" not in st.session_state:
    st.session_state["last_user_message"] = None

# --- Helper: Format timestamp ---
def format_time(ts):
    return ts.strftime("%H:%M")

# --- Top right controls ---
col1, col2, col3 = st.columns([8, 1, 1])
with col2:
    if st.button("🔄", help="Reset Chat"):
        st.session_state["history"] = []
        st.session_state["tool_events"] = []
        st.session_state["pending_ai"] = False
        st.session_state["last_user_message"] = None
        st.rerun()
with col3:
    st.download_button(
        "⬇️",
        data="\n".join(
            f"[{msg['role'].upper()} {msg['timestamp'].strftime('%H:%M')}]: {msg['content']}"
            for msg in st.session_state.get("history", [])
        ),
        file_name="valleyair_chat_transcript.txt",
        mime="text/plain",
        help="Download your chat as a text file",
    )

# --- Main Chat Container (scrollable) ---
st.title("\U0001F4A8 Valley Air RAG Chatbot")
st.markdown(
    "<div style='color:gray;font-size:0.95em;'>Ask anything about air quality, grants, permits, or Valley Air services. "
    "Your conversation is private and not stored.</div>",
    unsafe_allow_html=True,
)

# --- Main Chat Container ---
chat_container = st.container()
with chat_container:
    for msg in st.session_state.get("history", []):
        with st.chat_message(msg["role"]):
            if msg["role"] == "query_context":
                st.markdown(
                    f"<div style='background:#e8f5e9;padding:0.5em 1em;border-radius:6px;margin-bottom:1em;'>{msg['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(msg["content"], unsafe_allow_html=True)
            # Print sources as a markdown list after the answer (not in expander)
            if msg["role"] == "ai" and msg.get("sources") is not None:
                if msg["sources"]:
                    st.markdown("**Sources:**")
                    for src in msg["sources"]:
                        st.markdown(f"- {src['url']}")
                else:
                    st.markdown("_No sources available._")
            st.caption(msg["timestamp"].strftime("%H:%M"))

# --- Tool Events (subtle, not in chat) ---
# if st.session_state.get("tool_events"):
#     with st.expander("Show Tool Events", expanded=False):
#         for event in st.session_state["tool_events"]:
#             st.info(f"🛠️ {event.get('tool', 'Unknown')}: {event.get('description', '')}")

# --- Query Context (as a status, not a chat message) ---
if st.session_state.get("last_query_context"):
    st.markdown(
        f"<div style='background:#e8f5e9;padding:0.5em 1em;border-radius:6px;margin-bottom:1em;'>"
        f"<b>AI Context:</b><br>{st.session_state['last_query_context']}</div>",
        unsafe_allow_html=True,
    )

# --- Streamlit Callback Handler ---
class StreamlitCallbackHandler:
    def __init__(self, ai_msg_placeholder, tool_event_callback):
        self.ai_msg_placeholder = ai_msg_placeholder
        self.tool_event_callback = tool_event_callback
        self.streamed_answer = ""
        self.sources = []
        self.query_context_content = None

    def on_llm_new_token(self, token):
        self.streamed_answer += token
        self.ai_msg_placeholder.markdown(self.streamed_answer + "▌")

    def on_tool_start(self, tool, description):
        self.tool_event_callback(tool, description)

    def on_query_context(self, rewrites, keywords):
        self.query_context_content = (
            "<b>Expanded Queries:</b><br>" + "<br>".join([f"- {r}" for r in rewrites]) +
            "<br><b>BM25 Keywords:</b> " + ", ".join(keywords)
        )

    def on_done(self, sources):
        self.ai_msg_placeholder.markdown(self.streamed_answer)
        self.sources = sources

# --- Tool Event Callback ---
def add_tool_event(tool, description):
    st.session_state["tool_events"].append({
        "tool": tool,
        "description": description,
        "timestamp": datetime.now()
    })

# --- Chat input at the bottom ---
user_input = st.chat_input("Type your question...")
if user_input and not st.session_state["pending_ai"]:
    st.session_state["history"].append({
        "role": "user",
        "content": user_input.strip(),
        "timestamp": datetime.now(),
    })
    st.session_state["pending_ai"] = True
    st.session_state["last_user_message"] = user_input.strip()
    st.rerun()

# --- Workflow execution after rerun ---
if st.session_state["pending_ai"] and st.session_state["last_user_message"]:
    with chat_container:
        ai_msg_placeholder = st.chat_message("ai")
        ai_msg = ai_msg_placeholder.empty()
    callback_handler = StreamlitCallbackHandler(ai_msg, add_tool_event)
    query_context_html = None
    ai_sources = []
    for event in run_multiagent_workflow_streaming(st.session_state["last_user_message"], callback_handler=callback_handler):
        print("DEBUG: Event in app loop:", event)
        if event["type"] == "tool" and event["tool"] == "QueryContextAgent":
            callback_handler.on_query_context(event.get("rewrites", []), event.get("keywords", []))
            query_context_html = callback_handler.query_context_content
        elif event["type"] == "done":
            callback_handler.on_done(event.get("sources", []))
            ai_sources = event.get("sources", [])
    if query_context_html:
        st.session_state["history"].append({
            "role": "query_context",
            "content": query_context_html,
            "timestamp": datetime.now(),
        })
    print("DEBUG: AI sources:", ai_sources)
    st.session_state["history"].append({
        "role": "ai",
        "content": callback_handler.streamed_answer,
        "sources": ai_sources if ai_sources else [],
        "timestamp": datetime.now(),
    })
    st.session_state["pending_ai"] = False
    st.session_state["last_user_message"] = None
    st.rerun()