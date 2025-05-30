from flask import Flask, request, jsonify, render_template_string
from workflow import run_multiagent_workflow

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

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template_string(CHAT_TEMPLATE)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"answer": "Please provide a message.", "sources": []})
    answer, sources = run_multiagent_workflow(user_message)
    return jsonify({
        "answer": answer,
        "sources": [{"url": s.get("url", ""), "title": s.get("title", "Untitled")} for s in sources]
    }) 