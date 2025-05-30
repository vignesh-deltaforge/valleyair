"""
chat_app.py: Entry point for CLI and Flask app.

- All configuration, LLM, vectorstore, and agent logic has been moved to separate modules.
- For CLI: uses workflow.py
- For Flask: uses web/routes.py
- For tests: use tests/test_es.py
- For Streamlit: use app.py (untouched)
"""

import argparse
from workflow import run_multiagent_workflow
from tests.test_es import run_tests

# Optionally import Flask app if needed
try:
    from web.routes import app as flask_app
except ImportError:
    flask_app = None

def main():
    print("Welcome to the Valley Air RAG Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        answer, sources = run_multiagent_workflow(user_input)
        print("AI:", answer)
        print("Sources:")
        for src in sources:
            print("-", src.get("url", "No URL"))
            print("  Metadata:", src)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run ES and embedding tests')
    parser.add_argument('--web', action='store_true', help='Run the Flask web chat UI')
    args = parser.parse_args()
    if args.test:
        run_tests()
    elif args.web and flask_app is not None:
        flask_app.run(host="0.0.0.0", port=5001, debug=True)
    else:
        main() 