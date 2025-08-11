from dotenv import load_dotenv
from flask import Flask, request, jsonify
from graph.graph import app

load_dotenv()

"""
flask_app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data['question']

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # define agent state
    pass
"""
if __name__ == "__main__":
    #flask_app.run(port=5000, debug=True)
    while True:
        user_input = input("What is your question about psychology? ")
        response = app.invoke(
            {
                "question": user_input,
                "documents": [],
                "web_search": False,
                "generation": ""
            }
        )
        print(response["generation"])
