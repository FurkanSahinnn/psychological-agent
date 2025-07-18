from flask import Flask, request, jsonify
from langgraph_pipeline import psychology_agent, AgentState

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({"error": "Soru boş olamaz"}), 400
    
    # Agent'i çalıştır
    initial_state = AgentState(user_query=question, retrieved_documents=None, final_prompt=None, final_response=None)
    events = psychology_agent.stream(initial_state)
    
    final_response = ""
    for event in events:
        if "final_response" in event.get("generator", {}):
            final_response = event["generator"]["final_response"]
    
    return jsonify({
        "question": question,
        "answer": final_response
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True) 