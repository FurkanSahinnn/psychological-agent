from flask import Flask, request, jsonify
from langgraph_pipeline import psychology_agent, AgentState

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400
    
    print(f"📝 Question received: {question}")
    
    # Simple state initialization
    initial_state = AgentState(
        user_query=question,
        search_results=None,
        final_response=None
    )
    
    # Run the pipeline
    try:
        result = psychology_agent.invoke(initial_state)
        final_response = result.get("final_response", "Sorry, I couldn't generate a response.")
        
        print(f"✅ Answer ready: {len(final_response)} characters")
        
        return jsonify({
            "question": question,
            "answer": final_response
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({
            "question": question,
            "answer": f"An error occurred: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True) 