from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator
from conf import llm_client, LLM_MODEL
from tools import search_psychology_knowledge_base
from build_prompt import build_agent_prompt

class AgentState(TypedDict):
    user_query: str
    retrieved_documents: Optional[List[Dict[str, Any]]]
    final_prompt: Optional[str]
    final_response: Optional[str]

def retrieve_documents_node(state: AgentState) -> dict:
    """
    Retrieves relevant documents from Qdrant using the user's query.
    """
    print("--- 🧠 Searching the knowledge base... ---")
    user_query = state["user_query"]
    retrieved_documents = search_psychology_knowledge_base.invoke({"query": user_query})
    
    return {"retrieved_documents": retrieved_documents}

def generate_response_node(state: AgentState) -> dict:
    """
    Generates a response from the LLM using the retrieved documents and the user's query.
    """
    print("--- 🤖 LLM is generating a response... ---")
    user_query = state["user_query"]
    retrieved_documents = state.get("retrieved_documents") or []

    final_prompt = build_agent_prompt(user_query, retrieved_documents)

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.3,
    )
    final_response = response.choices[0].message.content or "I'm sorry, I couldn't generate a response."
    
    return {"final_prompt": final_prompt, "final_response": final_response}

def create_psychology_agent_graph():
    """
    Creates a LangGraph workflow (pipeline), defines nodes and edges.
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("retriever", retrieve_documents_node)
    workflow.add_node("generator", generate_response_node)

    workflow.set_entry_point("retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", END)

    return workflow.compile()

psychology_agent = create_psychology_agent_graph()
