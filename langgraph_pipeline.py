from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from conf import llm_client, LLM_MODEL
from tools import search_psychology_knowledge_base
import json

class AgentState(TypedDict):
    user_query: str
    search_results: Optional[str]
    final_response: Optional[str]

def search_node(state: AgentState) -> dict:
    """Always search first"""
    print("--- 🔍 Searching knowledge base... ---")
    
    results = search_psychology_knowledge_base.invoke({"query": state["user_query"]})
    
    # Format results for LLM
    formatted_results = []
    for i, result in enumerate(results):
        if "error" not in result:
            formatted_results.append(
                f"Document {i+1} (Source: {result['source']}, Page: {result['page']}):\n"
                f"{result['document']}\n"
            )
    
    search_text = "\n".join(formatted_results) if formatted_results else "No relevant documents found."
    print(f"--- ✅ Found {len(results)} documents ---")
    
    return {"search_results": search_text}

def generate_node(state: AgentState) -> dict:
    """Generate response based on search results"""
    print("--- 🤖 Generating response... ---")
    
    prompt = f"""You are an expert psychologist. Based on the search results below, answer the user's question.

SEARCH RESULTS:
{state['search_results']}

USER QUESTION:
{state['user_query']}

IMPORTANT RULES:
1. Only use information from the search results above
2. Always cite sources: (Source: Book Name, Page: X)
3. Use academic and professional language
4. Respond in the same language as the question
5. If search results are insufficient, say so clearly

Provide a comprehensive answer with proper citations."""

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    final_answer = response.choices[0].message.content
    print("--- ✅ Response generated ---")
    
    return {"final_response": final_answer}

def create_psychology_agent_graph():
    """Simple two-step workflow: search -> generate"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("search", search_node)
    workflow.add_node("generate", generate_node)
    
    # Define flow: search -> generate -> end
    workflow.set_entry_point("search")
    workflow.add_edge("search", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

psychology_agent = create_psychology_agent_graph()
