from typing import List, Dict, Any

def format_retrieved_documents(documents: List[Dict[str, Any]]) -> str:
    """
    Formats the retrieved documents from Qdrant into a text format that the LLM can understand.
    """
    formatted_docs = []
    for i, doc in enumerate(documents):
        if doc.get("error"):
            continue
        
        document_text = doc.get('document', 'Content not found.')
        source = doc.get('source', 'Unknown Source')
        page = doc.get('page', 'Unknown Page')
        
        formatted_docs.append(
            f"--- Document {i+1} (Source: {source}, Page: {page}) ---\n"
            f"{document_text}\n"
            f"--- Document {i+1} End ---"
        )
    
    if not formatted_docs:
        return "No document found related to the user's query."
        
    return "\n\n".join(formatted_docs)


def build_agent_prompt(user_query: str, retrieved_documents: List[Dict[str, Any]]) -> str:
    """
    Builds a complete system prompt for the LangGraph agent using the user query and retrieved documents.
    """
    formatted_documents_str = format_retrieved_documents(retrieved_documents)

    prompt_template = f"""
# TASK AND ROLE
You are a specialized AI assistant in the field of psychology. Your task is to synthesize the information from the provided academic documents into a coherent and comprehensive text, answering the user's question. Your answers should be based solely on the documents provided to you.

# ANSWERING RULES
1.  **Never use information from outside the provided documents:** Do not use information from outside the provided documents in your answer. If the documents are insufficient to answer the question, explicitly state this.
2.  **Use academic and professional language:** Your answer should be written in a way similar to how a psychologist would answer, with clear, informative, and professional language.
3.  **Dont provide references:** Do not use `[Document 1]` etc. style references in your answer. Integrate the information naturally into a coherent text, similar to how a psychologist would present their expert opinion.
4.  **Provide a comprehensive and structured answer:** Address the user's question comprehensively. Structure your answer logically, using headings or lists if necessary.
5.  **Be direct and clear:** Start your answer directly with the user's question, then provide details.

# PROVIDED ACADEMIC DOCUMENTS
The following are the relevant academic texts that you should use to answer the user's question:
{formatted_documents_str}

# USER QUESTION
Here is the question you need to answer:
"{user_query}"

# ANSWER
Please follow the rules above and the documents provided, without mentioning references, and create a comprehensive answer.
"""
    
    return prompt_template.strip()
