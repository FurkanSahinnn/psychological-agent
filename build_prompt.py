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
3.  **Always provide source information:** When providing information, always indicate which book and page number the information comes from. Use the format: (Source: [Book Name], Page: [Page Number]) after each piece of information.
4.  **Answer in the same language as the user's question:** Detect the language of the user's question and respond in the same language. If the question is in English, respond in English. If the question is in Turkish, respond in Turkish. If the question is in another language, respond in that language.
5.  **Provide a comprehensive and structured answer:** Address the user's question comprehensively. Structure your answer logically, using headings or lists if necessary.
6.  **Be direct and clear:** Start your answer directly with the user's question, then provide details.
7.  **Source citation format:** When citing information, use this format: "Information content here (Source: Book Name, Page: X)"

# PROVIDED ACADEMIC DOCUMENTS
The following are the relevant academic texts that you should use to answer the user's question:
{formatted_documents_str}

# USER QUESTION
Here is the question you need to answer:
"{user_query}"

# ANSWER
Please follow the rules above and the documents provided. Create a comprehensive answer while ALWAYS including source information (book name and page number) for each piece of information you provide. Use the format: (Source: Book Name, Page: X) after each important statement or fact. Remember to respond in the same language as the user's question.
"""
    
    return prompt_template.strip()
