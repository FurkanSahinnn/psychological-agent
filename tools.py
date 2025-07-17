from langchain_core.tools import tool
from typing import List, Dict, Any
from conf import embedding_client, qdrant_client, QDRANT_COLLECTION_NAME, EMBEDDING_MODEL

@tool
def search_psychology_knowledge_base(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Searches the psychology knowledge base in Qdrant for the most relevant results for the user's psychology-related question.
    Returns the top 'top_k' results.
    """
    try:
        # Convert the user query to a vector using the embedding model
        query_embedding = embedding_client.embeddings.create(
            input=[query],
            model=EMBEDDING_MODEL
        ).data[0].embedding

        # Perform semantic search in Qdrant
        search_results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )

        # Format the results and return them
        formatted_results = []
        for result in search_results:
            if result.payload:
                formatted_results.append({
                    "score": result.score,
                    "document": result.payload.get("text", ""),
                    "source": result.payload.get("source_title", "Bilinmeyen Kaynak"),
                    "page": result.payload.get("page_number", 0)
                })
        
        return formatted_results

    except Exception as e:
        print(f"An error occurred while searching the knowledge base: {e}")
        return [{"error": "An error occurred while searching the knowledge base."}]

# Add the search tool to the list. LangGraph will use this list.
tools = [search_psychology_knowledge_base]
