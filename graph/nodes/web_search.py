from typing import Any, Dict

from langchain_community.tools import DuckDuckGoSearchResults
from langchain.schema import Document
from graph.state import GraphState


web_search_tool = DuckDuckGoSearchResults(output_format="json")

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB-SEARCH---")
    question = state["question"]
    documents = state["documents"]

    web_results = web_search_tool.invoke(question) # JSON string
    web_results = Document(page_content=web_results)

    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {
        "question": question,
        "documents": documents,
    }