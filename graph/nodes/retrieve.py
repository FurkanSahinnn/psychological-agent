from typing import Any, Dict

from graph.state import GraphState
from ingestion import get_related_documents


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    documents = get_related_documents(question, 3)
    return {"documents": documents, "question": question}