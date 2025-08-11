import os
import getpass
from typing import List, Dict, Union
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from qdrant_client.http.models import QueryResponse
from openai import OpenAI
from langchain.schema import Document

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_IP_ADDRESS = os.getenv("QDRANT_IP_ADDRESS")

embedding_client = OpenAI(api_key=OPENAI_API_KEY)

qdrant_client = QdrantClient(
    url=f"http://{QDRANT_IP_ADDRESS}:6333",
    api_key=QDRANT_API_KEY,
    timeout=120
)

def format_qdrant_results(results: Union[List[ScoredPoint], QueryResponse]) -> List[Document]:
    """
    Convert from Qdrant results to LangChain Document Lists.
    """
    points = results.points if hasattr(results, "points") else results  # QueryResponse vs list
    documents: List[Document] = []
    for p in points:
        payload = p.payload or {}
        text = payload.get("text", "")
        if not text:
            # Skip empty contents
            continue
        metadata = {
            "source": payload.get("source_title", ""),
            "file": payload.get("source_file", ""),
            "page": payload.get("page_number", 0),
            "id": str(getattr(p, "id", "")) if getattr(p, "id", None) is not None else "",
            "score": getattr(p, "score", None),
        }
        documents.append(Document(page_content=text, metadata=metadata))
    return documents

def get_related_documents(query: str, top_k: int) -> List[Document]:
    query_embedding = embedding_client.embeddings.create(
        input=query,
        model="text-embedding-3-large"
    ).data[0].embedding

    search_results = qdrant_client.query_points(
        collection_name="psychology_knowledge_base",
        query=query_embedding,
        limit=top_k,
        with_payload=True
    )
    return format_qdrant_results(search_results)