import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

load_dotenv()

# Environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_IP_ADDRESS = os.getenv("QDRANT_IP_ADDRESS")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not found!")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not found!")
if not QDRANT_IP_ADDRESS:
    raise ValueError("QDRANT_IP_ADDRESS environment variable not found!")

# Clients
llm_client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_API_KEY,
)

embedding_client = OpenAI(api_key=OPENAI_API_KEY)

qdrant_client = QdrantClient(
    url=f"http://{QDRANT_IP_ADDRESS}:6333", 
    api_key=QDRANT_API_KEY,
    timeout=60
)

# Constants
QDRANT_COLLECTION_NAME = "psychology_knowledge_base"
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "google/gemini-2.5-flash"