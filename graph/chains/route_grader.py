from dotenv import load_dotenv
load_dotenv()

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from ..chain_constants import router_system_prompt, model_name
import os


class RouterQuery(BaseModel):
  """Route a user query to the most relevant datasource."""
  datasource: Literal["vectorstore", "websearch"] = Field(
    ...,
    description="Given a user question choose to route it to web seach or a vectorstore."
  )

llm = ChatOpenAI(
  api_key=os.getenv("OPENROUTER_API_KEY"),
  base_url="https://openrouter.ai/api/v1",
  model=model_name,
)
structured_llm_router = llm.with_structured_output(RouterQuery)

router_prompt = ChatPromptTemplate.from_messages(
  [
    ("system", router_system_prompt),
    ("human", "{question}")
  ]
)

question_router = router_prompt | structured_llm_router