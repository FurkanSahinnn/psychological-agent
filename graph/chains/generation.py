from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from ..chain_constants import generation_chat_prompt, model_name
import os


llm = ChatOpenAI(
  api_key=os.getenv("OPENROUTER_API_KEY"),
  base_url="https://openrouter.ai/api/v1",
  model=model_name,
)
prompt = ChatPromptTemplate.from_template(generation_chat_prompt)

generation_chain = prompt | llm | StrOutputParser()