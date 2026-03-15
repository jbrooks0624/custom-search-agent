import os
from dotenv import load_dotenv

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
else:
    raise RuntimeError("OPENAI_API_KEY not found in environment. Create a .env file.")

tavily_key = os.getenv("TAVILY_API_KEY") or os.getenv("tavily_api_key")
if tavily_key:
    os.environ["TAVILY_API_KEY"] = tavily_key
else:
    raise RuntimeError("TAVILY_API_KEY not found in environment. Create a .env file.")
