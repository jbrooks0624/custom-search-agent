import os
from dotenv import load_dotenv

load_dotenv()

if api_key := os.getenv("openai_api_key"):
    os.environ["OPENAI_API_KEY"] = api_key
if api_key := os.getenv("tavily_api_key"):
    os.environ["TAVILY_API_KEY"] = api_key

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from oai import OAI, Message
from tvly import Tavily
from workflow import run_search_pipeline, run_deep_research_pipeline

app = FastAPI(
    title="Search Agent API",
    description="RAG-powered search chatbot with standard and deep research modes",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oai_client = OAI()
tavily_client = Tavily()


class ChatRequest(BaseModel):
    """Request payload for the chat endpoint."""
    messages: list[Message] = Field(
        description="List of conversation messages"
    )
    deep_research: bool = Field(
        default=False,
        description="If True, use deep research mode with iterative search loops"
    )


class ChatResponse(BaseModel):
    """Response payload from the chat endpoint."""
    messages: list[Message] = Field(
        description="Updated list of messages including the assistant response"
    )
    iterations: int = Field(
        default=1,
        description="Number of research iterations performed"
    )
    total_ms: float = Field(
        description="Total time taken in milliseconds"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a chat request and return the assistant's response.
    
    Standard mode (~27s): Single-pass RAG with 3 queries
    Deep research mode (~45-150s): Iterative RAG with up to 3 iterations and 5 queries each
    """
    messages = request.messages.copy()
    
    if request.deep_research:
        timings = await run_deep_research_pipeline(
            oai_client=oai_client,
            tavily_client=tavily_client,
            messages=messages,
            return_timings=True,
        )
    else:
        timings = await run_search_pipeline(
            oai_client=oai_client,
            tavily_client=tavily_client,
            messages=messages,
            return_timings=True,
        )
    
    return ChatResponse(
        messages=messages,
        iterations=timings.num_iterations if timings else 1,
        total_ms=timings.total_ms if timings else 0,
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
