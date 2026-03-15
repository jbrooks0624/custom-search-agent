import os
import json
import asyncio
from typing import AsyncGenerator

from dotenv import load_dotenv

load_dotenv()

if api_key := os.getenv("openai_api_key"):
    os.environ["OPENAI_API_KEY"] = api_key
if api_key := os.getenv("tavily_api_key"):
    os.environ["TAVILY_API_KEY"] = api_key

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from oai import OAI, Message
from tvly import Tavily
from workflow import run_search_pipeline_with_status, run_deep_research_pipeline_with_status

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


class SourceInfo(BaseModel):
    """Source information."""
    title: str
    url: str


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
    sources: list[SourceInfo] = Field(
        default_factory=list,
        description="Sources used in the response"
    )


async def generate_chat_stream(request: ChatRequest) -> AsyncGenerator[str, None]:
    """Generate SSE stream with status updates and final response."""
    messages = request.messages.copy()
    status_queue: asyncio.Queue[str] = asyncio.Queue()
    
    async def status_callback(status: str):
        await status_queue.put(status)
    
    async def run_pipeline():
        if request.deep_research:
            return await run_deep_research_pipeline_with_status(
                oai_client=oai_client,
                tavily_client=tavily_client,
                messages=messages,
                status_callback=status_callback,
            )
        else:
            return await run_search_pipeline_with_status(
                oai_client=oai_client,
                tavily_client=tavily_client,
                messages=messages,
                status_callback=status_callback,
            )
    
    pipeline_task = asyncio.create_task(run_pipeline())
    
    while not pipeline_task.done():
        try:
            status = await asyncio.wait_for(status_queue.get(), timeout=0.1)
            yield f"data: {json.dumps({'status': status})}\n\n"
        except asyncio.TimeoutError:
            continue
    
    while not status_queue.empty():
        status = await status_queue.get()
        yield f"data: {json.dumps({'status': status})}\n\n"
    
    timings = pipeline_task.result()
    
    sources = [SourceInfo(title=s.title, url=s.url) for s in timings.sources] if timings else []
    
    response = ChatResponse(
        messages=messages,
        iterations=timings.num_iterations if timings else 1,
        total_ms=timings.total_ms if timings else 0,
        sources=sources,
    )
    yield f"data: {json.dumps({'done': True, **response.model_dump()})}\n\n"


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Process a chat request and stream status updates.
    
    Standard mode (~27s): Single-pass RAG with 3 queries
    Deep research mode (~45-150s): Iterative RAG with up to 3 iterations and 5 queries each
    """
    return StreamingResponse(
        generate_chat_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
