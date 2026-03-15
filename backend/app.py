import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator

from dotenv import load_dotenv

load_dotenv()

if api_key := os.getenv("openai_api_key"):
    os.environ["OPENAI_API_KEY"] = api_key
if api_key := os.getenv("tavily_api_key"):
    os.environ["TAVILY_API_KEY"] = api_key

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import APIConnectionError, APIError, RateLimitError
from pydantic import BaseModel, Field, field_validator

from oai import OAI, Message
from tvly import Tavily
from workflow import run_deep_research_pipeline_with_status, run_search_pipeline_with_status

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


class ErrorResponse(BaseModel):
    """Structured error response."""

    error: bool = True
    code: str
    message: str
    retry_after: int | None = None


MAX_MESSAGE_LENGTH = 10000
MAX_MESSAGES = 20


class ChatRequest(BaseModel):
    """Request payload for the chat endpoint."""

    messages: list[Message] = Field(
        description="List of conversation messages",
        min_length=1,
        max_length=MAX_MESSAGES,
    )
    deep_research: bool = Field(
        default=False, description="If True, use deep research mode with iterative search loops"
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, messages: list[Message]) -> list[Message]:
        for i, msg in enumerate(messages):
            if not msg.content or not msg.content.strip():
                raise ValueError(f"Message {i + 1} cannot be empty")
            if len(msg.content) > MAX_MESSAGE_LENGTH:
                raise ValueError(
                    f"Message {i + 1} exceeds maximum length of {MAX_MESSAGE_LENGTH} characters"
                )
            if msg.role not in ("user", "assistant", "system"):
                raise ValueError(f"Message {i + 1} has invalid role: {msg.role}")
        return messages


class SourceInfo(BaseModel):
    """Source information."""

    title: str
    url: str


class ChatResponse(BaseModel):
    """Response payload from the chat endpoint."""

    messages: list[Message] = Field(
        description="Updated list of messages including the assistant response"
    )
    iterations: int = Field(default=1, description="Number of research iterations performed")
    total_ms: float = Field(description="Total time taken in milliseconds")
    sources: list[SourceInfo] = Field(
        default_factory=list, description="Sources used in the response"
    )


def create_error_event(code: str, message: str, retry_after: int | None = None) -> str:
    """Create an SSE error event."""
    error = ErrorResponse(code=code, message=message, retry_after=retry_after)
    return f"data: {json.dumps(error.model_dump())}\n\n"


async def generate_chat_stream(request: ChatRequest) -> AsyncGenerator[str, None]:
    """Generate SSE stream with status updates, streamed answer tokens, and final response."""
    messages = request.messages.copy()
    status_queue: asyncio.Queue[str] = asyncio.Queue()
    content_queue: asyncio.Queue[str] = asyncio.Queue()

    async def status_callback(status: str):
        await status_queue.put(status)

    async def content_callback(delta: str):
        await content_queue.put(delta)

    async def run_pipeline():
        if request.deep_research:
            return await run_deep_research_pipeline_with_status(
                oai_client=oai_client,
                tavily_client=tavily_client,
                messages=messages,
                status_callback=status_callback,
                content_callback=content_callback,
            )
        else:
            return await run_search_pipeline_with_status(
                oai_client=oai_client,
                tavily_client=tavily_client,
                messages=messages,
                status_callback=status_callback,
                content_callback=content_callback,
            )

    pipeline_task = asyncio.create_task(run_pipeline())

    while not pipeline_task.done():
        try:
            status = await asyncio.wait_for(status_queue.get(), timeout=0.05)
            yield f"data: {json.dumps({'status': status})}\n\n"
        except asyncio.TimeoutError:
            pass
        try:
            delta = content_queue.get_nowait()
            yield f"data: {json.dumps({'content_delta': delta})}\n\n"
        except asyncio.QueueEmpty:
            pass

    while not status_queue.empty():
        status = await status_queue.get()
        yield f"data: {json.dumps({'status': status})}\n\n"

    while not content_queue.empty():
        delta = await content_queue.get()
        yield f"data: {json.dumps({'content_delta': delta})}\n\n"

    try:
        timings = pipeline_task.result()
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded: {e}")
        retry_after = int(e.response.headers.get("retry-after", 60)) if e.response else 60
        yield create_error_event(
            code="rate_limit",
            message="Rate limit exceeded. Please try again later.",
            retry_after=retry_after,
        )
        return
    except APIConnectionError as e:
        logger.error(f"API connection error: {e}")
        yield create_error_event(
            code="connection_error",
            message="Failed to connect to AI service. Please check your connection and try again.",
        )
        return
    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
        yield create_error_event(
            code="api_error",
            message="AI service error. Please try again.",
        )
        return
    except Exception as e:
        logger.exception(f"Unexpected error in pipeline: {e}")
        yield create_error_event(
            code="internal_error",
            message="An unexpected error occurred. Please try again.",
        )
        return

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

    Standard mode: Single-pass RAG with 3 queries
    Deep research mode: Iterative RAG with up to 3 iterations and 5 queries each
    """
    return StreamingResponse(
        generate_chat_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
