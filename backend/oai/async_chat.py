import time
from typing import TypeVar
from openai import AsyncOpenAI
from pydantic import BaseModel

from .config import OAIConfig
from .models import ChatInput, ChatOutput, TokenUsage

T = TypeVar("T", bound=BaseModel)


async def chat_async(
    client: AsyncOpenAI,
    config: OAIConfig,
    input: ChatInput,
    include_raw: bool = False,
) -> ChatOutput:
    """
    Asynchronous chat completion using the OpenAI Responses API.
    
    Args:
        client: AsyncOpenAI client instance
        config: Configuration for the API call
        input: Structured chat input
        include_raw: Whether to include raw API response in output
    
    Returns:
        Structured chat output with response and metadata
    """
    messages = input.to_api_messages()
    
    start_time = time.perf_counter()
    
    api_kwargs = {
        "model": config.model,
        "input": messages,
    }
    
    if config.temperature is not None:
        api_kwargs["temperature"] = config.temperature
    
    if config.max_tokens is not None:
        api_kwargs["max_output_tokens"] = config.max_tokens
    
    if config.reasoning_effort is not None:
        api_kwargs["reasoning"] = {"effort": config.reasoning_effort}
    
    response = await client.responses.create(**api_kwargs)
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    
    content = ""
    for item in response.output:
        if item.type == "message":
            for content_block in item.content:
                if content_block.type == "output_text":
                    content += content_block.text
    
    usage = TokenUsage(
        input_tokens=response.usage.input_tokens if response.usage else 0,
        output_tokens=response.usage.output_tokens if response.usage else 0,
        total_tokens=response.usage.total_tokens if response.usage else 0,
    )
    
    return ChatOutput(
        content=content,
        model=response.model,
        usage=usage,
        latency_ms=latency_ms,
        response_id=response.id,
        raw_response=response.model_dump() if include_raw else None,
    )


async def chat_async_structured(
    client: AsyncOpenAI,
    config: OAIConfig,
    input: ChatInput,
    response_model: type[T],
    include_raw: bool = False,
) -> tuple[T, ChatOutput]:
    """
    Asynchronous chat with structured output parsing.
    
    Uses OpenAI's structured output feature to parse the response
    directly into a Pydantic model.
    
    Args:
        client: AsyncOpenAI client instance
        config: Configuration for the API call
        input: Structured chat input
        response_model: Pydantic model class to parse response into
        include_raw: Whether to include raw API response in output
    
    Returns:
        Tuple of (parsed_response, chat_output)
    """
    from .structured import create_response_format
    
    messages = input.to_api_messages()
    response_format = create_response_format(response_model)
    
    start_time = time.perf_counter()
    
    api_kwargs = {
        "model": config.model,
        "input": messages,
        "text": response_format,
    }
    
    if config.temperature is not None:
        api_kwargs["temperature"] = config.temperature
    
    if config.max_tokens is not None:
        api_kwargs["max_output_tokens"] = config.max_tokens
    
    if config.reasoning_effort is not None:
        api_kwargs["reasoning"] = {"effort": config.reasoning_effort}
    
    response = await client.responses.create(**api_kwargs)
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    
    content = ""
    for item in response.output:
        if item.type == "message":
            for content_block in item.content:
                if content_block.type == "output_text":
                    content += content_block.text
    
    usage = TokenUsage(
        input_tokens=response.usage.input_tokens if response.usage else 0,
        output_tokens=response.usage.output_tokens if response.usage else 0,
        total_tokens=response.usage.total_tokens if response.usage else 0,
    )
    
    chat_output = ChatOutput(
        content=content,
        model=response.model,
        usage=usage,
        latency_ms=latency_ms,
        response_id=response.id,
        raw_response=response.model_dump() if include_raw else None,
    )
    
    parsed = response_model.model_validate_json(content)
    
    return parsed, chat_output
