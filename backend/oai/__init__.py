"""
OAI - A structured wrapper around the OpenAI Responses API.

Provides clean, type-safe interfaces for async chat completions with
Pydantic-based input/output handling.

Example:
    from oai import OAI, OAIConfig, ChatInput, Message

    client = OAI(config=OAIConfig(model="gpt-5-mini"))

    output = await client.chat_async(
        ChatInput(
            system_prompt="You are a helpful assistant.",
            messages=[Message(role="user", content="Hello!")]
        )
    )

    print(output.content)
    print(f"Tokens used: {output.usage.total_tokens}")
"""

from .client import OAI
from .config import ModelName, OAIConfig, ReasoningEffort
from .models import ChatInput, ChatOutput, Message, TokenUsage
from .structured import create_response_format, validate_response

__all__ = [
    "OAI",
    "OAIConfig",
    "ModelName",
    "ReasoningEffort",
    "ChatInput",
    "ChatOutput",
    "Message",
    "TokenUsage",
    "create_response_format",
    "validate_response",
]
