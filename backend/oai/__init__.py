"""
OAI - A structured wrapper around the OpenAI Responses API.

Provides clean, type-safe interfaces for chat completions with
Pydantic-based input/output handling.

Example:
    from oai import OAI, OAIConfig, ChatInput, Message
    
    client = OAI(config=OAIConfig(model="gpt-5-mini"))
    
    output = client.chat(
        ChatInput(
            system_prompt="You are a helpful assistant.",
            messages=[Message(role="user", content="Hello!")]
        )
    )
    
    print(output.content)
    print(f"Tokens used: {output.usage.total_tokens}")
    
    # With reasoning effort for complex tasks
    output = client.chat(
        ChatInput(messages=[Message(role="user", content="Solve this...")]),
        config=OAIConfig(model="gpt-5.4", reasoning_effort="high")
    )
"""

from .client import OAI
from .config import OAIConfig, ModelName, ReasoningEffort
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
