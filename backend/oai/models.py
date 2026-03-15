from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in the conversation."""

    role: str = Field(description="The role of the message author (system, user, assistant)")
    content: str = Field(description="The content of the message")


class ChatInput(BaseModel):
    """Structured input for chat requests."""

    messages: list[Message] = Field(description="The conversation messages")
    system_prompt: str | None = Field(default=None, description="Optional system prompt to prepend")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Custom metadata for tracking/logging"
    )

    def to_api_messages(self) -> list[dict[str, str]]:
        """Convert to OpenAI API message format."""
        api_messages = []

        if self.system_prompt:
            api_messages.append({"role": "system", "content": self.system_prompt})

        for msg in self.messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        return api_messages


class TokenUsage(BaseModel):
    """Token usage statistics from the API response."""

    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)


class ChatOutput(BaseModel):
    """Structured output from chat requests."""

    content: str = Field(description="The response content from the model")
    model: str = Field(description="The model that generated the response")
    usage: TokenUsage = Field(default_factory=TokenUsage, description="Token usage statistics")
    latency_ms: float = Field(default=0.0, description="Request latency in milliseconds")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Timestamp when the response was created"
    )
    response_id: str | None = Field(default=None, description="OpenAI response ID for tracking")
    raw_response: dict[str, Any] | None = Field(
        default=None, description="Raw API response (optional, for debugging)"
    )

    model_config = {
        "frozen": True,
    }
