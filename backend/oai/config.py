from typing import Literal
from pydantic import BaseModel, Field


ModelName = Literal[
    # GPT-5.4 series (latest flagship)
    "gpt-5.4",
    "gpt-5.4-pro",
    # GPT-5 series
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    # GPT-5.3 Codex (specialized for coding)
    "gpt-5.3-codex",
    # GPT-5.2 (previous frontier)
    "gpt-5.2",
]

ReasoningEffort = Literal[
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
]


class OAIConfig(BaseModel):
    """Configuration for OpenAI API calls."""
    
    model: ModelName = Field(
        default="gpt-5-mini",
        description="The model to use for the API call"
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature between 0 and 2 (not supported by GPT-5 models)"
    )
    max_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Maximum tokens in the response"
    )
    reasoning_effort: ReasoningEffort | None = Field(
        default=None,
        description="Reasoning effort level for GPT-5 models (none, minimal, low, medium, high, xhigh)"
    )
    timeout: float = Field(
        default=60.0,
        gt=0,
        description="Request timeout in seconds"
    )
    
    model_config = {
        "frozen": True,
    }
    
    def with_overrides(self, **kwargs) -> "OAIConfig":
        """Create a new config with specified overrides."""
        return self.model_copy(update=kwargs)
