from typing import TypeVar
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel

from .config import OAIConfig
from .models import ChatInput, ChatOutput
from .sync_chat import chat as sync_chat
from .async_chat import chat_async as async_chat, chat_async_structured

T = TypeVar("T", bound=BaseModel)


class OAI:
    """
    Main client wrapper for OpenAI API interactions.
    
    Provides a clean interface for synchronous and asynchronous chat
    completions with structured input/output handling.
    
    Example:
        client = OAI(config=OAIConfig(model="gpt-5-mini"))
        
        # Sync chat
        output = client.chat(ChatInput(messages=[...]))
        
        # Async chat
        output = await client.chat_async(ChatInput(messages=[...]))
        
        # Async with structured output
        result, output = await client.chat_async_structured(
            ChatInput(messages=[...]),
            response_model=MySchema
        )
    """
    
    def __init__(
        self,
        config: OAIConfig | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize the OAI client.
        
        Args:
            config: Default configuration for API calls
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self._config = config or OAIConfig()
        self._sync_client = OpenAI(api_key=api_key)
        self._async_client = AsyncOpenAI(api_key=api_key)
    
    @property
    def config(self) -> OAIConfig:
        """Current default configuration."""
        return self._config
    
    def chat(
        self,
        input: ChatInput,
        config: OAIConfig | None = None,
        include_raw: bool = False,
    ) -> ChatOutput:
        """
        Synchronous chat completion.
        
        Args:
            input: Structured chat input
            config: Optional config override for this call
            include_raw: Whether to include raw API response
        
        Returns:
            Structured chat output
        """
        effective_config = config or self._config
        return sync_chat(
            client=self._sync_client,
            config=effective_config,
            input=input,
            include_raw=include_raw,
        )
    
    async def chat_async(
        self,
        input: ChatInput,
        config: OAIConfig | None = None,
        include_raw: bool = False,
    ) -> ChatOutput:
        """
        Asynchronous chat completion.
        
        Args:
            input: Structured chat input
            config: Optional config override for this call
            include_raw: Whether to include raw API response
        
        Returns:
            Structured chat output
        """
        effective_config = config or self._config
        return await async_chat(
            client=self._async_client,
            config=effective_config,
            input=input,
            include_raw=include_raw,
        )
    
    async def chat_async_structured(
        self,
        input: ChatInput,
        response_model: type[T],
        config: OAIConfig | None = None,
        include_raw: bool = False,
    ) -> tuple[T, ChatOutput]:
        """
        Asynchronous chat with structured output parsing.
        
        Uses OpenAI's structured output feature to parse the response
        directly into a Pydantic model.
        
        Args:
            input: Structured chat input
            response_model: Pydantic model class for response parsing
            config: Optional config override for this call
            include_raw: Whether to include raw API response
        
        Returns:
            Tuple of (parsed_response, chat_output)
        """
        effective_config = config or self._config
        return await chat_async_structured(
            client=self._async_client,
            config=effective_config,
            input=input,
            response_model=response_model,
            include_raw=include_raw,
        )
