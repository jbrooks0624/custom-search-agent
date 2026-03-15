import pytest
from pydantic import BaseModel, Field

from oai import OAI, ChatInput, Message, OAIConfig


@pytest.fixture
def client() -> OAI:
    return OAI(config=OAIConfig(model="gpt-5-mini"))


@pytest.mark.asyncio
async def test_chat_async(client: OAI):
    """Test basic async chat completion."""
    input = ChatInput(messages=[Message(role="user", content="Hello, how are you?")])

    output = await client.chat_async(input)

    assert output.content, "Response content should not be empty"
    assert output.model, "Model name should be present"
    assert output.usage.total_tokens > 0, "Should have used some tokens"
    assert output.latency_ms > 0, "Should have measured latency"

    print("\n--- chat_async response ---")
    print(f"Content: {output.content}")
    print(f"Model: {output.model}")
    print(f"Tokens: {output.usage.total_tokens}")
    print(f"Latency: {output.latency_ms:.2f}ms")


class Activities(BaseModel):
    """Structured output for activity suggestions."""

    activities: list[str] = Field(description="List of activities to do today")


@pytest.mark.asyncio
async def test_chat_async_structured(client: OAI):
    """Test async chat with structured output parsing."""
    input = ChatInput(
        system_prompt="You suggest daily activities. Be concise.",
        messages=[Message(role="user", content="Tell me 3 activities to do today")],
    )

    result, output = await client.chat_async_structured(input=input, response_model=Activities)

    assert isinstance(result, Activities), "Should parse into Activities model"
    assert len(result.activities) == 3, f"Should have 3 activities, got {len(result.activities)}"
    assert all(isinstance(a, str) for a in result.activities), "All activities should be strings"
    assert output.usage.total_tokens > 0, "Should have used some tokens"

    print("\n--- chat_async_structured response ---")
    print("Activities:")
    for i, activity in enumerate(result.activities, 1):
        print(f"  {i}. {activity}")
    print(f"Model: {output.model}")
    print(f"Tokens: {output.usage.total_tokens}")
    print(f"Latency: {output.latency_ms:.2f}ms")
