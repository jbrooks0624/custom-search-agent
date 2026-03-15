import pytest

from oai import OAI, OAIConfig, Message
from tvly import Tavily, TavilyConfig
from workflow import run_search_pipeline


@pytest.fixture
def oai_client() -> OAI:
    return OAI(config=OAIConfig(model="gpt-5-mini"))


@pytest.fixture
def tavily_client() -> Tavily:
    return Tavily(config=TavilyConfig(max_results=2))


@pytest.mark.asyncio
async def test_run_search_pipeline_simple(oai_client: OAI, tavily_client: Tavily):
    """Test the full search pipeline with a simple question."""
    messages = [
        Message(role="user", content="What is retrieval augmented generation (RAG)?")
    ]
    
    initial_len = len(messages)
    
    timings = await run_search_pipeline(oai_client, tavily_client, messages, return_timings=True)
    
    assert len(messages) == initial_len + 1, "Should append one assistant message"
    assert messages[-1].role == "assistant"
    assert len(messages[-1].content) > 100, "Should have a substantive answer"
    
    print(f"\n--- Search Pipeline: Simple Question ---")
    print(f"User: {messages[0].content}")
    print(f"\n--- Timings ---")
    print(timings)
    print(f"\nAssistant ({len(messages[-1].content)} chars):")
    print(messages[-1].content[:500])
    if len(messages[-1].content) > 500:
        print(f"... [{len(messages[-1].content) - 500} more chars]")


@pytest.mark.asyncio
async def test_run_search_pipeline_with_extraction(oai_client: OAI, tavily_client: Tavily):
    """Test pipeline WITH extraction (default)."""
    messages = [
        Message(role="user", content="What is RAG in AI?")
    ]
    
    timings = await run_search_pipeline(oai_client, tavily_client, messages, return_timings=True, skip_extraction=False)
    
    assert len(messages) == 2
    
    print(f"\n--- Pipeline WITH Extraction ---")
    print(f"User: {messages[0].content}")
    print(f"\n{timings}")
    print(f"\nContext chars passed to summarizer: extracted bullet points")
    print(f"Answer length: {len(messages[-1].content)} chars")


@pytest.mark.asyncio
async def test_run_search_pipeline_without_extraction(oai_client: OAI, tavily_client: Tavily):
    """Test pipeline WITHOUT extraction (skip_extraction=True)."""
    messages = [
        Message(role="user", content="What is RAG in AI?")
    ]
    
    timings = await run_search_pipeline(oai_client, tavily_client, messages, return_timings=True, skip_extraction=True)
    
    assert len(messages) == 2
    
    print(f"\n--- Pipeline WITHOUT Extraction ---")
    print(f"User: {messages[0].content}")
    print(f"\n{timings}")
    print(f"\nContext chars passed to summarizer: scrubbed markdown directly")
    print(f"Answer length: {len(messages[-1].content)} chars")


@pytest.mark.asyncio
async def test_run_search_pipeline_conversation(oai_client: OAI, tavily_client: Tavily):
    """Test that messages list is properly mutated."""
    messages = [
        Message(role="user", content="What is Python used for?")
    ]
    
    # Keep reference to same list
    original_messages = messages
    
    await run_search_pipeline(oai_client, tavily_client, messages)
    
    # Should be the same list object, mutated
    assert messages is original_messages
    assert len(messages) == 2
    
    print(f"\n--- Pipeline Mutation Test ---")
    print(f"Messages list properly mutated: {len(messages)} messages")
    print(f"Last message role: {messages[-1].role}")
