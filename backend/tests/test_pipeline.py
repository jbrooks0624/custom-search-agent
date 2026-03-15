import pytest

from oai import OAI, Message, OAIConfig
from tvly import Tavily, TavilyConfig
from workflow import run_deep_research_pipeline, run_search_pipeline


@pytest.fixture
def oai_client() -> OAI:
    return OAI(config=OAIConfig(model="gpt-5-mini"))


@pytest.fixture
def tavily_client() -> Tavily:
    return Tavily(config=TavilyConfig(max_results=2))


@pytest.mark.asyncio
async def test_run_search_pipeline_simple(oai_client: OAI, tavily_client: Tavily):
    """Test the full search pipeline with a simple question."""
    messages = [Message(role="user", content="What is retrieval augmented generation (RAG)?")]

    initial_len = len(messages)

    timings = await run_search_pipeline(oai_client, tavily_client, messages, return_timings=True)

    assert len(messages) == initial_len + 1, "Should append one assistant message"
    assert messages[-1].role == "assistant"
    assert len(messages[-1].content) > 100, "Should have a substantive answer"

    print("\n--- Search Pipeline: Simple Question ---")
    print(f"User: {messages[0].content}")
    print("\n--- Timings ---")
    print(timings)
    print(f"\nAssistant ({len(messages[-1].content)} chars):")
    print(messages[-1].content[:500])
    if len(messages[-1].content) > 500:
        print(f"... [{len(messages[-1].content) - 500} more chars]")


@pytest.mark.asyncio
async def test_run_search_pipeline_with_extraction(oai_client: OAI, tavily_client: Tavily):
    """Test pipeline WITH extraction (default)."""
    messages = [Message(role="user", content="What is RAG in AI?")]

    timings = await run_search_pipeline(
        oai_client, tavily_client, messages, return_timings=True, skip_extraction=False
    )

    assert len(messages) == 2

    print("\n--- Pipeline WITH Extraction ---")
    print(f"User: {messages[0].content}")
    print(f"\n{timings}")
    print("\nContext chars passed to summarizer: extracted bullet points")
    print(f"Answer length: {len(messages[-1].content)} chars")


@pytest.mark.asyncio
async def test_run_search_pipeline_without_extraction(oai_client: OAI, tavily_client: Tavily):
    """Test pipeline WITHOUT extraction (skip_extraction=True)."""
    messages = [Message(role="user", content="What is RAG in AI?")]

    timings = await run_search_pipeline(
        oai_client, tavily_client, messages, return_timings=True, skip_extraction=True
    )

    assert len(messages) == 2

    print("\n--- Pipeline WITHOUT Extraction ---")
    print(f"User: {messages[0].content}")
    print(f"\n{timings}")
    print("\nContext chars passed to summarizer: scrubbed markdown directly")
    print(f"Answer length: {len(messages[-1].content)} chars")


@pytest.mark.asyncio
async def test_run_search_pipeline_conversation(oai_client: OAI, tavily_client: Tavily):
    """Test that messages list is properly mutated."""
    messages = [Message(role="user", content="What is Python used for?")]

    # Keep reference to same list
    original_messages = messages

    await run_search_pipeline(oai_client, tavily_client, messages)

    # Should be the same list object, mutated
    assert messages is original_messages
    assert len(messages) == 2

    print("\n--- Pipeline Mutation Test ---")
    print(f"Messages list properly mutated: {len(messages)} messages")
    print(f"Last message role: {messages[-1].role}")


@pytest.mark.asyncio
async def test_run_deep_research_pipeline(oai_client: OAI, tavily_client: Tavily):
    """Test the deep research pipeline with a complex question that may require multiple iterations."""
    messages = [
        Message(
            role="user",
            content="Compare the economic policies and their outcomes of the Biden and Trump administrations. Include specific data on GDP growth, unemployment, and inflation.",
        )
    ]

    timings = await run_deep_research_pipeline(
        oai_client, tavily_client, messages, max_iterations=3, return_timings=True
    )

    assert len(messages) == 2
    assert messages[-1].role == "assistant"

    print("\n--- Deep Research Pipeline ---")
    print(f"User: {messages[0].content}")
    print("\n--- Timings ---")
    print(timings)
    print(f"\nIterations: {timings.num_iterations}")
    print(f"Total queries: {timings.num_queries}")
    print(f"Total sources: {timings.num_sources}")
    print(f"\nAssistant ({len(messages[-1].content)} chars):")
    print(messages[-1].content[:1000])
    if len(messages[-1].content) > 1000:
        print(f"... [{len(messages[-1].content) - 1000} more chars]")


@pytest.mark.asyncio
async def test_deep_research_single_iteration(oai_client: OAI, tavily_client: Tavily):
    """Test deep research with a simpler question that should complete in one iteration."""
    messages = [Message(role="user", content="What is the current population of Tokyo?")]

    timings = await run_deep_research_pipeline(
        oai_client, tavily_client, messages, return_timings=True
    )

    assert len(messages) == 2

    print("\n--- Deep Research (Simple Question) ---")
    print(f"User: {messages[0].content}")
    print(f"\n{timings}")
    print(f"\nIterations: {timings.num_iterations}")
    print(f"Answer: {messages[-1].content[:500]}")


@pytest.mark.asyncio
async def test_deep_research_multi_iteration(oai_client: OAI, tavily_client: Tavily):
    """Test deep research with a highly specific multi-part question likely to need multiple iterations."""
    messages = [
        Message(
            role="user",
            content="""I need a detailed technical comparison for my startup:
1. What are the exact pricing tiers for OpenAI's GPT-5 API vs Anthropic's Claude 4 API (per million tokens)?
2. What are the specific rate limits for each tier?
3. Which companies have publicly announced switching from one to the other in 2026, and why?
4. What are the latency benchmarks from independent testing?""",
        )
    ]

    timings = await run_deep_research_pipeline(
        oai_client, tavily_client, messages, max_iterations=3, return_timings=True
    )

    assert len(messages) == 2

    print("\n--- Deep Research (Multi-Part Technical Question) ---")
    print(f"User: {messages[0].content[:200]}...")
    print("\n--- Timings ---")
    print(timings)
    print(f"\nIterations: {timings.num_iterations}")
    print(f"Total queries across iterations: {timings.num_queries}")
    print(f"\nAnswer ({len(messages[-1].content)} chars):")
    print(messages[-1].content[:1500])
    if len(messages[-1].content) > 1500:
        print(f"... [{len(messages[-1].content) - 1500} more chars]")
