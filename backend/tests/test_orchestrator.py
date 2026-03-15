import pytest

from oai import OAI, Message, OAIConfig
from workflow import SearchQueries, orchestrate


@pytest.fixture
def client() -> OAI:
    return OAI(config=OAIConfig(model="gpt-5-mini"))


@pytest.mark.asyncio
async def test_orchestrate_standard_simple_query(client: OAI):
    """Test standard mode with a simple, focused question."""
    messages = [Message(role="user", content="What is the current price of Bitcoin?")]

    queries, output = await orchestrate(client, messages, deep_research=False)

    assert isinstance(queries, SearchQueries)
    assert len(queries.queries) >= 1
    assert len(queries.queries) <= 3, "Standard mode should generate at most 3 queries"

    print("\n--- Standard Mode: Simple Query ---")
    print(f"User: {messages[0].content}")
    print(f"Queries generated: {len(queries.queries)}")
    for i, q in enumerate(queries.queries, 1):
        print(f"  {i}. {q}")
    print(f"Tokens: {output.usage.total_tokens}")
    print(f"Latency: {output.latency_ms:.2f}ms")


@pytest.mark.asyncio
async def test_orchestrate_standard_complex_query(client: OAI):
    """Test standard mode with a more complex question."""
    messages = [
        Message(
            role="user",
            content="Compare the pros and cons of React vs Vue for building a large-scale enterprise application",
        )
    ]

    queries, output = await orchestrate(client, messages, deep_research=False)

    assert isinstance(queries, SearchQueries)
    assert len(queries.queries) >= 1
    assert len(queries.queries) <= 3, "Standard mode should generate at most 3 queries"

    print("\n--- Standard Mode: Complex Query ---")
    print(f"User: {messages[0].content}")
    print(f"Queries generated: {len(queries.queries)}")
    for i, q in enumerate(queries.queries, 1):
        print(f"  {i}. {q}")
    print(f"Tokens: {output.usage.total_tokens}")
    print(f"Latency: {output.latency_ms:.2f}ms")


@pytest.mark.asyncio
async def test_orchestrate_deep_research_multifaceted(client: OAI):
    """Test deep research mode with a multi-faceted research question."""
    messages = [
        Message(
            role="user",
            content="I'm considering investing in NVIDIA stock. What are the key factors I should consider including their AI chip market position, recent financial performance, competition from AMD and Intel, and any regulatory risks?",
        )
    ]

    queries, output = await orchestrate(client, messages, deep_research=True)

    assert isinstance(queries, SearchQueries)
    assert len(queries.queries) >= 1
    assert len(queries.queries) <= 5, "Deep research mode should generate at most 5 queries"

    print("\n--- Deep Research Mode: Multi-faceted Query ---")
    print(f"User: {messages[0].content}")
    print(f"Queries generated: {len(queries.queries)}")
    for i, q in enumerate(queries.queries, 1):
        print(f"  {i}. {q}")
    print(f"Tokens: {output.usage.total_tokens}")
    print(f"Latency: {output.latency_ms:.2f}ms")


@pytest.mark.asyncio
async def test_orchestrate_deep_research_with_context(client: OAI):
    """Test deep research mode with previous context (simulating iterative search)."""
    messages = [
        Message(
            role="user", content="What are the health benefits and risks of intermittent fasting?"
        )
    ]

    previous_context = """
Previous search found:
- Intermittent fasting can improve insulin sensitivity
- Common methods include 16:8 and 5:2 fasting schedules
- Some studies show weight loss benefits

Missing information:
- Long-term effects on metabolism
- Risks for specific populations (diabetics, pregnant women)
- Effects on muscle mass and athletic performance
"""

    queries, output = await orchestrate(
        client, messages, deep_research=True, previous_context=previous_context
    )

    assert isinstance(queries, SearchQueries)
    assert len(queries.queries) >= 1
    assert len(queries.queries) <= 5

    print("\n--- Deep Research Mode: With Previous Context ---")
    print(f"User: {messages[0].content}")
    print("Previous context provided: Yes")
    print(f"Queries generated: {len(queries.queries)}")
    for i, q in enumerate(queries.queries, 1):
        print(f"  {i}. {q}")
    print(f"Tokens: {output.usage.total_tokens}")
    print(f"Latency: {output.latency_ms:.2f}ms")


@pytest.mark.asyncio
async def test_orchestrate_deep_research_technical(client: OAI):
    """Test deep research mode with a technical/research question."""
    messages = [
        Message(
            role="user",
            content="Explain the current state of quantum error correction, including recent breakthroughs, the main approaches being pursued by Google, IBM, and startups, and what milestones are needed for fault-tolerant quantum computing.",
        )
    ]

    queries, output = await orchestrate(client, messages, deep_research=True)

    assert isinstance(queries, SearchQueries)
    assert len(queries.queries) >= 1
    assert len(queries.queries) <= 5

    print("\n--- Deep Research Mode: Technical Research ---")
    print(f"User: {messages[0].content}")
    print(f"Queries generated: {len(queries.queries)}")
    for i, q in enumerate(queries.queries, 1):
        print(f"  {i}. {q}")
    print(f"Tokens: {output.usage.total_tokens}")
    print(f"Latency: {output.latency_ms:.2f}ms")
