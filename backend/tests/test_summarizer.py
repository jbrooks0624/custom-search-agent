import pytest

from oai import OAI, Message, OAIConfig
from workflow import SummarizerResponse, summarize


@pytest.fixture
def client() -> OAI:
    return OAI(config=OAIConfig(model="gpt-5-mini"))


SAMPLE_SEARCH_CONTEXT = """
## What is Retrieval-Augmented Generation (RAG)?

Source: https://example.com/rag-explained

Retrieval-Augmented Generation (RAG) is an AI framework that combines the power of large language models with external knowledge retrieval. Instead of relying solely on the model's training data, RAG systems fetch relevant information from a knowledge base at query time.

Key components:
1. **Retriever**: Searches a document store for relevant passages
2. **Generator**: Uses the retrieved context to generate responses

Benefits include reduced hallucinations, up-to-date information, and domain-specific knowledge without fine-tuning.

---

## RAG vs Fine-tuning

Source: https://example.com/rag-vs-finetuning

RAG is preferred when:
- Knowledge needs to be updated frequently
- You need traceable sources
- Domain knowledge is extensive

Fine-tuning is preferred when:
- You need to change the model's behavior or style
- The knowledge is static and well-defined
"""


@pytest.mark.asyncio
async def test_summarize_standard_mode(client: OAI):
    """Test summarizer in standard mode - should always provide final answer."""
    messages = [Message(role="user", content="What is RAG and when should I use it?")]

    response, output = await summarize(
        client=client,
        messages=messages,
        search_context=SAMPLE_SEARCH_CONTEXT,
        deep_research=False,
        num_iterations=1,
    )

    assert isinstance(response, SummarizerResponse)
    assert response.needs_more_research is False, "Standard mode should not request more research"
    assert len(response.content) > 100, "Should provide a substantive answer"

    print("\n--- Standard Mode: Summarizer Response ---")
    print(f"User: {messages[0].content}")
    print(f"Needs more research: {response.needs_more_research}")
    print(f"\nAnswer:\n{response.content}")
    print(f"\nTokens: {output.usage.total_tokens}")
    print(f"Duration: {output.latency_ms:.2f}ms")


@pytest.mark.asyncio
async def test_summarize_deep_research_sufficient_context(client: OAI):
    """Test summarizer in deep research mode with sufficient context."""
    messages = [Message(role="user", content="What is RAG?")]

    response, output = await summarize(
        client=client,
        messages=messages,
        search_context=SAMPLE_SEARCH_CONTEXT,
        deep_research=True,
        num_iterations=1,
        max_iterations=3,
    )

    assert isinstance(response, SummarizerResponse)
    assert len(response.content) > 50, "Should provide content"

    print("\n--- Deep Research Mode: Sufficient Context ---")
    print(f"User: {messages[0].content}")
    print(f"Needs more research: {response.needs_more_research}")
    print(f"\nContent:\n{response.content}")
    print(f"\nTokens: {output.usage.total_tokens}")
    print(f"Duration: {output.latency_ms:.2f}ms")


INCOMPLETE_SEARCH_CONTEXT = """
## Bitcoin Price Today

Source: https://example.com/bitcoin

Bitcoin is a decentralized cryptocurrency. It was created in 2009 by Satoshi Nakamoto.

Note: Price data unavailable due to API error.
"""


@pytest.mark.asyncio
async def test_summarize_deep_research_needs_more(client: OAI):
    """Test summarizer in deep research mode with insufficient context - should request more research."""
    messages = [
        Message(
            role="user",
            content="What is the current Bitcoin price and how has it performed over the last month? Also explain the key factors affecting its price.",
        )
    ]

    response, output = await summarize(
        client=client,
        messages=messages,
        search_context=INCOMPLETE_SEARCH_CONTEXT,
        deep_research=True,
        num_iterations=1,
        max_iterations=3,
    )

    assert isinstance(response, SummarizerResponse)
    assert len(response.content) > 50, "Should provide content or feedback"

    print("\n--- Deep Research Mode: Insufficient Context ---")
    print(f"User: {messages[0].content}")
    print(f"Needs more research: {response.needs_more_research}")
    print(f"\nContent:\n{response.content}")
    print(f"\nTokens: {output.usage.total_tokens}")
    print(f"Duration: {output.latency_ms:.2f}ms")


@pytest.mark.asyncio
async def test_summarize_deep_research_max_iterations_reached(client: OAI):
    """Test that summarizer provides final answer when max iterations reached."""
    messages = [Message(role="user", content="Explain quantum computing in detail")]

    response, output = await summarize(
        client=client,
        messages=messages,
        search_context=INCOMPLETE_SEARCH_CONTEXT,  # Even with incomplete context
        deep_research=True,
        num_iterations=3,  # Already at max
        max_iterations=3,
    )

    assert isinstance(response, SummarizerResponse)
    assert response.needs_more_research is False, (
        "Should not request more research at max iterations"
    )
    assert len(response.content) > 50, "Should provide best possible answer"

    print("\n--- Deep Research Mode: Max Iterations Reached ---")
    print(f"User: {messages[0].content}")
    print("Iteration: 3/3 (max reached)")
    print(f"Needs more research: {response.needs_more_research}")
    print(f"\nContent:\n{response.content}")
    print(f"\nTokens: {output.usage.total_tokens}")
    print(f"Duration: {output.latency_ms:.2f}ms")
