import pytest

from tvly import SearchInput, Tavily, TavilyConfig


@pytest.fixture
def client() -> Tavily:
    return Tavily(
        config=TavilyConfig(
            max_results=3,
            include_raw_content="markdown",
        )
    )


@pytest.mark.asyncio
async def test_search_async(client: Tavily):
    """Test basic async search."""
    input = SearchInput(query="What is retrieval augmented generation RAG?")

    output = await client.search_async(input)

    assert output.has_results, "Should have search results"
    assert len(output.results) <= 3, "Should respect max_results"
    assert output.query == input.query, "Should echo back query"
    assert output.latency_ms > 0, "Should measure latency"

    print("\n--- search_async response ---")
    print(f"Query: {output.query}")
    print(f"Results: {len(output.results)}")
    print(f"Response time: {output.response_time:.2f}s")
    print(f"Latency: {output.latency_ms:.2f}ms")

    for i, result in enumerate(output.results, 1):
        print(f"\n{i}. {result.title}")
        print(f"   URL: {result.url}")
        print(f"   Score: {result.score:.3f}")
        print(f"   Content preview: {result.content[:100]}...")


@pytest.mark.asyncio
async def test_search_with_answer(client: Tavily):
    """Test search with LLM-generated answer."""
    config = TavilyConfig(
        max_results=3,
        include_answer=True,
    )
    input = SearchInput(query="What are the benefits of RAG over fine-tuning?")

    output = await client.search_async(input, config=config)

    assert output.has_results, "Should have search results"
    assert output.answer is not None, "Should include an answer"

    print("\n--- search with answer ---")
    print(f"Query: {output.query}")
    print(f"\nAnswer: {output.answer}")
    print(f"\nSources: {len(output.results)}")


@pytest.mark.asyncio
async def test_get_markdown_content(client: Tavily):
    """Test getting combined markdown content from results."""
    input = SearchInput(query="Python asyncio tutorial")

    output = await client.search_async(input)

    markdown = output.get_markdown_content()

    if markdown:
        print("\n--- markdown content preview ---")
        print(markdown[:500] + "..." if len(markdown) > 500 else markdown)
    else:
        print("\n--- no raw content available ---")

    assert output.has_results, "Should have results"
