from tvly import SearchInput, SearchOutput, Tavily, TavilyConfig


async def search(
    client: Tavily,
    query: str,
    max_results: int = 2,
    deep_research: bool = False,
) -> str:
    """
    Execute a single search query and return the markdown content.

    Args:
        client: Tavily client instance
        query: The search query to execute
        max_results: Maximum number of results to fetch
        deep_research: If True, use advanced search depth for richer content

    Returns:
        Combined markdown content from all search results
    """
    config = TavilyConfig(
        max_results=max_results,
        include_raw_content="markdown",
        search_depth="advanced" if deep_research else "basic",
    )

    input = SearchInput(query=query)

    output = await client.search_async(input, config=config)

    return output.get_markdown_content()


async def search_with_output(
    client: Tavily,
    query: str,
    max_results: int = 2,
    deep_research: bool = False,
) -> tuple[str, SearchOutput]:
    """
    Execute a single search query and return both markdown and full output.

    Args:
        client: Tavily client instance
        query: The search query to execute
        max_results: Maximum number of results to fetch
        deep_research: If True, use advanced search depth for richer content

    Returns:
        Tuple of (markdown_content, SearchOutput)
    """
    config = TavilyConfig(
        max_results=max_results,
        include_raw_content="markdown",
        search_depth="advanced" if deep_research else "basic",
    )

    input = SearchInput(query=query)

    output = await client.search_async(input, config=config)

    return output.get_markdown_content(), output
