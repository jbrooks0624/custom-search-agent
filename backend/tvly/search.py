import time

from tavily import AsyncTavilyClient, TavilyClient

from .config import TavilyConfig
from .models import ImageResult, SearchInput, SearchOutput, SearchResult


def _build_search_kwargs(config: TavilyConfig) -> dict:
    """Build kwargs for Tavily search from config."""
    kwargs = {
        "search_depth": config.search_depth,
        "topic": config.topic,
        "max_results": config.max_results,
        "include_answer": config.include_answer,
        "include_raw_content": config.include_raw_content,
        "include_images": config.include_images,
        "timeout": config.timeout,
    }

    if config.search_depth == "advanced":
        kwargs["chunks_per_source"] = config.chunks_per_source

    if config.time_range:
        kwargs["time_range"] = config.time_range

    if config.include_domains:
        kwargs["include_domains"] = config.include_domains

    if config.exclude_domains:
        kwargs["exclude_domains"] = config.exclude_domains

    return kwargs


def _parse_response(response: dict, latency_ms: float, include_raw: bool) -> SearchOutput:
    """Parse Tavily API response into structured output."""
    results = []
    for r in response.get("results", []):
        results.append(
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                raw_content=r.get("raw_content"),
                published_date=r.get("published_date"),
            )
        )

    images = []
    for img in response.get("images", []):
        if isinstance(img, str):
            images.append(ImageResult(url=img))
        elif isinstance(img, dict):
            images.append(
                ImageResult(
                    url=img.get("url", ""),
                    description=img.get("description"),
                )
            )

    return SearchOutput(
        query=response.get("query", ""),
        results=results,
        answer=response.get("answer"),
        images=images,
        response_time=response.get("response_time", 0.0),
        latency_ms=latency_ms,
        raw_response=response if include_raw else None,
    )


def search(
    client: TavilyClient,
    config: TavilyConfig,
    input: SearchInput,
    include_raw: bool = False,
) -> SearchOutput:
    """
    Synchronous search using the Tavily API.

    Args:
        client: TavilyClient instance
        config: Configuration for the search
        input: Structured search input
        include_raw: Whether to include raw API response in output

    Returns:
        Structured search output with results and metadata
    """
    kwargs = _build_search_kwargs(config)

    start_time = time.perf_counter()
    response = client.search(query=input.query, **kwargs)
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000

    return _parse_response(response, latency_ms, include_raw)


async def search_async(
    client: AsyncTavilyClient,
    config: TavilyConfig,
    input: SearchInput,
    include_raw: bool = False,
) -> SearchOutput:
    """
    Asynchronous search using the Tavily API.

    Args:
        client: AsyncTavilyClient instance
        config: Configuration for the search
        input: Structured search input
        include_raw: Whether to include raw API response in output

    Returns:
        Structured search output with results and metadata
    """
    kwargs = _build_search_kwargs(config)

    start_time = time.perf_counter()
    response = await client.search(query=input.query, **kwargs)
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000

    return _parse_response(response, latency_ms, include_raw)
