"""
TVLY - A structured wrapper around the Tavily Search API.

Provides clean, type-safe interfaces for web search with
Pydantic-based input/output handling.

Example:
    from tvly import Tavily, TavilyConfig, SearchInput
    
    client = Tavily(config=TavilyConfig(max_results=5))
    
    output = client.search(SearchInput(query="What is RAG?"))
    
    print(f"Found {len(output.results)} results")
    for result in output.results:
        print(f"- {result.title}: {result.url}")
    
    # Get all raw content as markdown
    markdown = output.get_markdown_content()
"""

from .client import Tavily
from .config import TavilyConfig, SearchDepth, Topic, TimeRange
from .models import SearchInput, SearchOutput, SearchResult, ImageResult

__all__ = [
    "Tavily",
    "TavilyConfig",
    "SearchDepth",
    "Topic",
    "TimeRange",
    "SearchInput",
    "SearchOutput",
    "SearchResult",
    "ImageResult",
]
