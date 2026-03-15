from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    """Structured input for Tavily search requests."""

    query: str = Field(description="The search query")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Custom metadata for tracking/logging"
    )


class SearchResult(BaseModel):
    """A single search result from Tavily."""

    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the search result")
    content: str = Field(description="Relevant content snippet from the page")
    score: float = Field(description="Relevance score of the result")
    raw_content: str | None = Field(
        default=None, description="Full parsed content (if include_raw_content was enabled)"
    )
    published_date: str | None = Field(
        default=None, description="Publication date (only for news topic)"
    )


class ImageResult(BaseModel):
    """An image result from Tavily."""

    url: str = Field(description="URL of the image")
    description: str | None = Field(
        default=None,
        description="LLM-generated description (if include_image_descriptions was enabled)",
    )


class SearchOutput(BaseModel):
    """Structured output from Tavily search requests."""

    query: str = Field(description="The original search query")
    results: list[SearchResult] = Field(
        default_factory=list, description="List of search results ranked by relevancy"
    )
    answer: str | None = Field(
        default=None, description="LLM-generated answer (if include_answer was enabled)"
    )
    images: list[ImageResult] = Field(
        default_factory=list, description="Related images (if include_images was enabled)"
    )
    response_time: float = Field(default=0.0, description="API response time in seconds")
    latency_ms: float = Field(default=0.0, description="Total request latency in milliseconds")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Timestamp when the response was created"
    )
    raw_response: dict[str, Any] | None = Field(
        default=None, description="Raw API response (optional, for debugging)"
    )

    model_config = {
        "frozen": True,
    }

    @property
    def has_results(self) -> bool:
        """Check if there are any search results."""
        return len(self.results) > 0

    @property
    def top_result(self) -> SearchResult | None:
        """Get the top-ranked search result."""
        return self.results[0] if self.results else None

    def get_markdown_content(self) -> str:
        """Get all raw content concatenated as markdown."""
        contents = []
        for result in self.results:
            if result.raw_content:
                contents.append(
                    f"## {result.title}\n\nSource: {result.url}\n\n{result.raw_content}"
                )
        return "\n\n---\n\n".join(contents)
