from typing import Literal

from pydantic import BaseModel, Field

SearchDepth = Literal["basic", "advanced"]
Topic = Literal["general", "news", "finance"]
TimeRange = Literal["day", "week", "month", "year", "d", "w", "m", "y"]


class TavilyConfig(BaseModel):
    """Configuration for Tavily API calls."""

    search_depth: SearchDepth = Field(
        default="basic",
        description="Search depth: 'basic' for generic snippets, 'advanced' for more relevant content",
    )
    topic: Topic = Field(
        default="general", description="Search category: 'general', 'news', or 'finance'"
    )
    max_results: int = Field(
        default=5, ge=1, le=20, description="Maximum number of search results (1-20)"
    )
    chunks_per_source: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of content chunks per source (only for advanced search)",
    )
    include_answer: bool | Literal["basic", "advanced"] = Field(
        default=False,
        description="Include LLM-generated answer: False, True/'basic', or 'advanced'",
    )
    include_raw_content: bool | Literal["markdown", "text"] = Field(
        default="markdown",
        description="Include raw page content: False, True/'markdown', or 'text'",
    )
    include_images: bool = Field(
        default=False, description="Include query-related images in response"
    )
    time_range: TimeRange | None = Field(
        default=None, description="Filter results by time range: 'day', 'week', 'month', 'year'"
    )
    include_domains: list[str] = Field(
        default_factory=list, description="Domains to specifically include (max 300)"
    )
    exclude_domains: list[str] = Field(
        default_factory=list, description="Domains to specifically exclude (max 150)"
    )
    timeout: float = Field(default=60.0, gt=0, description="Request timeout in seconds")

    model_config = {
        "frozen": True,
    }

    def with_overrides(self, **kwargs) -> "TavilyConfig":
        """Create a new config with specified overrides."""
        return self.model_copy(update=kwargs)
