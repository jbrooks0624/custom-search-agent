from tavily import AsyncTavilyClient

from .config import TavilyConfig
from .models import SearchInput, SearchOutput
from .search import search_async


class Tavily:
    """
    Client wrapper for Tavily API interactions.

    Provides a clean interface for asynchronous web search
    with structured input/output handling.

    Example:
        client = Tavily(config=TavilyConfig(max_results=5))

        output = await client.search_async(SearchInput(query="Latest AI news"))

        for result in output.results:
            print(result.title, result.url)

        # Get combined markdown content
        print(output.get_markdown_content())
    """

    def __init__(
        self,
        config: TavilyConfig | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize the Tavily client.

        Args:
            config: Default configuration for API calls
            api_key: Tavily API key (defaults to TAVILY_API_KEY env var)
        """
        self._config = config or TavilyConfig()
        self._async_client = AsyncTavilyClient(api_key=api_key) if api_key else AsyncTavilyClient()

    @property
    def config(self) -> TavilyConfig:
        """Current default configuration."""
        return self._config

    async def search_async(
        self,
        input: SearchInput,
        config: TavilyConfig | None = None,
        include_raw: bool = False,
    ) -> SearchOutput:
        """
        Asynchronous web search.

        Args:
            input: Structured search input with query
            config: Optional config override for this call
            include_raw: Whether to include raw API response

        Returns:
            Structured search output with results
        """
        effective_config = config or self._config
        return await search_async(
            client=self._async_client,
            config=effective_config,
            input=input,
            include_raw=include_raw,
        )
