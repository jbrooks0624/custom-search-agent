import asyncio
import time

from oai import OAI, Message
from tvly import Tavily

from .orchestrator import orchestrate
from .search import search_with_output
from .scrubber import scrub_markdown
from .extractor import extract, format_extracted_content
from .summarizer import summarize


class PipelineTimings:
    """Timing information for pipeline steps."""
    def __init__(self):
        self.orchestrate_ms: float = 0
        self.search_ms: float = 0
        self.scrub_ms: float = 0
        self.extract_ms: float = 0
        self.summarize_ms: float = 0
        self.total_ms: float = 0
        self.num_queries: int = 0
        self.num_sources: int = 0
        self.num_extractions: int = 0
    
    def __repr__(self):
        return (
            f"PipelineTimings(\n"
            f"  orchestrate: {self.orchestrate_ms:.0f}ms ({self.num_queries} queries)\n"
            f"  search:      {self.search_ms:.0f}ms\n"
            f"  scrub:       {self.scrub_ms:.0f}ms ({self.num_sources} sources)\n"
            f"  extract:     {self.extract_ms:.0f}ms ({self.num_extractions} extractions)\n"
            f"  summarize:   {self.summarize_ms:.0f}ms\n"
            f"  total:       {self.total_ms:.0f}ms\n"
            f")"
        )


async def run_search_pipeline(
    oai_client: OAI,
    tavily_client: Tavily,
    messages: list[Message],
    return_timings: bool = False,
    skip_extraction: bool = False,
) -> PipelineTimings | None:
    """
    Run the standard search pipeline and append the assistant response to messages.
    
    Pipeline flow:
    1. Orchestrator generates search queries
    2. Execute searches in parallel
    3. Scrub and extract content from each result in parallel
    4. Summarizer synthesizes final answer
    5. Append assistant message to the messages list
    
    Args:
        oai_client: OAI client instance
        tavily_client: Tavily client instance
        messages: List of conversation messages (will be mutated with assistant response)
        return_timings: If True, return timing information
        skip_extraction: If True, skip LLM extraction and pass scrubbed content directly
    
    Returns:
        PipelineTimings if return_timings is True, else None
    """
    timings = PipelineTimings()
    total_start = time.perf_counter()
    
    # Step 1: Generate search queries
    start = time.perf_counter()
    queries_result, _ = await orchestrate(
        client=oai_client,
        messages=messages,
        deep_research=False,
    )
    timings.orchestrate_ms = (time.perf_counter() - start) * 1000
    timings.num_queries = len(queries_result.queries)
    
    # Step 2: Execute searches in parallel
    start = time.perf_counter()
    search_tasks = [
        search_with_output(tavily_client, query)
        for query in queries_result.queries
    ]
    search_results = await asyncio.gather(*search_tasks)
    timings.search_ms = (time.perf_counter() - start) * 1000
    
    # Step 3: Scrub content
    start = time.perf_counter()
    all_contents = []
    for markdown, output in search_results:
        for result in output.results:
            if result.raw_content:
                scrubbed = scrub_markdown(result.raw_content)
                if scrubbed:
                    all_contents.append(scrubbed)
    timings.scrub_ms = (time.perf_counter() - start) * 1000
    timings.num_sources = len(all_contents)
    
    # Step 4: Extract relevant facts in parallel (or skip)
    start = time.perf_counter()
    
    if skip_extraction:
        # Skip extraction, join scrubbed content directly
        context = "\n\n---\n\n".join(all_contents)
        timings.extract_ms = 0
        timings.num_extractions = 0
    else:
        user_query = next((m.content for m in messages if m.role == "user"), "")
        
        extract_tasks = [
            extract(oai_client, content, user_query)
            for content in all_contents
        ]
        extraction_results = await asyncio.gather(*extract_tasks)
        timings.extract_ms = (time.perf_counter() - start) * 1000
        timings.num_extractions = len(extraction_results)
        
        # Format extracted content
        extractions = [result for result, _ in extraction_results]
        context = format_extracted_content(extractions)
    
    # Step 5: Summarize and generate final answer
    start = time.perf_counter()
    summary_response, _ = await summarize(
        client=oai_client,
        messages=messages,
        search_context=context,
        deep_research=False,
    )
    timings.summarize_ms = (time.perf_counter() - start) * 1000
    
    # Step 6: Append assistant response to messages
    messages.append(Message(role="assistant", content=summary_response.content))
    
    timings.total_ms = (time.perf_counter() - total_start) * 1000
    
    return timings if return_timings else None
