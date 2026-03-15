import asyncio
import time
from collections.abc import Awaitable, Callable

from oai import OAI, ChatInput, Message, OAIConfig
from tvly import Tavily

from .extractor import extract, format_extracted_content
from .orchestrator import orchestrate
from .scrubber import scrub_markdown
from .search import search_with_output
from .summarizer import MAX_ITERATIONS, summarize, summarize_stream

StatusCallback = Callable[[str], Awaitable[None]]
ContentCallback = Callable[[str], Awaitable[None]]


async def _direct_response(client: OAI, messages: list[Message]) -> str:
    """Generate a direct response without search when no queries are needed."""
    config = OAIConfig(model="gpt-5-nano", reasoning_effort="minimal")

    chat_input = ChatInput(
        system_prompt="You are a helpful assistant. Respond naturally to the user's message.",
        messages=messages,
    )

    output = await client.chat_async(input=chat_input, config=config)
    return output.content


class Source:
    """A source used in the response."""

    def __init__(self, title: str, url: str):
        self.title = title
        self.url = url

    def to_dict(self):
        return {"title": self.title, "url": self.url}


class PipelineTimings:
    """Information for pipeline steps."""

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
        self.num_iterations: int = 1
        self.sources: list[Source] = []

    def __repr__(self):
        iterations_str = (
            f" over {self.num_iterations} iterations" if self.num_iterations > 1 else ""
        )
        return (
            f"PipelineTimings(\n"
            f"  orchestrate: {self.orchestrate_ms:.0f}ms ({self.num_queries} queries{iterations_str})\n"
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
        return_timings: If True, return pipeline metrics
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

    # If no queries needed, respond directly
    if not queries_result.queries:
        print("No queries needed, responding directly")
        start = time.perf_counter()
        response = await _direct_response(oai_client, messages)
        timings.summarize_ms = (time.perf_counter() - start) * 1000
        messages.append(Message(role="assistant", content=response))
        timings.total_ms = (time.perf_counter() - total_start) * 1000
        return timings if return_timings else None

    # Step 2: Execute searches in parallel
    start = time.perf_counter()
    search_tasks = [
        search_with_output(tavily_client, query, deep_research=False)
        for query in queries_result.queries
    ]
    search_results = await asyncio.gather(*search_tasks)
    timings.search_ms = (time.perf_counter() - start) * 1000

    # Step 3: Scrub content
    start = time.perf_counter()
    all_contents = []
    for _markdown, output in search_results:
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
        user_query = next((m.content for m in reversed(messages) if m.role == "user"), "")

        extract_tasks = [extract(oai_client, content, user_query) for content in all_contents]
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


async def run_deep_research_pipeline(
    oai_client: OAI,
    tavily_client: Tavily,
    messages: list[Message],
    max_iterations: int = MAX_ITERATIONS,
    return_timings: bool = False,
) -> PipelineTimings | None:
    """
    Run the deep research pipeline with iterative search loops.

    Pipeline flow (loops until complete or max_iterations reached):
    1. Orchestrator generates search queries (up to 5)
    2. Execute searches in parallel
    3. Scrub and extract content from each result
    4. Summarizer evaluates if more research is needed
    5. If needs_more_research=True and iterations remaining, loop back with feedback
    6. Append final assistant message to the messages list

    Args:
        oai_client: OAI client instance
        tavily_client: Tavily client instance
        messages: List of conversation messages (will be mutated with assistant response)
        max_iterations: Maximum number of research iterations (default 3)
        return_timings: If True, return pipeline metrics

    Returns:
        PipelineTimings if return_timings is True, else None
    """
    timings = PipelineTimings()
    total_start = time.perf_counter()

    num_iterations = 0
    previous_context: str | None = None
    accumulated_facts: list[str] = []

    while num_iterations < max_iterations:
        num_iterations += 1

        # Step 1: Generate search queries
        start = time.perf_counter()
        queries_result, _ = await orchestrate(
            client=oai_client,
            messages=messages,
            deep_research=True,
            previous_context=previous_context,
        )
        timings.orchestrate_ms += (time.perf_counter() - start) * 1000
        timings.num_queries += len(queries_result.queries)

        # If no queries needed (first iteration only), respond directly
        if not queries_result.queries and num_iterations == 1:
            start = time.perf_counter()
            response = await _direct_response(oai_client, messages)
            timings.summarize_ms = (time.perf_counter() - start) * 1000
            messages.append(Message(role="assistant", content=response))
            timings.total_ms = (time.perf_counter() - total_start) * 1000
            print("No queries needed, responding directly")
            return timings if return_timings else None

        # If no queries in later iterations, we're done
        if not queries_result.queries:
            break

        # Step 2: Execute searches in parallel (advanced depth for deep research)
        start = time.perf_counter()
        search_tasks = [
            search_with_output(tavily_client, query, deep_research=True)
            for query in queries_result.queries
        ]
        search_results = await asyncio.gather(*search_tasks)
        timings.search_ms += (time.perf_counter() - start) * 1000

        # Step 3: Scrub content
        start = time.perf_counter()
        all_contents = []
        for _markdown, output in search_results:
            for result in output.results:
                if result.raw_content:
                    scrubbed = scrub_markdown(result.raw_content)
                    if scrubbed:
                        all_contents.append(scrubbed)
        timings.scrub_ms += (time.perf_counter() - start) * 1000
        timings.num_sources += len(all_contents)

        # Step 4: Extract relevant facts in parallel
        start = time.perf_counter()
        user_query = next((m.content for m in reversed(messages) if m.role == "user"), "")

        extract_tasks = [extract(oai_client, content, user_query) for content in all_contents]
        extraction_results = await asyncio.gather(*extract_tasks)
        timings.extract_ms += (time.perf_counter() - start) * 1000
        timings.num_extractions += len(extraction_results)

        # Accumulate facts across iterations
        for extracted, _ in extraction_results:
            if extracted.relevant and extracted.facts:
                accumulated_facts.extend(extracted.facts)

        # Format all accumulated facts for summarizer
        context = "\n".join(f"• {fact}" for fact in accumulated_facts)

        # Step 5: Summarize and evaluate
        start = time.perf_counter()
        summary_response, _ = await summarize(
            client=oai_client,
            messages=messages,
            search_context=context,
            deep_research=True,
            num_iterations=num_iterations,
            max_iterations=max_iterations,
        )
        timings.summarize_ms += (time.perf_counter() - start) * 1000

        # Check if we should continue or are done
        if not summary_response.needs_more_research:
            break

        # Prepare context for next iteration
        previous_context = f"""=== ACCUMULATED RESEARCH (Iteration {num_iterations}) ===
{context}

=== FEEDBACK FROM SUMMARIZER ===
{summary_response.content}

Please generate new queries to address the missing information above."""

    timings.num_iterations = num_iterations

    # Step 6: Append final assistant response to messages
    messages.append(Message(role="assistant", content=summary_response.content))

    timings.total_ms = (time.perf_counter() - total_start) * 1000

    return timings if return_timings else None


async def run_search_pipeline_with_status(
    oai_client: OAI,
    tavily_client: Tavily,
    messages: list[Message],
    status_callback: StatusCallback,
    skip_extraction: bool = False,
    content_callback: ContentCallback | None = None,
) -> PipelineTimings:
    """
    Run the standard search pipeline with status updates.
    """
    timings = PipelineTimings()
    total_start = time.perf_counter()

    # Step 1: Generate search queries
    await status_callback("Planning search...")
    start = time.perf_counter()
    queries_result, _ = await orchestrate(
        client=oai_client,
        messages=messages,
        deep_research=False,
    )
    timings.orchestrate_ms = (time.perf_counter() - start) * 1000
    timings.num_queries = len(queries_result.queries)

    # If no queries needed, respond directly
    if not queries_result.queries:
        await status_callback("Generating response...")
        start = time.perf_counter()
        response = await _direct_response(oai_client, messages)
        timings.summarize_ms = (time.perf_counter() - start) * 1000
        messages.append(Message(role="assistant", content=response))
        timings.total_ms = (time.perf_counter() - total_start) * 1000
        return timings

    # Step 2: Execute searches in parallel
    await status_callback("Searching the web...")
    start = time.perf_counter()
    search_tasks = [
        search_with_output(tavily_client, query, deep_research=False)
        for query in queries_result.queries
    ]
    search_results = await asyncio.gather(*search_tasks)
    timings.search_ms = (time.perf_counter() - start) * 1000

    # Step 3: Scrub content and collect sources
    start = time.perf_counter()
    all_contents = []
    seen_urls = set()
    for _markdown, output in search_results:
        for result in output.results:
            if result.raw_content:
                scrubbed = scrub_markdown(result.raw_content)
                if scrubbed:
                    all_contents.append(scrubbed)
                    if result.url and result.url not in seen_urls:
                        seen_urls.add(result.url)
                        timings.sources.append(
                            Source(title=result.title or result.url, url=result.url)
                        )
    timings.scrub_ms = (time.perf_counter() - start) * 1000
    timings.num_sources = len(all_contents)

    # Step 4: Extract relevant facts in parallel (or skip)
    if not skip_extraction:
        await status_callback("Analyzing sources...")
        start = time.perf_counter()
        user_query = next((m.content for m in reversed(messages) if m.role == "user"), "")

        extract_tasks = [extract(oai_client, content, user_query) for content in all_contents]
        extraction_results = await asyncio.gather(*extract_tasks)
        timings.extract_ms = (time.perf_counter() - start) * 1000
        timings.num_extractions = len(extraction_results)

        extractions = [result for result, _ in extraction_results]
        context = format_extracted_content(extractions)
    else:
        context = "\n\n---\n\n".join(all_contents)
        timings.extract_ms = 0
        timings.num_extractions = 0

    # Step 5: Summarize and generate final answer (stream if content_callback provided)
    await status_callback("Generating response...")
    start = time.perf_counter()
    if content_callback is not None:
        full_content: list[str] = []
        async for delta in summarize_stream(
            client=oai_client,
            messages=messages,
            search_context=context,
            num_iterations=1,
        ):
            full_content.append(delta)
            await content_callback(delta)
        summary_content = "".join(full_content)
    else:
        summary_response, _ = await summarize(
            client=oai_client,
            messages=messages,
            search_context=context,
            deep_research=False,
        )
        summary_content = summary_response.content
    timings.summarize_ms = (time.perf_counter() - start) * 1000

    # Step 6: Append assistant response to messages
    messages.append(Message(role="assistant", content=summary_content))

    timings.total_ms = (time.perf_counter() - total_start) * 1000

    return timings


async def run_deep_research_pipeline_with_status(
    oai_client: OAI,
    tavily_client: Tavily,
    messages: list[Message],
    status_callback: StatusCallback,
    max_iterations: int = MAX_ITERATIONS,
    content_callback: ContentCallback | None = None,
) -> PipelineTimings:
    """
    Run the deep research pipeline with status updates.
    """
    timings = PipelineTimings()
    total_start = time.perf_counter()

    num_iterations = 0
    previous_context: str | None = None
    accumulated_facts: list[str] = []

    while num_iterations < max_iterations:
        num_iterations += 1
        print(f"Iteration {num_iterations}")
        # Step 1: Generate search queries
        await status_callback("Planning research...")
        start = time.perf_counter()
        queries_result, _ = await orchestrate(
            client=oai_client,
            messages=messages,
            deep_research=True,
            previous_context=previous_context,
        )
        timings.orchestrate_ms += (time.perf_counter() - start) * 1000
        timings.num_queries += len(queries_result.queries)

        # If no queries needed (first iteration only), respond directly
        if not queries_result.queries and num_iterations == 1:
            await status_callback("Generating response...")
            start = time.perf_counter()
            response = await _direct_response(oai_client, messages)
            timings.summarize_ms = (time.perf_counter() - start) * 1000
            messages.append(Message(role="assistant", content=response))
            timings.total_ms = (time.perf_counter() - total_start) * 1000
            return timings

        if not queries_result.queries:
            break

        # Step 2: Execute searches in parallel (advanced depth for deep research)
        await status_callback("Searching the web...")
        start = time.perf_counter()
        search_tasks = [
            search_with_output(tavily_client, query, deep_research=True)
            for query in queries_result.queries
        ]
        search_results = await asyncio.gather(*search_tasks)
        timings.search_ms += (time.perf_counter() - start) * 1000

        # Step 3: Scrub content and collect sources
        start = time.perf_counter()
        all_contents = []
        seen_urls = {s.url for s in timings.sources}
        for _markdown, output in search_results:
            for result in output.results:
                if result.raw_content:
                    scrubbed = scrub_markdown(result.raw_content)
                    if scrubbed:
                        all_contents.append(scrubbed)
                        if result.url and result.url not in seen_urls:
                            seen_urls.add(result.url)
                            timings.sources.append(
                                Source(title=result.title or result.url, url=result.url)
                            )
        timings.scrub_ms += (time.perf_counter() - start) * 1000
        timings.num_sources += len(all_contents)

        # Step 4: Extract relevant facts in parallel
        await status_callback("Analyzing sources...")
        start = time.perf_counter()
        user_query = next((m.content for m in reversed(messages) if m.role == "user"), "")

        extract_tasks = [extract(oai_client, content, user_query) for content in all_contents]
        extraction_results = await asyncio.gather(*extract_tasks)
        timings.extract_ms += (time.perf_counter() - start) * 1000
        timings.num_extractions += len(extraction_results)

        for extracted, _ in extraction_results:
            if extracted.relevant and extracted.facts:
                accumulated_facts.extend(extracted.facts)

        context = "\n".join(f"• {fact}" for fact in accumulated_facts)

        # Step 5: Summarize and evaluate (stream only on final iteration when callback provided)
        await status_callback("Synthesizing findings...")
        start = time.perf_counter()
        is_final_round = num_iterations == max_iterations
        stream_final = content_callback is not None and is_final_round

        if stream_final:
            full_content_list: list[str] = []
            async for delta in summarize_stream(
                client=oai_client,
                messages=messages,
                search_context=context,
                num_iterations=num_iterations,
            ):
                full_content_list.append(delta)
                await content_callback(delta)
            summary_content = "".join(full_content_list)
        else:
            summary_response, _ = await summarize(
                client=oai_client,
                messages=messages,
                search_context=context,
                deep_research=True,
                num_iterations=num_iterations,
                max_iterations=max_iterations,
            )
            summary_content = summary_response.content
            # Require at least 2 iterations in deep research so we don't stop after one round
            force_another_round = num_iterations == 1 and max_iterations >= 2
            if summary_response.needs_more_research or force_another_round:
                if force_another_round and not summary_response.needs_more_research:
                    previous_context = f"""=== ACCUMULATED RESEARCH (Iteration {num_iterations}) ===
{context}

=== FEEDBACK ===
Run another round of search to deepen coverage. Target any gaps: specific figures (GDP, unemployment, inflation, debt by administration), and main criticisms and defenses from both sides."""
                else:
                    previous_context = f"""=== ACCUMULATED RESEARCH (Iteration {num_iterations}) ===
{context}

=== FEEDBACK FROM SUMMARIZER ===
{summary_response.content}

Please generate new queries to address the missing information above."""
                timings.summarize_ms += (time.perf_counter() - start) * 1000
                continue

        timings.summarize_ms += (time.perf_counter() - start) * 1000
        break

    timings.num_iterations = num_iterations
    messages.append(Message(role="assistant", content=summary_content))
    timings.total_ms = (time.perf_counter() - total_start) * 1000

    return timings
