"""
Performance benchmarking tests for search and deep research pipelines.

This module provides detailed breakdowns for:
- Individual pipeline steps (orchestrate, search, scrub, extract, summarize)
- Per-operation metrics (time per query, time per source, time per extraction)
- Concurrent vs sequential performance comparisons
- Bottleneck identification

Run with: pytest tests/test_performance.py -v -s
"""

import asyncio
import statistics
import time
from dataclasses import dataclass, field

import pytest

from oai import OAI, Message, OAIConfig
from tvly import Tavily, TavilyConfig
from workflow import run_deep_research_pipeline, run_search_pipeline
from workflow.extractor import extract
from workflow.orchestrator import orchestrate
from workflow.scrubber import scrub_markdown
from workflow.search import search_with_output
from workflow.summarizer import summarize


@dataclass
class StepTiming:
    """Metrics for a single operation."""

    name: str
    duration_ms: float
    count: int = 1
    metadata: dict = field(default_factory=dict)

    @property
    def avg_ms(self) -> float:
        return self.duration_ms / self.count if self.count > 0 else 0

    def __repr__(self) -> str:
        if self.count > 1:
            return f"{self.name}: {self.duration_ms:.0f}ms total ({self.avg_ms:.0f}ms avg × {self.count})"
        return f"{self.name}: {self.duration_ms:.0f}ms"


@dataclass
class PerformanceReport:
    """Detailed performance breakdown for a pipeline run."""

    test_name: str
    query: str
    timings: list[StepTiming] = field(default_factory=list)
    total_ms: float = 0
    bottleneck: str = ""
    bottleneck_pct: float = 0

    def add_timing(self, name: str, duration_ms: float, count: int = 1, **metadata):
        self.timings.append(StepTiming(name, duration_ms, count, metadata))

    def finalize(self):
        if not self.timings:
            return
        self.total_ms = sum(t.duration_ms for t in self.timings)
        slowest = max(self.timings, key=lambda t: t.duration_ms)
        self.bottleneck = slowest.name
        self.bottleneck_pct = (slowest.duration_ms / self.total_ms * 100) if self.total_ms > 0 else 0

    def __repr__(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"PERFORMANCE REPORT: {self.test_name}",
            f"{'='*60}",
            f"Query: {self.query[:80]}{'...' if len(self.query) > 80 else ''}",
            "",
            "Step Breakdown:",
        ]
        for t in self.timings:
            pct = (t.duration_ms / self.total_ms * 100) if self.total_ms > 0 else 0
            bar = "█" * int(pct / 5)
            lines.append(f"  {t.name:20} {t.duration_ms:8.0f}ms ({pct:5.1f}%) {bar}")
            if t.count > 1:
                lines.append(f"    └─ {t.count} ops @ {t.avg_ms:.0f}ms avg")

        lines.extend([
            "",
            f"Total:     {self.total_ms:.0f}ms",
            f"Bottleneck: {self.bottleneck} ({self.bottleneck_pct:.1f}% of total)",
            f"{'='*60}",
        ])
        return "\n".join(lines)


@pytest.fixture
def oai_client() -> OAI:
    return OAI(config=OAIConfig(model="gpt-5-mini"))


@pytest.fixture
def tavily_client() -> Tavily:
    return Tavily(config=TavilyConfig(max_results=2))


class TestIndividualStepPerformance:
    """Test performance of individual pipeline steps in isolation."""

    @pytest.mark.asyncio
    async def test_orchestrator_latency(self, oai_client: OAI):
        """Measure orchestrator LLM call duration."""
        messages = [Message(role="user", content="What is the current state of AI in healthcare?")]

        timings = []
        for i in range(3):
            start = time.perf_counter()
            result, _ = await orchestrate(oai_client, messages, deep_research=False)
            duration_ms = (time.perf_counter() - start) * 1000
            timings.append(duration_ms)

        print(f"\n--- Orchestrator (3 runs) ---")
        print(f"Times: {[f'{t:.0f}ms' for t in timings]}")
        print(f"Mean:  {statistics.mean(timings):.0f}ms")
        print(f"Stdev: {statistics.stdev(timings):.0f}ms")
        print(f"Min:   {min(timings):.0f}ms")
        print(f"Max:   {max(timings):.0f}ms")

        # First call often has cold start overhead
        warm_mean = statistics.mean(timings[1:])
        print(f"Warm mean (excluding first): {warm_mean:.0f}ms")

    @pytest.mark.asyncio
    async def test_search_latency(self, tavily_client: Tavily):
        """Measure Tavily search API duration."""
        queries = [
            "artificial intelligence healthcare 2026",
            "machine learning medical diagnosis",
            "AI drug discovery latest research",
        ]

        print(f"\n--- Search (per query) ---")
        timings = []
        for query in queries:
            start = time.perf_counter()
            result = await search_with_output(tavily_client, query)
            duration_ms = (time.perf_counter() - start) * 1000
            timings.append(duration_ms)
            print(f"  '{query[:40]}...': {duration_ms:.0f}ms")

        print(f"\nMean: {statistics.mean(timings):.0f}ms")

    @pytest.mark.asyncio
    async def test_search_parallel_vs_sequential(self, tavily_client: Tavily):
        """Compare parallel vs sequential search execution."""
        queries = [
            "AI in healthcare 2026",
            "machine learning diagnosis",
            "deep learning medical imaging",
        ]

        # Sequential
        start = time.perf_counter()
        for query in queries:
            await search_with_output(tavily_client, query)
        sequential_ms = (time.perf_counter() - start) * 1000

        # Parallel
        start = time.perf_counter()
        tasks = [search_with_output(tavily_client, q) for q in queries]
        await asyncio.gather(*tasks)
        parallel_ms = (time.perf_counter() - start) * 1000

        speedup = sequential_ms / parallel_ms if parallel_ms > 0 else 0

        print(f"\n--- Search: Parallel vs Sequential ({len(queries)} queries) ---")
        print(f"Sequential: {sequential_ms:.0f}ms")
        print(f"Parallel:   {parallel_ms:.0f}ms")
        print(f"Speedup:    {speedup:.2f}x")

    @pytest.mark.asyncio
    async def test_scrubber_performance(self):
        """Measure scrubber performance on different content sizes."""
        test_contents = [
            ("Small (1KB)", "Test content. " * 50),
            ("Medium (10KB)", "Test content with some markdown. # Header\n\n" * 200),
            ("Large (50KB)", "# Header\n\n" + "This is test content with various formatting.\n\n" * 1000),
        ]

        print(f"\n--- Scrubber Performance ---")
        for name, content in test_contents:
            iterations = 100
            start = time.perf_counter()
            for _ in range(iterations):
                scrub_markdown(content)
            duration_ms = (time.perf_counter() - start) * 1000

            print(f"{name}: {duration_ms:.1f}ms total, {duration_ms/iterations:.2f}ms avg")

    @pytest.mark.asyncio
    async def test_extractor_latency(self, oai_client: OAI):
        """Measure extractor LLM call duration."""
        content = """
        # AI in Healthcare: A Comprehensive Overview

        Artificial intelligence is transforming healthcare in numerous ways.
        According to a 2026 study, AI diagnostic tools have achieved 94% accuracy
        in detecting certain cancers. The global AI healthcare market is expected
        to reach $45 billion by 2028.

        Key applications include:
        - Medical imaging analysis
        - Drug discovery acceleration
        - Patient outcome prediction
        - Administrative workflow optimization
        """
        query = "What is the accuracy of AI diagnostic tools?"

        timings = []
        for _ in range(3):
            start = time.perf_counter()
            await extract(oai_client, content, query)
            duration_ms = (time.perf_counter() - start) * 1000
            timings.append(duration_ms)

        print(f"\n--- Extractor (3 runs) ---")
        print(f"Times: {[f'{t:.0f}ms' for t in timings]}")
        print(f"Mean:  {statistics.mean(timings):.0f}ms")

    @pytest.mark.asyncio
    async def test_extraction_parallel_scaling(self, oai_client: OAI):
        """Test how extraction scales with parallel sources."""
        content = "Test content with facts. AI accuracy is 95%. Market size is $50B."
        query = "What are the key statistics?"

        for num_sources in [1, 2, 4, 6]:
            tasks = [extract(oai_client, content, query) for _ in range(num_sources)]
            start = time.perf_counter()
            await asyncio.gather(*tasks)
            duration_ms = (time.perf_counter() - start) * 1000

            per_source = duration_ms / num_sources
            print(f"{num_sources} sources parallel: {duration_ms:.0f}ms total, {per_source:.0f}ms effective/source")

    @pytest.mark.asyncio
    async def test_summarizer_latency(self, oai_client: OAI):
        """Measure summarizer LLM call duration."""
        messages = [Message(role="user", content="What is AI used for in healthcare?")]
        context = """
        - AI diagnostic tools achieve 94% accuracy in cancer detection
        - Global AI healthcare market expected to reach $45B by 2028
        - Key applications: imaging, drug discovery, predictions, admin
        """

        timings = []
        for _ in range(3):
            start = time.perf_counter()
            await summarize(oai_client, messages, context, deep_research=False)
            duration_ms = (time.perf_counter() - start) * 1000
            timings.append(duration_ms)

        print(f"\n--- Summarizer (3 runs) ---")
        print(f"Times: {[f'{t:.0f}ms' for t in timings]}")
        print(f"Mean:  {statistics.mean(timings):.0f}ms")


class TestPipelinePerformance:
    """Test full pipeline performance with detailed breakdowns."""

    @pytest.mark.asyncio
    async def test_search_pipeline_detailed_timing(self, oai_client: OAI, tavily_client: Tavily):
        """Get detailed breakdown for search pipeline."""
        query = "What are the latest developments in quantum computing?"
        messages = [Message(role="user", content=query)]

        report = PerformanceReport("Search Pipeline", query)

        total_start = time.perf_counter()

        # Step 1: Orchestrate
        start = time.perf_counter()
        queries_result, _ = await orchestrate(oai_client, messages, deep_research=False)
        report.add_timing("orchestrate", (time.perf_counter() - start) * 1000,
                          num_queries=len(queries_result.queries))

        # Step 2: Search (parallel)
        start = time.perf_counter()
        search_tasks = [search_with_output(tavily_client, q) for q in queries_result.queries]
        search_results = await asyncio.gather(*search_tasks)
        report.add_timing("search", (time.perf_counter() - start) * 1000,
                          count=len(queries_result.queries))

        # Step 3: Scrub
        start = time.perf_counter()
        all_contents = []
        for _markdown, output in search_results:
            for result in output.results:
                if result.raw_content:
                    scrubbed = scrub_markdown(result.raw_content)
                    if scrubbed:
                        all_contents.append(scrubbed)
        report.add_timing("scrub", (time.perf_counter() - start) * 1000,
                          count=len(all_contents))

        # Step 4: Extract (parallel)
        start = time.perf_counter()
        user_query = messages[0].content
        extract_tasks = [extract(oai_client, content, user_query) for content in all_contents]
        extraction_results = await asyncio.gather(*extract_tasks)
        report.add_timing("extract", (time.perf_counter() - start) * 1000,
                          count=len(extraction_results))

        # Step 5: Summarize
        context = "\n".join(f"- {fact}" for e, _ in extraction_results
                            if e.relevant for fact in e.facts)
        start = time.perf_counter()
        await summarize(oai_client, messages, context, deep_research=False)
        report.add_timing("summarize", (time.perf_counter() - start) * 1000)

        report.total_ms = (time.perf_counter() - total_start) * 1000
        report.finalize()

        print(report)

    @pytest.mark.asyncio
    async def test_deep_research_pipeline_detailed_timing(self, oai_client: OAI, tavily_client: Tavily):
        """Get detailed breakdown for deep research pipeline."""
        query = "Compare the economic policies of recent US administrations with specific GDP and unemployment data"
        messages = [Message(role="user", content=query)]

        timings = await run_deep_research_pipeline(
            oai_client, tavily_client, messages, max_iterations=2, return_timings=True
        )

        report = PerformanceReport("Deep Research Pipeline", query)
        report.add_timing("orchestrate", timings.orchestrate_ms, count=timings.num_iterations)
        report.add_timing("search", timings.search_ms, count=timings.num_queries)
        report.add_timing("scrub", timings.scrub_ms, count=timings.num_sources)
        report.add_timing("extract", timings.extract_ms, count=timings.num_extractions)
        report.add_timing("summarize", timings.summarize_ms, count=timings.num_iterations)

        report.total_ms = timings.total_ms
        report.finalize()

        print(report)
        print(f"\nIterations completed: {timings.num_iterations}")

    @pytest.mark.asyncio
    async def test_pipeline_with_vs_without_extraction(self, oai_client: OAI, tavily_client: Tavily):
        """Compare pipeline performance with and without extraction step."""
        query = "What is retrieval augmented generation?"

        # With extraction
        messages_with = [Message(role="user", content=query)]
        timings_with = await run_search_pipeline(
            oai_client, tavily_client, messages_with,
            return_timings=True, skip_extraction=False
        )

        # Without extraction
        messages_without = [Message(role="user", content=query)]
        timings_without = await run_search_pipeline(
            oai_client, tavily_client, messages_without,
            return_timings=True, skip_extraction=True
        )

        print(f"\n--- With vs Without Extraction ---")
        print(f"\nWITH extraction:")
        print(f"  Extract step:  {timings_with.extract_ms:.0f}ms ({timings_with.num_extractions} sources)")
        print(f"  Total:         {timings_with.total_ms:.0f}ms")

        print(f"\nWITHOUT extraction:")
        print(f"  Extract step:  {timings_without.extract_ms:.0f}ms (skipped)")
        print(f"  Total:         {timings_without.total_ms:.0f}ms")

        savings = timings_with.total_ms - timings_without.total_ms
        pct = (savings / timings_with.total_ms * 100) if timings_with.total_ms > 0 else 0
        print(f"\nTime saved by skipping extraction: {savings:.0f}ms ({pct:.1f}%)")


class TestBottleneckAnalysis:
    """Identify and analyze performance bottlenecks."""

    @pytest.mark.asyncio
    async def test_llm_call_breakdown(self, oai_client: OAI, tavily_client: Tavily):
        """Analyze time spent in LLM calls vs other operations."""
        query = "What is machine learning?"
        messages = [Message(role="user", content=query)]

        timings = await run_search_pipeline(
            oai_client, tavily_client, messages, return_timings=True
        )

        llm_time = timings.orchestrate_ms + timings.extract_ms + timings.summarize_ms
        network_time = timings.search_ms
        cpu_time = timings.scrub_ms

        print(f"\n--- Time Distribution ---")
        print(f"LLM calls:     {llm_time:.0f}ms ({llm_time/timings.total_ms*100:.1f}%)")
        print(f"  - Orchestrate: {timings.orchestrate_ms:.0f}ms")
        print(f"  - Extract:     {timings.extract_ms:.0f}ms ({timings.num_extractions} calls)")
        print(f"  - Summarize:   {timings.summarize_ms:.0f}ms")
        print(f"Network (Tavily): {network_time:.0f}ms ({network_time/timings.total_ms*100:.1f}%)")
        print(f"CPU (scrub):      {cpu_time:.0f}ms ({cpu_time/timings.total_ms*100:.1f}%)")
        print(f"Total:            {timings.total_ms:.0f}ms")

    @pytest.mark.asyncio
    async def test_extraction_scaling_impact(self, oai_client: OAI, tavily_client: Tavily):
        """Test how extraction time scales with number of sources."""
        queries_to_test = [
            ("Simple (1 query)", "What is Python?"),
            ("Complex (likely 3 queries)", "Compare Python, JavaScript, and Rust for web development with performance benchmarks"),
        ]

        print(f"\n--- Extraction Scaling Impact ---")
        for name, query in queries_to_test:
            messages = [Message(role="user", content=query)]
            timings = await run_search_pipeline(
                oai_client, tavily_client, messages, return_timings=True
            )

            extract_per_source = (timings.extract_ms / timings.num_extractions
                                  if timings.num_extractions > 0 else 0)

            print(f"\n{name}:")
            print(f"  Queries:     {timings.num_queries}")
            print(f"  Sources:     {timings.num_sources}")
            print(f"  Extractions: {timings.num_extractions}")
            print(f"  Extract time: {timings.extract_ms:.0f}ms ({extract_per_source:.0f}ms/source)")
            print(f"  Total time:   {timings.total_ms:.0f}ms")


class TestConcurrencyOptimization:
    """Test opportunities for improved concurrency."""

    @pytest.mark.asyncio
    async def test_scrubber_parallelization_potential(self):
        """Measure potential gains from parallelizing scrubber."""
        import concurrent.futures

        contents = ["# Test\n\nContent here " * 100 for _ in range(6)]

        # Current: Sequential
        start = time.perf_counter()
        for content in contents:
            scrub_markdown(content)
        sequential_ms = (time.perf_counter() - start) * 1000

        # Potential: Thread pool (CPU-bound so threads, not asyncio)
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(scrub_markdown, contents))
        parallel_ms = (time.perf_counter() - start) * 1000

        print(f"\n--- Scrubber Parallelization Potential ---")
        print(f"Sequential ({len(contents)} sources): {sequential_ms:.1f}ms")
        print(f"Parallel (ThreadPool):  {parallel_ms:.1f}ms")
        print(f"Speedup: {sequential_ms/parallel_ms:.2f}x")

    @pytest.mark.asyncio
    async def test_overlapping_operations_potential(self, oai_client: OAI, tavily_client: Tavily):
        """Estimate gains from overlapping orchestrate with search start."""
        query = "What is quantum computing?"
        messages = [Message(role="user", content=query)]

        # Current: Wait for orchestrate to complete before starting search
        start = time.perf_counter()
        queries_result, _ = await orchestrate(oai_client, messages, deep_research=False)
        orchestrate_time = time.perf_counter() - start

        if queries_result.queries:
            start = time.perf_counter()
            tasks = [search_with_output(tavily_client, q) for q in queries_result.queries]
            await asyncio.gather(*tasks)
            search_time = time.perf_counter() - start

            print(f"\n--- Overlapping Operations Analysis ---")
            print(f"Orchestrate: {orchestrate_time*1000:.0f}ms")
            print(f"Search:      {search_time*1000:.0f}ms")
            print(f"\nNote: These currently run sequentially.")
            print("Potential optimization: Stream queries as they're generated")
            print("and start searching while orchestrator continues.")


class TestRegressionBaseline:
    """Establish performance baselines for regression detection."""

    @pytest.mark.asyncio
    async def test_establish_baseline(self, oai_client: OAI, tavily_client: Tavily):
        """
        Run standard benchmark queries.
        Use this to detect performance regressions.
        """
        benchmarks = [
            ("simple_factual", "What is the capital of France?"),
            ("technical_query", "What is retrieval augmented generation?"),
            ("comparison_query", "Compare React and Vue.js for web development"),
        ]

        print(f"\n{'='*60}")
        print("PERFORMANCE BASELINE")
        print(f"{'='*60}")

        results = []
        for name, query in benchmarks:
            messages = [Message(role="user", content=query)]
            timings = await run_search_pipeline(
                oai_client, tavily_client, messages, return_timings=True
            )

            results.append({
                "name": name,
                "query": query,
                "total_ms": timings.total_ms,
                "orchestrate_ms": timings.orchestrate_ms,
                "search_ms": timings.search_ms,
                "extract_ms": timings.extract_ms,
                "summarize_ms": timings.summarize_ms,
                "num_queries": timings.num_queries,
                "num_sources": timings.num_sources,
            })

            print(f"\n{name}:")
            print(f"  Total: {timings.total_ms:.0f}ms")
            print(f"  Steps: orch={timings.orchestrate_ms:.0f}, search={timings.search_ms:.0f}, "
                  f"extract={timings.extract_ms:.0f}, summarize={timings.summarize_ms:.0f}")

        print(f"\n{'='*60}")
        avg_total = statistics.mean(r["total_ms"] for r in results)
        print(f"Average total time: {avg_total:.0f}ms")
        print(f"{'='*60}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
