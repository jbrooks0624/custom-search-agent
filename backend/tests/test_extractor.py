import pytest

from oai import OAI, OAIConfig
from tvly import Tavily, TavilyConfig, SearchInput
from workflow import (
    extract,
    extract_multiple,
    format_extracted_content,
    get_extraction_stats,
    scrub_markdown,
    ExtractedContent,
)


@pytest.fixture
def client() -> OAI:
    return OAI(config=OAIConfig(model="gpt-5-nano"))


SAMPLE_ML_CONTENT = """
# What is Machine Learning?

Machine learning is a subset of artificial intelligence that enables systems to learn from data.

## Key Statistics

- The global machine learning market was valued at $15.44 billion in 2021
- Expected to grow at a CAGR of 38.8% from 2022 to 2030
- 77% of devices currently use ML in some form

## Types of Machine Learning

1. **Supervised Learning**: Uses labeled data, accuracy rates can exceed 95% for well-defined problems
2. **Unsupervised Learning**: Finds patterns in unlabeled data
3. **Reinforcement Learning**: Learns through trial and error, used by DeepMind's AlphaGo

## Applications

- Healthcare: ML models can detect cancer with 94.5% accuracy
- Finance: Fraud detection systems process 500+ transactions per second
- Autonomous vehicles: Tesla's Autopilot uses 8 cameras and 12 ultrasonic sensors

According to Dr. Andrew Ng, "Machine learning is the most important general-purpose technology of our era."
"""

IRRELEVANT_CONTENT = """
# Best Pizza Recipes

Pizza is a beloved Italian dish that has become popular worldwide.

## Classic Margherita

- Fresh mozzarella cheese
- San Marzano tomatoes
- Fresh basil leaves
- Extra virgin olive oil

Bake at 450°F for 12-15 minutes until the crust is golden brown.
"""


@pytest.mark.asyncio
async def test_extract_relevant_content(client: OAI):
    """Test extraction from relevant content."""
    query = "What is machine learning and what are its applications?"
    
    result, output = await extract(client, SAMPLE_ML_CONTENT, query)
    
    assert isinstance(result, ExtractedContent)
    assert result.relevant is True
    assert len(result.facts) > 0
    
    print(f"\n--- Extract: Relevant Content ---")
    print(f"Query: {query}")
    print(f"Relevant: {result.relevant}")
    print(f"Facts extracted: {len(result.facts)}")
    for fact in result.facts:
        print(f"  • {fact}")
    print(f"Tokens: {output.usage.total_tokens}")
    print(f"Latency: {output.latency_ms:.2f}ms")


@pytest.mark.asyncio
async def test_extract_irrelevant_content(client: OAI):
    """Test extraction from irrelevant content returns empty."""
    query = "What is machine learning and what are its applications?"
    
    result, output = await extract(client, IRRELEVANT_CONTENT, query)
    
    assert isinstance(result, ExtractedContent)
    assert result.relevant is False
    assert len(result.facts) == 0
    
    print(f"\n--- Extract: Irrelevant Content ---")
    print(f"Query: {query}")
    print(f"Content topic: Pizza recipes")
    print(f"Relevant: {result.relevant}")
    print(f"Facts extracted: {len(result.facts)}")


@pytest.mark.asyncio
async def test_extract_empty_content(client: OAI):
    """Test extraction from empty content."""
    query = "What is machine learning?"
    
    result, output = await extract(client, "", query)
    
    assert result.relevant is False
    assert len(result.facts) == 0
    assert output is None


@pytest.mark.asyncio
async def test_extract_multiple_contents(client: OAI):
    """Test extracting from multiple contents concurrently."""
    query = "What is machine learning?"
    contents = [SAMPLE_ML_CONTENT, IRRELEVANT_CONTENT, SAMPLE_ML_CONTENT]
    
    results = await extract_multiple(client, contents, query)
    
    assert len(results) == 3
    assert results[0].relevant is True  # ML content
    assert results[1].relevant is False  # Pizza content
    assert results[2].relevant is True  # ML content
    
    print(f"\n--- Extract Multiple ---")
    print(f"Query: {query}")
    print(f"Sources processed: {len(results)}")
    print(f"Relevant sources: {sum(1 for r in results if r.relevant)}")
    print(f"Total facts: {sum(len(r.facts) for r in results)}")


@pytest.mark.asyncio
async def test_format_extracted_content(client: OAI):
    """Test formatting extracted content into markdown."""
    query = "What is machine learning?"
    contents = [SAMPLE_ML_CONTENT, IRRELEVANT_CONTENT]
    
    results = await extract_multiple(client, contents, query)
    formatted = format_extracted_content(results)
    
    assert len(formatted) > 0
    assert formatted.startswith("- ")  # Should be bullet points
    
    print(f"\n--- Formatted Extraction ---")
    print(f"Formatted output ({len(formatted)} chars):")
    print(formatted)


@pytest.mark.asyncio
async def test_extraction_stats(client: OAI):
    """Test extraction statistics."""
    query = "What is machine learning?"
    contents = [SAMPLE_ML_CONTENT, SAMPLE_ML_CONTENT]
    
    results = await extract_multiple(client, contents, query)
    stats = get_extraction_stats(contents, results)
    
    assert stats["sources_processed"] == 2
    assert stats["sources_relevant"] == 2
    assert stats["total_facts"] > 0
    assert stats["reduction_percent"] > 0
    
    print(f"\n--- Extraction Stats ---")
    print(f"Original: {stats['original_chars']} chars")
    print(f"Extracted: {stats['extracted_chars']} chars")
    print(f"Reduction: {stats['reduction_percent']}%")
    print(f"Sources relevant: {stats['sources_relevant']}/{stats['sources_processed']}")
    print(f"Total facts: {stats['total_facts']}")


@pytest.mark.asyncio
async def test_full_pipeline_scrub_then_extract():
    """Test the full pipeline: raw content -> scrub -> extract."""
    # Get real search results
    tavily = Tavily(config=TavilyConfig(max_results=2, include_raw_content="markdown"))
    oai = OAI(config=OAIConfig(model="gpt-5-nano"))
    
    query = "What is retrieval augmented generation RAG?"
    search_input = SearchInput(query=query)
    search_output = await tavily.search_async(search_input)
    
    # Step 1: Scrub raw content
    raw_contents = [r.raw_content for r in search_output.results if r.raw_content]
    scrubbed_contents = [scrub_markdown(c, max_chars=4000) for c in raw_contents]
    
    # Step 2: Extract relevant facts
    extractions = await extract_multiple(oai, scrubbed_contents, query)
    
    # Step 3: Format for summarizer
    formatted = format_extracted_content(extractions)
    
    print(f"\n--- Full Pipeline: Scrub → Extract ---")
    print(f"Query: {query}")
    print(f"Sources: {len(raw_contents)}")
    
    total_raw = sum(len(c) for c in raw_contents)
    total_scrubbed = sum(len(c) for c in scrubbed_contents)
    total_extracted = len(formatted)
    
    print(f"\nCompression:")
    print(f"  Raw:       {total_raw:,} chars")
    print(f"  Scrubbed:  {total_scrubbed:,} chars ({100 - total_scrubbed/total_raw*100:.1f}% reduction)")
    print(f"  Extracted: {total_extracted:,} chars ({100 - total_extracted/total_raw*100:.1f}% total reduction)")
    
    print(f"\nExtracted facts preview:")
    lines = formatted.split('\n')[:10]
    for line in lines:
        print(f"  {line}")
    if len(formatted.split('\n')) > 10:
        print(f"  ... and {len(formatted.split(chr(10))) - 10} more facts")
