import asyncio

import pytest

from oai import OAI, OAIConfig
from workflow import (
    ExtractedContent,
    extract,
    format_extracted_content,
    scrub_markdown,
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

    print("\n--- Extract: Relevant Content ---")
    print(f"Query: {query}")
    print(f"Relevant: {result.relevant}")
    print(f"Facts extracted: {len(result.facts)}")
    for fact in result.facts:
        print(f"  • {fact}")
    print(f"Tokens: {output.usage.total_tokens}")
    print(f"Duration: {output.latency_ms:.2f}ms")


@pytest.mark.asyncio
async def test_extract_irrelevant_content(client: OAI):
    """Test extraction from irrelevant content returns empty."""
    query = "What is machine learning and what are its applications?"

    result, output = await extract(client, IRRELEVANT_CONTENT, query)

    assert isinstance(result, ExtractedContent)
    assert result.relevant is False
    assert len(result.facts) == 0

    print("\n--- Extract: Irrelevant Content ---")
    print(f"Query: {query}")
    print("Content topic: Pizza recipes")
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

    tasks = [extract(client, content, query) for content in contents]
    results_with_output = await asyncio.gather(*tasks)
    results = [r for r, _ in results_with_output]

    assert len(results) == 3
    assert results[0].relevant is True  # ML content
    assert results[1].relevant is False  # Pizza content
    assert results[2].relevant is True  # ML content

    print("\n--- Extract Multiple ---")
    print(f"Query: {query}")
    print(f"Sources processed: {len(results)}")
    print(f"Relevant sources: {sum(1 for r in results if r.relevant)}")
    print(f"Total facts: {sum(len(r.facts) for r in results)}")


@pytest.mark.asyncio
async def test_format_extracted_content(client: OAI):
    """Test formatting extracted content into markdown."""
    query = "What is machine learning?"
    contents = [SAMPLE_ML_CONTENT, IRRELEVANT_CONTENT]

    tasks = [extract(client, content, query) for content in contents]
    results_with_output = await asyncio.gather(*tasks)
    results = [r for r, _ in results_with_output]
    formatted = format_extracted_content(results)

    assert len(formatted) > 0
    assert formatted.startswith("- ")  # Should be bullet points

    print("\n--- Formatted Extraction ---")
    print(f"Formatted output ({len(formatted)} chars):")
    print(formatted)
