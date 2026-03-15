from pydantic import BaseModel, Field

from oai import OAI, OAIConfig, ChatInput, Message, ChatOutput


class ExtractedContent(BaseModel):
    """Structured output from the content extractor."""
    relevant: bool = Field(
        description="Whether the content contains information relevant to the query"
    )
    facts: list[str] = Field(
        default_factory=list,
        description="List of relevant facts, numbers, quotes, and key points extracted from the content"
    )


EXTRACTOR_SYSTEM_PROMPT = """You are a precise information extractor. Your task is to extract only the facts, data points, statistics, quotes, and key information that are directly relevant to answering the user's query.

Guidelines:
1. Extract ONLY information that helps answer the user's specific question
2. Format each fact as a concise bullet point
3. Include specific numbers, dates, percentages, and quotes when available
4. Preserve source attribution if mentioned (e.g., "According to [source]...")
5. If the content is completely irrelevant to the query, set relevant to false and return an empty facts list
6. Do not add interpretation or analysis - just extract the raw facts
7. Aim for 5-15 bullet points per source, focusing on the most important information

Be ruthless about relevance - if a fact doesn't help answer the query, don't include it."""


async def extract(
    client: OAI,
    content: str,
    query: str,
) -> tuple[ExtractedContent, ChatOutput]:
    """
    Extract relevant facts from scrubbed content based on the user's query.
    
    Args:
        client: OAI client instance
        content: Scrubbed markdown content from a single source
        query: The user's original query
    
    Returns:
        Tuple of (ExtractedContent, ChatOutput)
    """
    if not content or not content.strip():
        return ExtractedContent(relevant=False, facts=[]), None
    
    config = OAIConfig(model="gpt-5-nano", reasoning_effort="minimal")
    
    user_message = f"""User Query: {query}

Content to Extract From:
{content}

Extract only the facts relevant to answering the user's query."""
    
    input = ChatInput(
        system_prompt=EXTRACTOR_SYSTEM_PROMPT,
        messages=[Message(role="user", content=user_message)],
    )
    
    result, output = await client.chat_async_structured(
        input=input,
        response_model=ExtractedContent,
        config=config,
    )
    
    return result, output


async def extract_multiple(
    client: OAI,
    contents: list[str],
    query: str,
) -> list[ExtractedContent]:
    """
    Extract relevant facts from multiple scrubbed contents concurrently.
    
    Args:
        client: OAI client instance
        contents: List of scrubbed markdown contents
        query: The user's original query
    
    Returns:
        List of ExtractedContent results
    """
    import asyncio
    
    tasks = [extract(client, content, query) for content in contents]
    results = await asyncio.gather(*tasks)
    
    return [result for result, _ in results]


def format_extracted_content(extractions: list[ExtractedContent]) -> str:
    """
    Format multiple extracted contents into a single markdown string.
    
    Args:
        extractions: List of ExtractedContent from multiple sources
    
    Returns:
        Combined markdown string with all relevant facts
    """
    all_facts = []
    
    for i, extraction in enumerate(extractions, 1):
        if extraction.relevant and extraction.facts:
            all_facts.extend(extraction.facts)
    
    if not all_facts:
        return ""
    
    # Format as bullet points
    return "\n".join(f"- {fact}" for fact in all_facts)


def get_extraction_stats(
    original_contents: list[str],
    extractions: list[ExtractedContent],
) -> dict:
    """
    Get statistics about the extraction process.
    
    Args:
        original_contents: Original scrubbed contents
        extractions: Extracted contents
    
    Returns:
        Dictionary with stats
    """
    original_chars = sum(len(c) for c in original_contents)
    
    formatted = format_extracted_content(extractions)
    extracted_chars = len(formatted)
    
    relevant_count = sum(1 for e in extractions if e.relevant)
    total_facts = sum(len(e.facts) for e in extractions)
    
    reduction = original_chars - extracted_chars
    reduction_pct = (reduction / original_chars * 100) if original_chars > 0 else 0
    
    return {
        "original_chars": original_chars,
        "extracted_chars": extracted_chars,
        "reduction_chars": reduction,
        "reduction_percent": round(reduction_pct, 1),
        "sources_processed": len(extractions),
        "sources_relevant": relevant_count,
        "total_facts": total_facts,
    }
