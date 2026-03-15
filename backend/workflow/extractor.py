from pydantic import BaseModel, Field

from oai import OAI, ChatInput, ChatOutput, Message, OAIConfig


class ExtractedContent(BaseModel):
    """Structured output from the content extractor."""

    relevant: bool = Field(
        description="Whether the content contains information relevant to the query"
    )
    facts: list[str] = Field(
        default_factory=list,
        description="List of relevant facts, numbers, quotes, and key points extracted from the content",
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


def format_extracted_content(extractions: list[ExtractedContent]) -> str:
    """
    Format multiple extracted contents into a single markdown string.

    Args:
        extractions: List of ExtractedContent from multiple sources

    Returns:
        Combined markdown string with all relevant facts
    """
    all_facts = []

    for _i, extraction in enumerate(extractions, 1):
        if extraction.relevant and extraction.facts:
            all_facts.extend(extraction.facts)

    if not all_facts:
        return ""

    return "\n".join(f"- {fact}" for fact in all_facts)
