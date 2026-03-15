from pydantic import BaseModel, Field

from oai import OAI, ChatInput, ChatOutput, Message, OAIConfig


class SummarizerResponse(BaseModel):
    """Structured output from the summarizer agent."""

    needs_more_research: bool = Field(
        description="True if more research is needed, False if the answer is complete"
    )
    content: str = Field(
        description="Either the final answer to the user, or feedback for the orchestrator about what additional information is needed"
    )


MAX_ITERATIONS = 3


def _get_system_prompt(deep_research: bool, can_continue: bool) -> str:
    base_prompt = """You are an expert research synthesizer. Your task is to analyze the provided search results and generate a comprehensive, accurate answer to the user's question.

Guidelines:
1. Synthesize information from multiple sources to provide a complete answer
2. Cite specific facts and data points from the search results
3. Be objective and present multiple perspectives when relevant
4. If information conflicts between sources, acknowledge the discrepancy
5. Structure your response clearly with appropriate formatting"""

    if deep_research and can_continue:
        return (
            base_prompt
            + """

DEEP RESEARCH MODE: You have the ability to request additional research if the current search results are insufficient to fully answer the user's question.

If you determine that:
- Critical information is missing
- The search results don't adequately cover important aspects of the question
- More specific or up-to-date information would significantly improve the answer

Then set needs_more_research to true and provide specific guidance in your content about what additional information should be searched for. Be specific about the knowledge gaps.

If the search results are sufficient to provide a comprehensive answer, set needs_more_research to false and provide the final answer."""
        )
    else:
        return (
            base_prompt
            + """

Provide the best possible answer based on the available search results. Set needs_more_research to false."""
        )


def _format_summarizer_input(
    messages: list[Message],
    search_context: str,
    num_iterations: int,
) -> str:
    """Format the input for the summarizer agent."""
    parts = []

    parts.append("=== USER QUESTION ===")
    for msg in messages:
        if msg.role == "user":
            parts.append(msg.content)

    parts.append("\n=== SEARCH RESULTS ===")
    parts.append(search_context)

    if num_iterations > 1:
        parts.append("\n=== RESEARCH ITERATION ===")
        parts.append(f"This is iteration {num_iterations} of the research process.")

    parts.append("\n=== TASK ===")
    parts.append(
        "Analyze the search results and provide a comprehensive answer to the user's question."
    )

    return "\n".join(parts)


async def summarize(
    client: OAI,
    messages: list[Message],
    search_context: str,
    deep_research: bool = False,
    num_iterations: int = 1,
    max_iterations: int = MAX_ITERATIONS,
) -> tuple[SummarizerResponse, ChatOutput]:
    """
    Summarize search results and either provide final answer or request more research.

    Args:
        client: OAI client instance
        messages: Original conversation messages
        search_context: Combined markdown content from search results
        deep_research: Whether deep research mode is enabled
        num_iterations: Current iteration number (1-indexed)
        max_iterations: Maximum allowed iterations

    Returns:
        Tuple of (SummarizerResponse, ChatOutput)
    """
    can_continue = deep_research and num_iterations < max_iterations

    config = OAIConfig(model="gpt-5-nano", reasoning_effort="minimal")

    formatted_input = _format_summarizer_input(messages, search_context, num_iterations)

    input = ChatInput(
        system_prompt=_get_system_prompt(deep_research, can_continue),
        messages=[Message(role="user", content=formatted_input)],
        metadata={
            "deep_research": deep_research,
            "num_iterations": num_iterations,
            "max_iterations": max_iterations,
            "can_continue": can_continue,
        },
    )

    result, output = await client.chat_async_structured(
        input=input,
        response_model=SummarizerResponse,
        config=config,
    )

    if not can_continue and result.needs_more_research:
        result = SummarizerResponse(
            needs_more_research=False,
            content=result.content,
        )

    return result, output
