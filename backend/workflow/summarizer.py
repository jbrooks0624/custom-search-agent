from collections.abc import AsyncIterator

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
    tone = """Write as the assistant. Do NOT say "the search results you supplied", "the results you provided", "I'll present a synthesis", or "I can provide a comparison". Use "I found", "research shows", or "sources indicate" when citing. Never imply the user gave you the data.

Guidelines: Synthesize the gathered sources; cite specific facts; be objective; if sources conflict, state briefly and move on; structure clearly."""

    if deep_research and can_continue:
        return (
            """You are the research assistant. You ran the web search and gathered the sources below. The user did not supply them—you did.

You must choose one: (A) give the final answer to the user now, or (B) request another round of search because coverage is incomplete. See DEEP RESEARCH MODE below.

"""
            + tone
            + """

DEEP RESEARCH MODE: Set needs_more_research to true to request another round when:
- The user asked for multiple specific things and the results do not clearly cover all of them
- Critical data is missing (numbers, dates, comparisons) that the user explicitly asked for
- The question has several distinct sub-questions and at least one is poorly covered
- Sources conflict or are thin and one more round could clarify

When requesting more research, in your content state exactly what is missing (e.g. "Need specific GDP figures for 2023", "Need more on criticism from the left") so the next search can target those gaps.

Only set needs_more_research to false when you have solid, specific coverage of what the user asked for. When in doubt, prefer one more round of research for complex or multi-part questions."""
        )
    else:
        return (
            """You are the research assistant. You ran the web search and gathered the sources below. The user did not supply them—you did. Your reply is the final answer shown to the user.

"""
            + tone
            + """

Provide the best possible answer. Set needs_more_research to false."""
        )


def _format_summarizer_input(
    messages: list[Message],
    search_context: str,
    num_iterations: int,
    deep_research: bool = False,
) -> str:
    """Format the input for the summarizer agent."""
    parts = []

    # Include full conversation
    parts.append("=== CONVERSATION ===")
    for msg in messages:
        role = "User" if msg.role == "user" else "Assistant"
        parts.append(f"{role}: {msg.content}")

    parts.append("\n=== SOURCES (gathered by your search; answer using these) ===")
    parts.append(search_context)

    if num_iterations > 1:
        parts.append("\n=== RESEARCH ITERATION ===")
        parts.append(f"This is iteration {num_iterations} of the research process.")
    else:
        parts.append("\n=== RESEARCH ITERATION ===")
        if deep_research:
            parts.append(
                "This is round 1 of up to 3. For questions that ask for multiple specific data points or comparisons (e.g. GDP, unemployment, inflation, debt, criticisms, defenses), you MUST set needs_more_research to true so another search round can run. Do not give the final answer on the first round—request more research."
            )
        else:
            parts.append("This is the first round of search. If the question is complex or multi-part, consider requesting more research to fill gaps.")

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

    # Use slightly higher reasoning in deep research so the model actually evaluates gaps
    config = OAIConfig(
        model="gpt-5-nano",
        reasoning_effort="low" if (deep_research and can_continue) else "minimal",
    )

    formatted_input = _format_summarizer_input(
        messages, search_context, num_iterations, deep_research=deep_research
    )

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


async def summarize_stream(
    client: OAI,
    messages: list[Message],
    search_context: str,
    num_iterations: int = 1,
) -> AsyncIterator[str]:
    """
    Stream the final summarizer answer (no structured output).

    Use when the response is the final user-facing answer so tokens can be
    streamed to the client. Same prompt as non-streaming "final" case.
    """
    config = OAIConfig(model="gpt-5-nano", reasoning_effort="minimal")
    formatted_input = _format_summarizer_input(
        messages, search_context, num_iterations, deep_research=False
    )
    system_prompt = _get_system_prompt(deep_research=False, can_continue=False)

    input = ChatInput(
        system_prompt=system_prompt,
        messages=[Message(role="user", content=formatted_input)],
        metadata={"stream": True, "num_iterations": num_iterations},
    )

    async for delta in client.chat_async_stream(input=input, config=config):
        yield delta
