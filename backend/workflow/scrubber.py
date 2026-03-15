import re

# Pre-compiled regex patterns for performance
# These are compiled once at module load time

_NAV_PATTERNS = [
    re.compile(r"^\s*\[Skip to .*?\].*$", re.MULTILINE),
    re.compile(r"^\s*\[Jump to .*?\].*$", re.MULTILINE),
    re.compile(r"^\s*Navigation\s*$", re.MULTILINE),
    re.compile(r"^\s*Menu\s*$", re.MULTILINE),
    re.compile(r"^\s*Toggle navigation\s*$", re.MULTILINE),
    re.compile(r"^\s*Search\s*$", re.MULTILINE),
    re.compile(r"^\s*Sign [Ii]n\s*$", re.MULTILINE),
    re.compile(r"^\s*Sign [Uu]p\s*$", re.MULTILINE),
    re.compile(r"^\s*Log [Ii]n\s*$", re.MULTILINE),
    re.compile(r"^\s*Log [Oo]ut\s*$", re.MULTILINE),
    re.compile(r"^\s*Subscribe\s*$", re.MULTILINE),
    re.compile(r"^\s*Newsletter\s*$", re.MULTILINE),
]

_SOCIAL_PATTERNS = [
    re.compile(r"^\s*Share\s*(this|on|via)?.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Tweet\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Follow us.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Like us.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(
        r"^\s*\[?(Facebook|Twitter|LinkedIn|Instagram|YouTube|Pinterest|Reddit|WhatsApp|Telegram|Email)\]?\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(
        r"^\s*Share on (Facebook|Twitter|LinkedIn|WhatsApp).*$", re.MULTILINE | re.IGNORECASE
    ),
    re.compile(r"^\s*Connect with us.*$", re.MULTILINE | re.IGNORECASE),
]

_COOKIE_PATTERNS = [
    re.compile(r"we use cookies.*?(\.|$)", re.MULTILINE | re.IGNORECASE),
    re.compile(r"this (site|website) uses cookies.*?(\.|$)", re.MULTILINE | re.IGNORECASE),
    re.compile(r"by (continuing|using).*?cookies.*?(\.|$)", re.MULTILINE | re.IGNORECASE),
    re.compile(r"accept (all )?cookies?", re.MULTILINE | re.IGNORECASE),
    re.compile(r"cookie (policy|settings|preferences)", re.MULTILINE | re.IGNORECASE),
    re.compile(r"privacy (policy|notice|settings)", re.MULTILINE | re.IGNORECASE),
    re.compile(r"manage (your )?preferences", re.MULTILINE | re.IGNORECASE),
]

_FOOTER_PATTERNS = [
    re.compile(r"^\s*©\s*\d{4}.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Copyright\s*©?.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*All [Rr]ights [Rr]eserved.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Terms (of|and) (Service|Use|Conditions).*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Privacy Policy.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Contact [Uu]s.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*About [Uu]s.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Careers.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Advertise.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Help Center.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*FAQ\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Support\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Sitemap\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Accessibility\s*$", re.MULTILINE | re.IGNORECASE),
]

_FOOTER_LINKS_PATTERN = re.compile(
    r"^.*\|\s*(Terms|Privacy|Contact|About|Careers|Help|FAQ|Support|Sitemap|Accessibility).*$",
    re.MULTILINE | re.IGNORECASE,
)

_RELATED_PATTERNS = [
    re.compile(
        r"^#+\s*(Related|Recommended|Popular|Trending|More|You (might|may) also (like|enjoy)|See also|Further reading).*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(Related|Recommended) (articles?|posts?|stories|content|links?).*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(r"^\s*More from.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Also read:?.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Read (more|next|also):?.*$", re.MULTILINE | re.IGNORECASE),
]

_AD_PATTERNS = [
    re.compile(r"^\s*\[?Advertisement\]?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*\[?Sponsored\]?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*\[?Ad\]?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Promoted\s*$", re.MULTILINE | re.IGNORECASE),
]

_BREADCRUMB_PATTERN = re.compile(r"^\s*(\w+\s*[>›»]\s*)+\w+\s*$", re.MULTILINE)
_MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\([^\)]+\)")
_IMAGE_PLACEHOLDER_PATTERN = re.compile(r"^\s*\[Image:.*?\]\s*$", re.MULTILINE)
_PHOTO_PLACEHOLDER_PATTERN = re.compile(r"^\s*\[Photo:.*?\]\s*$", re.MULTILINE)
_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_EMPTY_HEADER_PATTERN = re.compile(r"^#+\s*$", re.MULTILINE)
_HORIZONTAL_RULE_PATTERN = re.compile(r"^\s*[-*_]{3,}\s*$", re.MULTILINE)
_MULTIPLE_NEWLINES_PATTERN = re.compile(r"\n{3,}")
_WHITESPACE_LINES_PATTERN = re.compile(r"^\s+$", re.MULTILINE)


def scrub_markdown(content: str, max_chars: int | None = 6000) -> str:
    """
    Clean and compress raw markdown content by removing boilerplate and noise.

    Args:
        content: Raw markdown content from web scraping
        max_chars: Maximum characters to keep (None for no limit)

    Returns:
        Cleaned markdown content
    """
    if not content:
        return ""

    text = content

    # Remove navigation patterns
    for pattern in _NAV_PATTERNS:
        text = pattern.sub("", text)

    # Remove social media and sharing patterns
    for pattern in _SOCIAL_PATTERNS:
        text = pattern.sub("", text)

    # Remove cookie/privacy notices
    for pattern in _COOKIE_PATTERNS:
        text = pattern.sub("", text)

    # Remove footer patterns
    for pattern in _FOOTER_PATTERNS:
        text = pattern.sub("", text)

    # Remove footer link lines (e.g., "Privacy Policy | Terms of Service | Contact Us")
    text = _FOOTER_LINKS_PATTERN.sub("", text)

    # Remove "Related articles" / "You might also like" sections
    for pattern in _RELATED_PATTERNS:
        text = pattern.sub("", text)

    # Remove advertisement markers
    for pattern in _AD_PATTERNS:
        text = pattern.sub("", text)

    # Remove breadcrumb patterns (e.g., "Home > Category > Article")
    text = _BREADCRUMB_PATTERN.sub("", text)

    # Remove image alt text placeholders that don't add value
    text = _MARKDOWN_IMAGE_PATTERN.sub("", text)
    text = _IMAGE_PLACEHOLDER_PATTERN.sub("", text)
    text = _PHOTO_PLACEHOLDER_PATTERN.sub("", text)

    # Remove excessive link formatting but keep text
    # [Link text](url) -> Link text
    text = _MARKDOWN_LINK_PATTERN.sub(r"\1", text)

    # Remove empty markdown headers
    text = _EMPTY_HEADER_PATTERN.sub("", text)

    # Remove horizontal rules
    text = _HORIZONTAL_RULE_PATTERN.sub("", text)

    # Collapse multiple newlines into maximum of 2
    text = _MULTIPLE_NEWLINES_PATTERN.sub("\n\n", text)

    # Remove lines that are just whitespace
    text = _WHITESPACE_LINES_PATTERN.sub("", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    # Truncate if needed
    if max_chars and len(text) > max_chars:
        # Try to cut at a paragraph break
        truncated = text[:max_chars]
        last_para = truncated.rfind("\n\n")
        if last_para > max_chars * 0.7:  # If we can keep at least 70%
            text = truncated[:last_para].strip()
        else:
            # Cut at last sentence
            last_sentence = max(
                truncated.rfind(". "),
                truncated.rfind(".\n"),
                truncated.rfind("? "),
                truncated.rfind("?\n"),
                truncated.rfind("! "),
                truncated.rfind("!\n"),
            )
            if last_sentence > max_chars * 0.7:
                text = truncated[: last_sentence + 1].strip()
            else:
                text = truncated.strip() + "..."

    return text


def scrub_multiple(contents: list[str], max_chars_per_source: int | None = 6000) -> list[str]:
    """
    Scrub multiple markdown contents.

    Args:
        contents: List of raw markdown contents
        max_chars_per_source: Maximum characters per source

    Returns:
        List of cleaned markdown contents
    """
    return [scrub_markdown(c, max_chars_per_source) for c in contents]


def scrub_multiple_parallel(
    contents: list[str], max_chars_per_source: int | None = 6000, max_workers: int = 4
) -> list[str]:
    """
    Scrub multiple markdown contents in parallel using threads.

    Since scrubbing is CPU-bound (regex operations), we use a ThreadPoolExecutor
    to parallelize across multiple cores.

    Args:
        contents: List of raw markdown contents
        max_chars_per_source: Maximum characters per source
        max_workers: Maximum number of worker threads

    Returns:
        List of cleaned markdown contents (preserves order)
    """
    import concurrent.futures
    from functools import partial

    if len(contents) <= 1:
        return [scrub_markdown(c, max_chars_per_source) for c in contents]

    scrub_fn = partial(scrub_markdown, max_chars=max_chars_per_source)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(scrub_fn, contents))


def get_scrub_stats(original: str, scrubbed: str) -> dict:
    """
    Get statistics about the scrubbing process.

    Args:
        original: Original content
        scrubbed: Scrubbed content

    Returns:
        Dictionary with stats
    """
    original_chars = len(original)
    scrubbed_chars = len(scrubbed)
    reduction = original_chars - scrubbed_chars
    reduction_pct = (reduction / original_chars * 100) if original_chars > 0 else 0

    return {
        "original_chars": original_chars,
        "scrubbed_chars": scrubbed_chars,
        "reduction_chars": reduction,
        "reduction_percent": round(reduction_pct, 1),
    }
