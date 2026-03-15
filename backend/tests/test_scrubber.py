import pytest

from tvly import SearchInput, Tavily, TavilyConfig
from workflow import scrub_markdown

# Sample raw markdown with typical web page noise
SAMPLE_RAW_MARKDOWN = """
[Skip to main content]

Navigation

Menu

Sign In
Sign Up

Home > Technology > AI > Article

# Understanding Large Language Models

Share on Facebook
Share on Twitter
Share on LinkedIn

Tweet

We use cookies to improve your experience. By continuing to use this site, you accept our cookie policy.

Accept all cookies

Cookie settings

---

Large language models (LLMs) are a type of artificial intelligence that can understand and generate human-like text. They are trained on vast amounts of text data and can perform a wide variety of tasks.

## How LLMs Work

LLMs use a transformer architecture that allows them to process text in parallel and capture long-range dependencies. The key innovation is the attention mechanism, which allows the model to focus on relevant parts of the input.

### Key Components

1. **Tokenization**: Breaking text into smaller units
2. **Embeddings**: Converting tokens into numerical vectors
3. **Attention layers**: Learning relationships between tokens
4. **Feed-forward networks**: Processing the representations

## Applications

LLMs are used in many applications including:

- Chatbots and virtual assistants
- Code generation and completion
- Translation and summarization
- Content creation and editing

![AI Image](https://example.com/image.png)

[Image: A diagram showing transformer architecture]

## Limitations

Despite their capabilities, LLMs have several limitations:

- They can hallucinate incorrect information
- They have a knowledge cutoff date
- They may exhibit biases present in training data

---

## Related Articles

- Understanding GPT-4
- The Future of AI
- Machine Learning Basics

### You might also like

More from Technology

Read next: AI News Roundup

---

Advertisement

Sponsored

---

© 2026 TechBlog Inc. All Rights Reserved.

Privacy Policy | Terms of Service | Contact Us

About Us | Careers | Advertise

Help Center | FAQ | Support

Sitemap | Accessibility

Follow us on Twitter
Like us on Facebook
Connect with us on LinkedIn

Newsletter

Subscribe
"""


def test_scrub_removes_navigation():
    """Test that navigation elements are removed."""
    scrubbed = scrub_markdown(SAMPLE_RAW_MARKDOWN, max_chars=None)

    assert "[Skip to main content]" not in scrubbed
    assert "Navigation" not in scrubbed or "navigation" not in scrubbed.lower()
    assert "Sign In" not in scrubbed
    assert "Sign Up" not in scrubbed


def test_scrub_removes_social():
    """Test that social sharing elements are removed."""
    scrubbed = scrub_markdown(SAMPLE_RAW_MARKDOWN, max_chars=None)

    assert "Share on Facebook" not in scrubbed
    assert "Share on Twitter" not in scrubbed
    assert "Tweet" not in scrubbed
    assert "Follow us" not in scrubbed


def test_scrub_removes_cookies():
    """Test that cookie notices are removed."""
    scrubbed = scrub_markdown(SAMPLE_RAW_MARKDOWN, max_chars=None)

    assert "We use cookies" not in scrubbed
    assert "Accept all cookies" not in scrubbed
    assert "Cookie settings" not in scrubbed


def test_scrub_removes_footer():
    """Test that footer elements are removed."""
    scrubbed = scrub_markdown(SAMPLE_RAW_MARKDOWN, max_chars=None)

    assert "© 2026" not in scrubbed
    assert "All Rights Reserved" not in scrubbed
    assert "Privacy Policy" not in scrubbed
    assert "Terms of Service" not in scrubbed
    assert "Sitemap" not in scrubbed


def test_scrub_removes_related():
    """Test that related articles sections are removed."""
    scrubbed = scrub_markdown(SAMPLE_RAW_MARKDOWN, max_chars=None)

    assert "Related Articles" not in scrubbed
    assert "You might also like" not in scrubbed
    assert "Read next:" not in scrubbed


def test_scrub_removes_ads():
    """Test that advertisement markers are removed."""
    scrubbed = scrub_markdown(SAMPLE_RAW_MARKDOWN, max_chars=None)

    assert "Advertisement" not in scrubbed
    assert "Sponsored" not in scrubbed


def test_scrub_preserves_content():
    """Test that actual content is preserved."""
    scrubbed = scrub_markdown(SAMPLE_RAW_MARKDOWN, max_chars=None)

    assert "Understanding Large Language Models" in scrubbed
    assert "LLMs use a transformer architecture" in scrubbed
    assert "Tokenization" in scrubbed
    assert "Chatbots and virtual assistants" in scrubbed
    assert "They can hallucinate incorrect information" in scrubbed


def test_scrub_removes_images():
    """Test that image markdown is removed."""
    scrubbed = scrub_markdown(SAMPLE_RAW_MARKDOWN, max_chars=None)

    assert "![AI Image]" not in scrubbed
    assert "[Image:" not in scrubbed


def test_scrub_simplifies_links():
    """Test that links are simplified to just text."""
    content = "Check out [this article](https://example.com/article) for more info."
    scrubbed = scrub_markdown(content, max_chars=None)

    assert "this article" in scrubbed
    assert "https://example.com" not in scrubbed
    assert "[" not in scrubbed
    assert "](" not in scrubbed


def test_scrub_truncation():
    """Test that long content is truncated properly."""
    long_content = "This is a sentence. " * 500  # ~10,000 chars
    scrubbed = scrub_markdown(long_content, max_chars=1000)

    assert len(scrubbed) <= 1000
    # Should end at a sentence boundary or with ellipsis
    assert scrubbed.endswith(".") or scrubbed.endswith("...")


@pytest.mark.asyncio
async def test_scrub_real_search_results():
    """Test scrubbing actual search results from Tavily."""
    client = Tavily(
        config=TavilyConfig(
            max_results=3,
            include_raw_content="markdown",
        )
    )

    input = SearchInput(query="What is machine learning?")
    output = await client.search_async(input)

    print("\n--- Real Search Results Scrubbing ---")
    print(f"Query: {input.query}")
    print(f"Results: {len(output.results)}")

    for i, result in enumerate(output.results, 1):
        if result.raw_content:
            scrubbed = scrub_markdown(result.raw_content, max_chars=4000)
            print(f"\n{i}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Original: {len(result.raw_content)} chars")
            print(f"   Scrubbed: {len(scrubbed)} chars")
            print(f"   Preview: {scrubbed[:200]}...")
