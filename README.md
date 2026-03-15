# Search Agent

A web search-augmented chatbot with standard and deep research modes. Uses real-time web search (Tavily) to ground LLM responses in current information. Built with FastAPI, React, OpenAI, and Tavily.

## Architecture

**Frontend** — React + TypeScript + Vite. Connects to the backend via SSE for streaming status updates and responses.

**Backend** — FastAPI app that runs a search pipeline. Each request flows through: the orchestrator (query generation), Tavily (web search), scrubber (content cleaning), extractor (fact extraction), and summarizer (answer synthesis). All LLM calls use OpenAI; search uses Tavily.

**Pipeline** — Orchestrator turns the user message into search queries. Searches run in parallel. The scrubber strips boilerplate from raw markdown. The extractor runs one LLM call per source (in parallel) to pull relevant facts. The summarizer synthesizes a final answer from those facts. In deep research mode, the summarizer can request another round of search if information is missing.

**External APIs** — OpenAI for all LLM calls (orchestrator, extractor, summarizer). Tavily for web search.

## Modes

### Standard Search
Single-pass search pipeline:
1. **Orchestrator** generates up to 3 search queries
2. **Search** executes queries in parallel via Tavily
3. **Scrubber** cleans HTML/markdown boilerplate
4. **Extractor** pulls relevant facts (parallel LLM calls)
5. **Summarizer** synthesizes final answer

### Deep Research
Iterative search with up to 3 research loops:
1. Same as standard, but with up to 5 queries per iteration
2. Summarizer evaluates if more research is needed
3. If gaps remain, loops back with accumulated context
4. Continues until complete or max iterations reached

## Setup

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/)

### Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | Yes |
| `TAVILY_API_KEY` | Tavily API key for web search | Yes |

### Run with Docker

```bash
# Build and run
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`

The backend uses a volume mount and runs with `--reload`, so code changes are picked up automatically.

## API Reference

### POST /chat

Process a chat request with streaming status updates. **Rate limited:** 10 requests per minute per client IP (see [Rate limiting](#rate-limiting)).

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "What is quantum computing?"}
  ],
  "deep_research": false
}
```

**Response (SSE stream):**
```
data: {"status": "Planning search..."}
data: {"status": "Searching the web..."}
data: {"status": "Analyzing sources..."}
data: {"status": "Generating response..."}
data: {"done": true, "messages": [...], "iterations": 1, "total_ms": 15234, "sources": [...]}
```

### Rate limiting

The `/chat` endpoint is rate limited with [SlowAPI](https://github.com/laurentS/slowapi): **10 requests per minute per client IP**. When the limit is exceeded, the API returns `429 Too Many Requests` with a JSON body:

```json
{
  "error": true,
  "code": "rate_limit",
  "message": "Too many requests. Please try again later.",
  "retry_after": 60
}
```

The `Retry-After` header is also set (in seconds). The frontend surfaces this as the same user-facing message used for upstream (e.g. OpenAI) rate limits. `/health` is not rate limited.

### GET /health

Health check endpoint.

**Response:**
```json
{"status": "ok"}
```

## Project Structure

```
├── backend/
│   ├── app.py              # FastAPI application & endpoints
│   ├── oai/                # OpenAI API wrapper
│   │   ├── client.py       # OAI client class
│   │   ├── config.py       # Model configs (gpt-5-nano, gpt-5-mini, etc.)
│   │   ├── async_chat.py   # Async chat completions
│   │   └── structured.py   # Structured output parsing
│   ├── tvly/               # Tavily API wrapper
│   │   ├── client.py       # Tavily client class
│   │   ├── config.py       # Search configs
│   │   └── search.py       # Search operations
│   ├── workflow/           # Pipeline orchestration
│   │   ├── main.py         # Pipeline entry points
│   │   ├── orchestrator.py # Query generation
│   │   ├── search.py       # Search execution
│   │   ├── scrubber.py     # Content cleaning
│   │   ├── extractor.py    # Fact extraction
│   │   └── summarizer.py   # Answer synthesis
│   └── tests/              # Test suite
│       ├── test_pipeline.py
│       ├── test_performance.py
│       └── ...
├── frontend/
│   ├── src/
│   │   ├── App.tsx         # Main app component
│   │   ├── components/     # React components
│   │   ├── hooks/          # Custom hooks (useChat)
│   │   └── api/            # API client
│   └── ...
└── README.md
```

## Testing

```bash
cd backend

# Run all tests
uv run pytest -v

# Run specific test file
uv run pytest tests/test_pipeline.py -v -s

# Run performance benchmarks
uv run pytest tests/test_performance.py -v -s
```

## Development

### Linting

```bash
# Backend
cd backend
uv run ruff check .
uv run ruff format .

# Frontend
cd frontend
npm run lint
```

### Adding New Models

Edit `backend/oai/config.py` to add new model options:

```python
ModelName = Literal[
    "gpt-5.4",
    "gpt-5-mini",
    "gpt-5-nano",
    # Add new models here
]
```
