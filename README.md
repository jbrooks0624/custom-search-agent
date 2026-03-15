# Search Agent

A web search-augmented chatbot with standard and deep research modes. Uses real-time web search (Tavily) to ground LLM responses in current information. Built with FastAPI, React, OpenAI, and Tavily.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                  Frontend                                    │
│                         React + TypeScript + Vite                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ WelcomeScreen│  │ MessageList │  │  ChatInput  │  │      useChat       │ │
│  │             │  │             │  │             │  │   (SSE streaming)   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ SSE /chat
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                  Backend                                     │
│                              FastAPI + Python                                │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                           Pipeline (workflow/)                         │  │
│  │                                                                        │  │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐            │  │
│  │   │ Orchestrator │───▶│    Search    │───▶│   Scrubber   │            │  │
│  │   │  (gpt-5-nano)│    │   (Tavily)   │    │   (regex)    │            │  │
│  │   └──────────────┘    └──────────────┘    └──────────────┘            │  │
│  │          │                   │                   │                     │  │
│  │          │ generates         │ parallel          │ cleans              │  │
│  │          │ queries           │ web search        │ HTML/markdown       │  │
│  │          ▼                   ▼                   ▼                     │  │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐            │  │
│  │   │  Extractor   │◀───│   Results    │◀───│   Content    │            │  │
│  │   │  (gpt-5-nano)│    │              │    │              │            │  │
│  │   └──────────────┘    └──────────────┘    └──────────────┘            │  │
│  │          │                                                             │  │
│  │          │ extracts relevant facts                                     │  │
│  │          ▼                                                             │  │
│  │   ┌──────────────┐         ┌──────────────────────────────────────┐   │  │
│  │   │  Summarizer  │────────▶│            Final Response            │   │  │
│  │   │  (gpt-5-nano)│         │                                      │   │  │
│  │   └──────────────┘         └──────────────────────────────────────┘   │  │
│  │          │                                                             │  │
│  │          │ (deep research only)                                        │  │
│  │          ▼                                                             │  │
│  │   ┌──────────────┐                                                     │  │
│  │   │ Loop if more │                                                     │  │
│  │   │   research   │──────▶ Back to Orchestrator (up to 3 iterations)   │  │
│  │   │    needed    │                                                     │  │
│  │   └──────────────┘                                                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌─────────────────────┐              ┌─────────────────────┐               │
│  │      oai/           │              │       tvly/         │               │
│  │  OpenAI Wrapper     │              │   Tavily Wrapper    │               │
│  │  - Async chat       │              │   - Web search      │               │
│  │  - Structured output│              │   - Raw content     │               │
│  └─────────────────────┘              └─────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
                │                                    │
                ▼                                    ▼
        ┌──────────────┐                    ┌──────────────┐
        │  OpenAI API  │                    │  Tavily API  │
        │   (GPT-5)    │                    │ (Web Search) │
        └──────────────┘                    └──────────────┘
```

## Pipeline Modes

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
- Python 3.10+ (3.13 recommended)
- Node.js 18+
- [uv](https://github.com/astral-sh/uv) (Python package manager)

### Environment Variables

Create `backend/.env`:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | Yes |
| `TAVILY_API_KEY` | Tavily API key for web search | Yes |

### Backend Setup

```bash
cd backend

# Install dependencies
uv sync

# Run development server
uv run uvicorn app:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The app will be available at `http://localhost:5173`.

## API Reference

### POST /chat

Process a chat request with streaming status updates.

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
