FROM python:3.13-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Expose port
EXPOSE 8000

# Run with reload for development
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
