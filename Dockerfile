# lltop: streaming proxy in front of Ollama with a live dashboard.
#
#   docker build -t lltop .
#   docker run --rm -p 127.0.0.1:11435:11435 \
#     --add-host=host.docker.internal:host-gateway lltop
#
# LLTOP_UPSTREAM is where Ollama is, as seen from inside the container.

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first (cached until uv.lock changes), then the project itself.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.12-slim
RUN useradd --system --uid 10001 lltop
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    LLTOP_HOST=0.0.0.0 \
    LLTOP_PORT=11435 \
    LLTOP_UPSTREAM=http://host.docker.internal:11434
USER lltop
EXPOSE 11435
HEALTHCHECK --interval=30s --timeout=5s CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"LLTOP_PORT\"]}/lltop/api/state', timeout=4)"]
CMD ["lltop"]
