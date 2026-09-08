FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache

COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY webex_byova_ws_adapter ./webex_byova_ws_adapter
RUN uv sync --frozen --no-dev

USER 65532:65532
EXPOSE 8086
ENTRYPOINT ["/app/.venv/bin/cisco-webex-byova-ws-adapter"]
