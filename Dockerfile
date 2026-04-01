FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
COPY aquashrimp/__init__.py ./aquashrimp/
RUN uv pip install --system --no-cache \
    "fastapi>=0.115.0" "uvicorn[standard]>=0.30.0" \
    "pydantic>=2.7.0" "numpy>=1.26.0" "openenv-core>=0.2.0" "openai>=1.0.0"

FROM python:3.11-slim AS runtime
# curl needed for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home aquashrimp
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY aquashrimp/ ./aquashrimp/
COPY openenv.yaml ./
COPY inference.py ./
RUN chown -R aquashrimp:aquashrimp /app
USER aquashrimp
# WORKERS=1: environment is stateful (one session per worker); scale by running multiple containers
ENV AQUASHRIMP_TASK=1 PORT=7860 WORKERS=1
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:${PORT}/health || exit 1
EXPOSE 7860
CMD uvicorn aquashrimp.server.app:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS}
