FROM python:3.11-slim-bookworm
WORKDIR /app

# Install system dependencies (curl needed for HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies via pip (pinned ranges, no uv required)
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "pydantic>=2.7.0" \
    "numpy>=1.26.0" \
    "openenv-core>=0.2.0" \
    "openai>=1.0.0"

# Copy application code
COPY aquashrimp/ ./aquashrimp/
COPY server/ ./server/
COPY openenv.yaml inference.py ./

# Non-root user
RUN useradd --create-home aquashrimp && chown -R aquashrimp:aquashrimp /app
USER aquashrimp

# WORKERS=1: environment is stateful; scale by running multiple containers
ENV AQUASHRIMP_TASK=1 PORT=7860 WORKERS=1

HEALTHCHECK --interval=30s --timeout=5s \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 7860
CMD uvicorn aquashrimp.server.app:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS}
