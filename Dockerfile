FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY aquashrimp/__init__.py ./aquashrimp/
RUN pip install --no-cache-dir --prefix=/install \
    "fastapi>=0.115.0" "uvicorn[standard]>=0.30.0" \
    "pydantic>=2.7.0" "numpy>=1.26.0"

FROM python:3.11-slim AS runtime
# curl needed for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home aquashrimp
WORKDIR /app
COPY --from=builder /install /usr/local
COPY aquashrimp/ ./aquashrimp/
COPY openenv.yaml ./
RUN chown -R aquashrimp:aquashrimp /app
USER aquashrimp
# WORKERS=1: environment is stateful (one session per worker); scale by running multiple containers
ENV AQUASHRIMP_TASK=1 PORT=7860 WORKERS=1
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:${PORT}/health || exit 1
EXPOSE 7860
CMD uvicorn aquashrimp.server.app:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS}
