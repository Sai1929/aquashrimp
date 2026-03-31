"""Root-level server/app.py — required by OpenEnv checker.

Re-exports the FastAPI app from aquashrimp.server.app.
Run with: uvicorn server.app:app --host 0.0.0.0 --port 7860
"""
from aquashrimp.server.app import app  # noqa: F401

__all__ = ["app"]
