"""Root-level server/app.py — required by OpenEnv checker.

Provides a main() entry point that starts the AquaShrimp FastAPI server.
Run with: python server/app.py
Or via entry point: server (defined in [project.scripts])
"""
import os
import uvicorn
from aquashrimp.server.app import app  # noqa: F401


def main() -> None:
    """Start the AquaShrimp OpenEnv HTTP server."""
    port = int(os.environ.get("PORT", 7860))
    workers = int(os.environ.get("WORKERS", 1))
    uvicorn.run(
        "aquashrimp.server.app:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
    )


if __name__ == "__main__":
    main()
