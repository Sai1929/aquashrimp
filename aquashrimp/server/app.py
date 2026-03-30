"""FastAPI application factory for AquaShrimp.

Task selected via AQUASHRIMP_TASK environment variable (1, 2, or 3).
Deploy 3 HuggingFace Spaces from one Docker image by setting AQUASHRIMP_TASK.
"""
import os
from fastapi import FastAPI
from aquashrimp.server.router import router

TASK_NAMES = {
    "1": "NurseryPond (Easy)",
    "2": "SemiIntensiveFarm (Medium)",
    "3": "CommercialGrowOut (Hard)",
}

task_id = os.environ.get("AQUASHRIMP_TASK", "1")
task_name = TASK_NAMES.get(task_id, "Unknown")

app = FastAPI(
    title="AquaShrimp OpenEnv",
    description=f"Shrimp aquaculture operations environment — Task: {task_name}",
    version="1.0.0",
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "name": "openenv-aquashrimp",
        "version": "1.0.0",
        "task": task_name,
        "task_id": int(task_id),
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "health": "GET /health",
        },
        "species": "Litopenaeus vannamei (Whiteleg shrimp)",
        "reward_range": [-1.0, 1.0],
    }
