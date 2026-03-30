"""FastAPI application factory for AquaShrimp.

All 3 tasks are served from a single app under /task/1, /task/2, /task/3.
The legacy root endpoints (/reset, /step, /grade, /state) still work and
are routed to the task set by AQUASHRIMP_TASK env var (default 1).
"""
import os
from fastapi import FastAPI
from aquashrimp.server.router import make_router, router as default_router

TASK_NAMES = {
    1: "NurseryPond (Easy)",
    2: "SemiIntensiveFarm (Medium)",
    3: "CommercialGrowOut (Hard)",
}

app = FastAPI(
    title="AquaShrimp OpenEnv",
    description="Shrimp aquaculture operations environment — all 3 tasks in one Space",
    version="1.0.0",
)

# Legacy root endpoints (backward compat — reads AQUASHRIMP_TASK, default task 1)
app.include_router(default_router)

# All 3 tasks mounted at /task/{id}
for _tid in [1, 2, 3]:
    app.include_router(
        make_router(_tid),
        prefix=f"/task/{_tid}",
        tags=[TASK_NAMES[_tid]],
    )


@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "openenv-aquashrimp",
        "version": "1.0.0",
        "species": "Litopenaeus vannamei (Whiteleg shrimp)",
        "reward_range": [-1.0, 1.0],
        "tasks": {
            str(tid): {
                "name": TASK_NAMES[tid],
                "reset":  f"/task/{tid}/reset",
                "step":   f"/task/{tid}/step",
                "grade":  f"/task/{tid}/grade",
                "state":  f"/task/{tid}/state",
                "health": f"/task/{tid}/health",
            }
            for tid in [1, 2, 3]
        },
        "docs": "/docs",
    }