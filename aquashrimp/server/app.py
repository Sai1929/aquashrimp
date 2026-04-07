"""FastAPI application factory for AquaShrimp.

All 3 tasks are served from a single app under /task/1, /task/2, /task/3.
The legacy root endpoints (/reset, /step, /grade, /state) still work and
are routed to the task set by AQUASHRIMP_TASK env var (default 1).
"""
import os
from fastapi import FastAPI, Request
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


_TASK_IDS = {
    1: "nursery_pond",
    2: "semi_intensive_farm",
    3: "commercial_grow_out",
}
_MAX_STEPS = {1: 30, 2: 60, 3: 90}


@app.get("/metadata", tags=["Info"])
async def metadata():
    """OpenEnv standard metadata endpoint — lists all tasks and their graders."""
    return {
        "name": "aquashrimp",
        "version": "1.0.0",
        "description": (
            "AquaShrimp: Precision shrimp aquaculture operations management. "
            "Agent manages Litopenaeus vannamei farms."
        ),
        "tasks": [
            {
                "id": _TASK_IDS[tid],
                "display_name": TASK_NAMES[tid],
                "max_steps": _MAX_STEPS[tid],
                "grader": {
                    "endpoint": f"GET /task/{tid}/grade",
                    "score_range": [0.0, 1.0],
                },
                "endpoints": {
                    "reset":  f"/task/{tid}/reset",
                    "step":   f"/task/{tid}/step",
                    "grade":  f"/task/{tid}/grade",
                    "state":  f"/task/{tid}/state",
                    "health": f"/task/{tid}/health",
                },
            }
            for tid in [1, 2, 3]
        ],
    }


@app.get("/schema", tags=["Info"])
async def schema():
    """OpenEnv standard schema endpoint — returns JSON schemas for action/observation/state."""
    return {
        "action": {
            "type": "object",
            "description": "Daily farm management action (Task 1 / NurseryPond default)",
            "properties": {
                "feed_kg":             {"type": "number",  "minimum": 0,   "maximum": 50},
                "feeding_frequency":   {"type": "integer", "minimum": 2,   "maximum": 6},
                "aeration_hours":      {"type": "number",  "minimum": 0,   "maximum": 24},
                "water_exchange_frac": {"type": "number",  "minimum": 0.0, "maximum": 0.15},
                "check_feeding_trays": {"type": "boolean"},
                "lime_application_kg": {"type": "number",  "minimum": 0,   "maximum": 20},
            },
            "required": [
                "feed_kg", "feeding_frequency", "aeration_hours",
                "water_exchange_frac", "check_feeding_trays", "lime_application_kg",
            ],
        },
        "observation": {
            "type": "object",
            "description": "Daily pond state observation",
            "properties": {
                "day":                       {"type": "integer"},
                "mean_weight_g":             {"type": "number"},
                "survival_rate":             {"type": "number"},
                "do_mg_l":                   {"type": "number"},
                "ph":                        {"type": "number"},
                "temperature_c":             {"type": "number"},
                "tan_mg_l":                  {"type": "number"},
                "feed_demand_estimate_kg":   {"type": "number"},
                "reward":                    {"type": "number"},
                "grade":                     {"type": "number"},
                "done":                      {"type": "boolean"},
            },
        },
        "state": {
            "type": "object",
            "description": "Episode state",
            "properties": {
                "day":       {"type": "integer"},
                "done":      {"type": "boolean"},
                "max_steps": {"type": "integer"},
            },
        },
    }


@app.post("/mcp", tags=["MCP"])
async def mcp(request: Request):
    """OpenEnv standard MCP endpoint — minimal JSON-RPC 2.0 response."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    return {
        "jsonrpc": "2.0",
        "result": {
            "capabilities": {"environment": True, "tasks": 3},
            "name": "aquashrimp",
        },
        "id": body.get("id") if isinstance(body, dict) else None,
    }


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
        "metadata": "/metadata",
    }