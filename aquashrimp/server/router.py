"""FastAPI router factory for AquaShrimp.

make_router(task_id) returns a self-contained router with its own env instance.
This allows all 3 tasks to be served from a single FastAPI app.

Module-level `router` is kept for backward compatibility (reads AQUASHRIMP_TASK env var).
"""
from __future__ import annotations
import os
from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


# ── Env factory ───────────────────────────────────────────────────────────────

def _create_env(task_id: int):
    if task_id == 1:
        from aquashrimp.tasks.nursery_pond import NurseryPondEnvironment
        return NurseryPondEnvironment()
    elif task_id == 2:
        from aquashrimp.tasks.semi_intensive_farm import SemiIntensiveFarmEnvironment
        return SemiIntensiveFarmEnvironment()
    elif task_id == 3:
        from aquashrimp.tasks.commercial_grow_out import CommercialGrowOutEnvironment
        return CommercialGrowOutEnvironment()
    else:
        raise ValueError(f"Invalid task_id={task_id}, must be 1, 2, or 3")


# ── Request models ─────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    seed: int = 42


class NurseryStepRequest(BaseModel):
    feed_kg: float = 5.0
    feeding_frequency: int = 4
    aeration_hours: float = 20.0
    water_exchange_frac: float = 0.05
    check_feeding_trays: bool = False
    lime_application_kg: float = 0.0


class SemiStepRequest(BaseModel):
    pond_feeds: list[dict] = []
    aeration_allocation: dict[str, float] = {}
    water_exchange: dict[str, float] = {}
    check_trays: dict[str, bool] = {}
    lime_per_pond: dict[str, float] = {}
    probiotic_ponds: list[int] = []
    antibiotic_ponds: list[int] = []
    partial_harvest: dict | None = None


class CommercialStepRequest(BaseModel):
    pond_feeds: list[dict] = []
    aeration_per_pond: dict[str, float] = {}
    water_exchange: dict[str, float] = {}
    check_trays: dict[str, bool] = {}
    pond_inspections: list[int] = []
    lime_per_pond: dict[str, float] = {}
    biosecurity_measure: bool = False
    treatments: list[dict] = []
    harvests: list[dict] = []
    disinfect_pond: list[int] = []
    regulatory_report: bool = False


def _obs_to_dict(obs) -> dict:
    """Convert observation dataclass to JSON-serializable dict."""
    import dataclasses
    from enum import Enum

    def _convert(v):
        if dataclasses.is_dataclass(v):
            return {k: _convert(val) for k, val in dataclasses.asdict(v).items()}
        elif isinstance(v, list):
            return [_convert(i) for i in v]
        elif isinstance(v, dict):
            return {str(k): _convert(val) for k, val in v.items()}
        elif isinstance(v, Enum):
            return v.value
        return v

    return _convert(obs)


def _build_action(task_id: int, request: dict):
    """Parse raw request dict into the correct action dataclass."""
    if task_id == 1:
        from aquashrimp.models.actions import NurseryPondAction
        return NurseryPondAction(**{k: request[k] for k in request if k in NurseryPondAction.__dataclass_fields__})

    elif task_id == 2:
        from aquashrimp.models.actions import SemiIntensiveFarmAction, PondFeedAction, PartialHarvestAction
        pond_feeds = [PondFeedAction(**pf) for pf in request.get("pond_feeds", [])]
        partial = None
        if request.get("partial_harvest"):
            partial = PartialHarvestAction(**request["partial_harvest"])
        return SemiIntensiveFarmAction(
            pond_feeds=pond_feeds,
            aeration_allocation={int(k): v for k, v in request.get("aeration_allocation", {}).items()},
            water_exchange={int(k): v for k, v in request.get("water_exchange", {}).items()},
            check_trays={int(k): v for k, v in request.get("check_trays", {}).items()},
            lime_per_pond={int(k): v for k, v in request.get("lime_per_pond", {}).items()},
            probiotic_ponds=request.get("probiotic_ponds", []),
            antibiotic_ponds=request.get("antibiotic_ponds", []),
            partial_harvest=partial,
        )

    elif task_id == 3:
        from aquashrimp.models.actions import (
            CommercialGrowOutAction, PondFeedAction,
            PondTreatmentAction, HarvestAction,
        )
        from aquashrimp.models.enums import TreatmentType, HarvestType
        pond_feeds = [PondFeedAction(**pf) for pf in request.get("pond_feeds", [])]
        treatments = [
            PondTreatmentAction(pond_id=t["pond_id"], treatment=TreatmentType(t["treatment"]))
            for t in request.get("treatments", [])
        ]
        harvests = [
            HarvestAction(
                pond_id=h["pond_id"],
                harvest_type=HarvestType(h["harvest_type"]),
                size_threshold_g=h.get("size_threshold_g"),
                fraction=h.get("fraction"),
            ) for h in request.get("harvests", [])
        ]
        return CommercialGrowOutAction(
            pond_feeds=pond_feeds,
            aeration_per_pond={int(k): v for k, v in request.get("aeration_per_pond", {}).items()},
            water_exchange={int(k): v for k, v in request.get("water_exchange", {}).items()},
            check_trays={int(k): v for k, v in request.get("check_trays", {}).items()},
            pond_inspections=request.get("pond_inspections", []),
            lime_per_pond={int(k): v for k, v in request.get("lime_per_pond", {}).items()},
            biosecurity_measure=request.get("biosecurity_measure", False),
            treatments=treatments,
            harvests=harvests,
            disinfect_pond=request.get("disinfect_pond", []),
            regulatory_report=request.get("regulatory_report", False),
        )

    raise ValueError(f"Unknown task_id={task_id}")


# ── Router factory ─────────────────────────────────────────────────────────────

def make_router(task_id: int) -> APIRouter:
    """Return a self-contained APIRouter for the given task_id (1, 2, or 3).

    Each router owns its own environment instance via closure.
    """
    r = APIRouter()
    _state: dict[str, Any] = {"env": None}

    def _get_env():
        if _state["env"] is None:
            _state["env"] = _create_env(task_id)
        return _state["env"]

    @r.get("/health")
    async def health():
        return {"status": "healthy", "task_id": task_id}

    @r.post("/reset")
    async def reset(request: ResetRequest = ResetRequest()):
        env = _get_env()
        env.seed = request.seed
        obs = env.reset()
        return _obs_to_dict(obs)

    @r.get("/state")
    async def get_state():
        env = _get_env()
        if env.state is None:
            raise HTTPException(status_code=400, detail="Call /reset first")
        import dataclasses
        from enum import Enum

        def _convert(v):
            if dataclasses.is_dataclass(v):
                return {k: _convert(val) for k, val in dataclasses.asdict(v).items()}
            elif isinstance(v, list):
                return [_convert(i) for i in v]
            elif isinstance(v, dict):
                return {str(k): _convert(val) for k, val in v.items()}
            elif isinstance(v, Enum):
                return v.value
            return v

        return _convert(env.state)

    _MAX_STEPS = {1: 30, 2: 60, 3: 90}

    @r.get("/grade")
    async def get_grade():
        env = _get_env()
        # Return default grade before reset so validators always get a 200
        if env.state is None:
            return {
                "grade": 0.0,
                "task_id": task_id,
                "episode_done": False,
                "steps": 0,
                "max_steps": _MAX_STEPS.get(task_id, 30),
            }
        return {
            "grade": env.grade,
            "task_id": task_id,
            "episode_done": env.state.done,
            "steps": env.state.day,
            "max_steps": env.state.max_steps,
        }

    @r.post("/step")
    async def step(request: dict):
        env = _get_env()
        if env.state is None:
            raise HTTPException(status_code=400, detail="Call /reset first")
        if env.state.done:
            raise HTTPException(status_code=400, detail="Episode done, call /reset")
        try:
            action = _build_action(task_id, request)
            obs = env.step(action)
            return _obs_to_dict(obs)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

    return r


# ── Backward-compatible default router (reads AQUASHRIMP_TASK env var) ─────────

_default_task_id: int = int(os.environ.get("AQUASHRIMP_TASK", "1"))
router = make_router(_default_task_id)