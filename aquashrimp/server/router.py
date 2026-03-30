"""FastAPI router: POST /reset, POST /step, GET /state, GET /health."""
from __future__ import annotations
import os
from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# ── Global environment instance (per-worker state) ────────────────────────────
_env: Any = None
_task_id: int = int(os.environ.get("AQUASHRIMP_TASK", "1"))


def _get_env():
    global _env
    if _env is None:
        _env = _create_env(_task_id)
    return _env


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
        raise ValueError(f"Invalid AQUASHRIMP_TASK={task_id}, must be 1, 2, or 3")


# ── Request models ──────────────────────────────────────────────────────────
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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    return {"status": "ok", "task_id": _task_id}


@router.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    global _env
    env = _get_env()
    env.seed = request.seed
    obs = env.reset()
    return _obs_to_dict(obs)


@router.get("/state")
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


@router.get("/grade")
async def get_grade():
    """Return the current normalized episode grade [0.0, 1.0]."""
    env = _get_env()
    if env.state is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return {
        "grade": env.grade,
        "task_id": _task_id,
        "episode_done": env.state.done,
        "steps": env.state.day,
        "max_steps": env.state.max_steps,
    }


@router.post("/step")
async def step(request: dict):
    env = _get_env()
    if env.state is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    if env.state.done:
        raise HTTPException(status_code=400, detail="Episode done, call /reset")

    try:
        if _task_id == 1:
            from aquashrimp.models.actions import NurseryPondAction
            action = NurseryPondAction(**{k: request[k] for k in request if k in NurseryPondAction.__dataclass_fields__})
            obs = env.step(action)

        elif _task_id == 2:
            from aquashrimp.models.actions import SemiIntensiveFarmAction, PondFeedAction, PartialHarvestAction
            pond_feeds = [PondFeedAction(**pf) for pf in request.get("pond_feeds", [])]
            partial = None
            if request.get("partial_harvest"):
                partial = PartialHarvestAction(**request["partial_harvest"])
            action = SemiIntensiveFarmAction(
                pond_feeds=pond_feeds,
                aeration_allocation={int(k): v for k, v in request.get("aeration_allocation", {}).items()},
                water_exchange={int(k): v for k, v in request.get("water_exchange", {}).items()},
                check_trays={int(k): v for k, v in request.get("check_trays", {}).items()},
                lime_per_pond={int(k): v for k, v in request.get("lime_per_pond", {}).items()},
                probiotic_ponds=request.get("probiotic_ponds", []),
                antibiotic_ponds=request.get("antibiotic_ponds", []),
                partial_harvest=partial,
            )
            obs = env.step(action)

        elif _task_id == 3:
            from aquashrimp.models.actions import (
                CommercialGrowOutAction, PondFeedAction,
                PondTreatmentAction, HarvestAction
            )
            from aquashrimp.models.enums import TreatmentType, HarvestType
            pond_feeds = [PondFeedAction(**pf) for pf in request.get("pond_feeds", [])]
            treatments = [
                PondTreatmentAction(
                    pond_id=t["pond_id"],
                    treatment=TreatmentType(t["treatment"])
                ) for t in request.get("treatments", [])
            ]
            harvests = [
                HarvestAction(
                    pond_id=h["pond_id"],
                    harvest_type=HarvestType(h["harvest_type"]),
                    size_threshold_g=h.get("size_threshold_g"),
                    fraction=h.get("fraction"),
                ) for h in request.get("harvests", [])
            ]
            action = CommercialGrowOutAction(
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
            obs = env.step(action)

        else:
            raise HTTPException(status_code=500, detail=f"Unknown task_id={_task_id}")

        return _obs_to_dict(obs)

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
