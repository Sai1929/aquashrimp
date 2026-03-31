#!/usr/bin/env python3
"""AquaShrimp OpenEnv — inference script.

Runs a rule-based agent against the environment via HTTP and reports the grade.

Usage:
    # Against local server (Task 1)
    AQUASHRIMP_TASK=1 uvicorn aquashrimp.server.app:app --port 7860 &
    python inference.py --base-url http://localhost:7860

    # Against HuggingFace Space (Task 1)
    python inference.py --base-url https://vsai00-aquashrimp-task1.hf.space

    # Multi-task Space, specific task prefix
    python inference.py --base-url https://vsai00-aquashrimp.hf.space/task/1

    # Run multiple episodes
    python inference.py --base-url https://vsai00-aquashrimp-task1.hf.space --episodes 5 --seed 42
"""
from __future__ import annotations
import argparse
import json
import sys
import urllib.request
import urllib.error


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _post(url: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        return json.loads(res.read())


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as res:
        return json.loads(res.read())


# ── Rule-based action selection ───────────────────────────────────────────────

def _select_action(obs: dict) -> dict:
    """Threshold-based rule agent — works for all 3 tasks."""
    task_id = obs.get("_task_id", 1)

    # Task 1: NurseryPond
    if task_id == 1 or "feed_demand_estimate_kg" in obs:
        feed = obs.get("feed_demand_estimate_kg", 8.0)
        tray = obs.get("tray_consumption_fraction")
        if tray is not None and tray < 0.6:
            feed *= 0.8
        return {
            "feed_kg": round(float(feed), 3),
            "feeding_frequency": 4,
            "aeration_hours": 20.0,
            "water_exchange_frac": 0.08 if obs.get("tan_mg_L", 0) > 0.1 else 0.05,
            "check_feeding_trays": True,
            "lime_application_kg": 5.0 if obs.get("ph", 8.0) < 7.8 else 0.0,
        }

    # Task 2 & 3: multi-pond (send minimal safe action)
    ponds = obs.get("ponds", [])
    pond_feeds = [
        {
            "pond_id": p["pond_id"],
            "feed_kg": round(float(p.get("feed_demand_estimate_kg", 10.0)), 3),
            "frequency": 4,
        }
        for p in ponds
        if p.get("fallow_status", "active") == "active"
    ]
    n = max(len(ponds), 1)
    alloc = {str(p["pond_id"]): round(1.0 / n, 4) for p in ponds}
    exchange = {str(p["pond_id"]): 0.05 for p in ponds}
    check = {str(p["pond_id"]): True for p in ponds}
    lime = {str(p["pond_id"]): 5.0 if p.get("ph", 8.0) < 7.8 else 0.0 for p in ponds}

    if task_id == 2:
        return {
            "pond_feeds": pond_feeds,
            "aeration_allocation": alloc,
            "water_exchange": exchange,
            "check_trays": check,
            "lime_per_pond": lime,
            "probiotic_ponds": [
                p["pond_id"] for p in ponds if p.get("redness_score", 0) > 1.5
            ],
            "antibiotic_ponds": [],
            "partial_harvest": None,
        }
    else:  # task 3
        return {
            "pond_feeds": pond_feeds,
            "aeration_per_pond": {str(p["pond_id"]): 20.0 for p in ponds},
            "water_exchange": exchange,
            "check_trays": check,
            "pond_inspections": [],
            "lime_per_pond": lime,
            "biosecurity_measure": obs.get("neighbor_outbreak", False),
            "treatments": [],
            "harvests": [],
            "disinfect_pond": [],
            "regulatory_report": obs.get("regulatory_report_overdue", False),
        }


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(base_url: str, seed: int = 42, verbose: bool = False) -> dict:
    """Run one full episode. Returns summary dict."""
    obs = _post(f"{base_url}/reset", {"seed": seed})

    # Detect task from obs shape
    if "ponds" in obs:
        task_id = 2 if len(obs["ponds"]) <= 4 else 3
    else:
        task_id = 1

    total_reward = 0.0
    steps = 0

    while not obs.get("done", False):
        obs["_task_id"] = task_id
        action = _select_action(obs)
        obs = _post(f"{base_url}/step", action)
        total_reward += obs.get("reward", 0.0)
        steps += 1
        if verbose:
            print(
                f"  step {steps:3d}: reward={obs['reward']:+.4f}  "
                f"grade={obs.get('grade', 0):.4f}",
                flush=True,
            )

    grade_resp = _get(f"{base_url}/grade")
    return {
        "task_id": task_id,
        "seed": seed,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "mean_reward": round(total_reward / max(steps, 1), 4),
        "grade": round(grade_resp["grade"], 6),
        "episode_done": grade_resp["episode_done"],
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="AquaShrimp OpenEnv inference — run rule-based agent via HTTP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-url", default="http://localhost:7860",
        help="Base URL of the AquaShrimp server (default: http://localhost:7860)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes (default 1)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-step output")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    # Health check
    try:
        health = _get(f"{base}/health")
        print(f"Connected: {base}  task_id={health.get('task_id', '?')}")
    except Exception as e:
        print(f"ERROR: Cannot reach {base}/health — {e}", file=sys.stderr)
        return 1

    grades = []
    for ep in range(args.episodes):
        print(f"\nEpisode {ep + 1}/{args.episodes} (seed={args.seed + ep})")
        result = run_episode(base, seed=args.seed + ep, verbose=args.verbose)
        grades.append(result["grade"])
        print(
            f"  => steps={result['steps']}  "
            f"total_reward={result['total_reward']:+.4f}  "
            f"mean_reward={result['mean_reward']:+.4f}  "
            f"grade={result['grade']:.4f}"
        )

    mean_grade = sum(grades) / len(grades)
    print(f"\n{'='*50}")
    print(f"Episodes : {args.episodes}  |  Seed: {args.seed}")
    print(f"Mean grade: {mean_grade:.4f}  (range [0.0, 1.0])")
    print(f"{'='*50}")
    return 0


if __name__ == "__main__":
    sys.exit(main())