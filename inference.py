#!/usr/bin/env python3
"""AquaShrimp OpenEnv — LLM inference script.

Uses an OpenAI-compatible client to drive a shrimp aquaculture farm agent.

Environment variables (with defaults):
    API_BASE_URL   LLM endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model id       (default: meta-llama/Llama-3.3-70B-Instruct)
    HF_TOKEN       HF / API key   (default: "" — LLM falls back to rule-based)

STDOUT format (parsed by validator):
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import urllib.request

from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "meta-llama/Llama-3.3-70B-Instruct"
API_KEY      = os.getenv("HF_TOKEN")     or os.getenv("API_KEY") or ""

BENCHMARK            = "aquashrimp"
SUCCESS_SCORE_THRESHOLD = 0.1

# ── Structured log helpers ────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Environment HTTP helpers ──────────────────────────────────────────────────

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


# ── LLM agent ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert shrimp aquaculture farm manager controlling a Litopenaeus vannamei (Whiteleg shrimp) farm.

Your goal is to maximize the episode grade [0.0 to 1.0] by making good daily farm management decisions.

Key rules:
- Shrimp need dissolved oxygen (DO) > 5 mg/L — aerate at least 18-22 hours/day or they die
- Feed at the suggested feed_demand_estimate_kg to avoid waste and TAN spikes
- Apply lime (5-10 kg) when pH < 7.8 to maintain alkalinity
- Check feeding trays to detect disease early (low tray consumption = disease warning)
- Increase water exchange (0.08-0.10) when TAN > 0.1 mg/L

You will receive the current observation as JSON and must respond with ONLY a valid JSON action object:
{
  "feed_kg": <float, 0-50>,
  "feeding_frequency": <int, 2-6>,
  "aeration_hours": <float, 0-24>,
  "water_exchange_frac": <float, 0-0.15>,
  "check_feeding_trays": <bool>,
  "lime_application_kg": <float, 0-20>
}

Respond with ONLY the JSON object, no explanation."""


def _fallback_action(obs: dict) -> dict:
    return {
        "feed_kg": float(obs.get("feed_demand_estimate_kg", 8.0)),
        "feeding_frequency": 4,
        "aeration_hours": 20.0,
        "water_exchange_frac": 0.05,
        "check_feeding_trays": True,
        "lime_application_kg": 5.0 if obs.get("ph", 8.0) < 7.8 else 0.0,
    }


def get_llm_action(client: OpenAI, obs: dict) -> dict:
    obs_summary = {k: v for k, v in obs.items() if k not in ("reward_breakdown", "done", "reward")}
    user_msg = (
        f"Current farm observation (day {obs.get('day', 0)}):\n"
        f"{json.dumps(obs_summary, indent=2)}\n\nWhat action do you take?"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=256,
        )
        content = response.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except Exception as e:
        print(f"[DEBUG] LLM error: {e} — using rule-based fallback", file=sys.stderr, flush=True)
        return _fallback_action(obs)


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(env_url: str, client: OpenAI, seed: int, task_name: str) -> None:
    """Run one full episode, emitting [START] / [STEP] / [END] to stdout."""
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = _post(f"{env_url}/reset", {"seed": seed})

        while not obs.get("done", False):
            action = get_llm_action(client, obs)
            error_str = None
            try:
                obs = _post(f"{env_url}/step", action)
            except Exception as e:
                error_str = str(e)
                obs = {"done": True, "reward": 0.0}

            reward = float(obs.get("reward", 0.0))
            done   = bool(obs.get("done", False))
            rewards.append(reward)
            steps_taken += 1

            log_step(
                step=steps_taken,
                action=json.dumps(action, separators=(",", ":")),
                reward=reward,
                done=done,
                error=error_str,
            )

        # Fetch final grade
        try:
            grade_resp = _get(f"{env_url}/grade")
            score = float(grade_resp.get("grade", 0.0))
        except Exception:
            score = sum(rewards) / max(len(rewards), 1) if rewards else 0.0

        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="AquaShrimp OpenEnv LLM inference")
    parser.add_argument("--env-url",  default="http://localhost:7860")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    if not API_KEY:
        print("WARNING: HF_TOKEN not set — LLM calls will fail; using rule-based fallback.", file=sys.stderr)

    client  = OpenAI(api_key=API_KEY or "dummy", base_url=API_BASE_URL)
    env_url = args.env_url.rstrip("/")

    # Resolve task name from health endpoint (best-effort)
    task_name = "AquaShrimp"
    try:
        health    = _get(f"{env_url}/health")
        task_id   = health.get("task_id", 1)
        task_name = {1: "NurseryPond", 2: "SemiIntensiveFarm", 3: "CommercialGrowOut"}.get(int(task_id), "AquaShrimp")
    except Exception as e:
        print(f"WARNING: Cannot reach {env_url}/health — {e}", file=sys.stderr, flush=True)

    for ep in range(args.episodes):
        try:
            run_episode(env_url, client, seed=args.seed + ep, task_name=task_name)
        except Exception as e:
            print(f"WARNING: Episode {ep + 1} failed — {e}", file=sys.stderr, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
