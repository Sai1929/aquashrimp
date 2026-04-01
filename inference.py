#!/usr/bin/env python3
"""AquaShrimp OpenEnv — LLM inference script.

Uses an LLM (via OpenAI-compatible client) as the agent to manage a shrimp
aquaculture farm through the AquaShrimp HTTP environment API.

Required environment variables:
    API_BASE_URL   The API endpoint for the LLM (OpenAI-compatible).
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Usage:
    export API_BASE_URL="https://api-inference.huggingface.co/v1"
    export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
    export HF_TOKEN="hf_..."
    python inference.py --env-url https://vsai00-aquashrimp-task1.hf.space
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import urllib.request
import urllib.error

from openai import OpenAI


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

You will receive the current observation as JSON and must respond with ONLY a valid JSON action object.

For Task 1 (NurseryPond), your response must be a JSON object with these fields:
{
  "feed_kg": <float, 0-50>,
  "feeding_frequency": <int, 2-6>,
  "aeration_hours": <float, 0-24>,
  "water_exchange_frac": <float, 0-0.15>,
  "check_feeding_trays": <bool>,
  "lime_application_kg": <float, 0-20>
}

Respond with ONLY the JSON object, no explanation."""


def get_llm_action(client: OpenAI, model: str, obs: dict) -> dict:
    """Ask the LLM to choose an action given the current observation."""
    obs_summary = {
        k: v for k, v in obs.items()
        if k not in ("reward_breakdown", "done", "reward")
    }
    user_msg = f"Current farm observation (day {obs.get('day', 0)}):\n{json.dumps(obs_summary, indent=2)}\n\nWhat action do you take?"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=256,
    )

    content = response.choices[0].message.content.strip()

    # Parse JSON from LLM response
    try:
        # Strip markdown code fences if present
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except json.JSONDecodeError:
        # Fallback to safe default action if LLM response is malformed
        return {
            "feed_kg": float(obs.get("feed_demand_estimate_kg", 8.0)),
            "feeding_frequency": 4,
            "aeration_hours": 20.0,
            "water_exchange_frac": 0.05,
            "check_feeding_trays": True,
            "lime_application_kg": 5.0 if obs.get("ph", 8.0) < 7.8 else 0.0,
        }


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(env_url: str, client: OpenAI, model: str, seed: int = 42, verbose: bool = False) -> dict:
    """Run one full episode using the LLM as the agent."""
    obs = _post(f"{env_url}/reset", {"seed": seed})
    total_reward = 0.0
    steps = 0

    while not obs.get("done", False):
        action = get_llm_action(client, model, obs)
        obs = _post(f"{env_url}/step", action)
        total_reward += obs.get("reward", 0.0)
        steps += 1
        if verbose:
            print(
                f"  step {steps:3d}: reward={obs.get('reward', 0):+.4f}  "
                f"grade={obs.get('grade', 0):.4f}  "
                f"weight={obs.get('mean_weight_g', 0):.2f}g",
                flush=True,
            )

    grade_resp = _get(f"{env_url}/grade")
    return {
        "seed": seed,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "mean_reward": round(total_reward / max(steps, 1), 4),
        "grade": round(grade_resp["grade"], 6),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="AquaShrimp OpenEnv LLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--env-url", default="http://localhost:7860",
        help="AquaShrimp environment base URL (default: http://localhost:7860)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Load required env vars
    api_base_url = os.environ.get("API_BASE_URL")
    model_name = os.environ.get("MODEL_NAME")
    hf_token = os.environ.get("HF_TOKEN")

    if not api_base_url:
        print("ERROR: API_BASE_URL environment variable not set.", file=sys.stderr)
        print("  export API_BASE_URL=https://api-inference.huggingface.co/v1", file=sys.stderr)
        return 1
    if not model_name:
        print("ERROR: MODEL_NAME environment variable not set.", file=sys.stderr)
        print("  export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct", file=sys.stderr)
        return 1
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set.", file=sys.stderr)
        print("  export HF_TOKEN=hf_...", file=sys.stderr)
        return 1

    # Build OpenAI-compatible client
    client = OpenAI(api_key=hf_token, base_url=api_base_url)

    env_url = args.env_url.rstrip("/")

    # Health check
    try:
        health = _get(f"{env_url}/health")
        print(f"Connected: {env_url}  task_id={health.get('task_id', '?')}")
    except Exception as e:
        print(f"ERROR: Cannot reach {env_url}/health — {e}", file=sys.stderr)
        return 1

    print(f"Model: {model_name}  |  API: {api_base_url}")

    grades = []
    for ep in range(args.episodes):
        print(f"\nEpisode {ep + 1}/{args.episodes}  (seed={args.seed + ep})")
        result = run_episode(env_url, client, model_name, seed=args.seed + ep, verbose=args.verbose)
        grades.append(result["grade"])
        print(
            f"  => steps={result['steps']}  "
            f"total_reward={result['total_reward']:+.4f}  "
            f"mean_reward={result['mean_reward']:+.4f}  "
            f"grade={result['grade']:.4f}"
        )

    mean_grade = sum(grades) / len(grades)
    print(f"\n{'='*50}")
    print(f"Model    : {model_name}")
    print(f"Episodes : {args.episodes}  |  Seed: {args.seed}")
    print(f"Mean grade: {mean_grade:.4f}  (range [0.0, 1.0])")
    print(f"{'='*50}")
    return 0


if __name__ == "__main__":
    sys.exit(main())