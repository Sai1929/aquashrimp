#!/usr/bin/env python3
"""Reproducible benchmark across all AquaShrimp agents and tasks.

Produces a full results table with mean per-step reward, grade [0-1],
survival rate, and FCR. All runs use seed=42 for reproducibility.

Usage:
    python scripts/benchmark_all.py                  # 5 episodes each (fast)
    python scripts/benchmark_all.py --episodes 20    # more episodes for tighter CI
    python scripts/benchmark_all.py --task nursery_pond --agent rule  # single combo
    python scripts/benchmark_all.py --seed 0         # different seed

Reference scores (seed=42, 5 episodes):
    Agent      NurseryPond  SemiIntensive  CommercialGrowOut
    random     +0.09        +0.23          -0.03
    rule       +0.21        +0.37          +0.10
    optimal    +0.26        +0.40          +0.11
"""
from __future__ import annotations
import argparse
import statistics
import sys
import time
from typing import NamedTuple


TASKS = ["nursery_pond", "semi_intensive_farm", "commercial_grow_out"]
AGENTS = ["random", "rule", "optimal"]

TASK_IDS = {"nursery_pond": 1, "semi_intensive_farm": 2, "commercial_grow_out": 3}
TASK_DISPLAY = {
    "nursery_pond": "NurseryPond",
    "semi_intensive_farm": "SemiIntensive",
    "commercial_grow_out": "CommercialGrowOut",
}


class EpisodeResult(NamedTuple):
    reward_per_step: float
    grade: float
    survival: float
    steps: int


def make_env(task: str, seed: int):
    if task == "nursery_pond":
        from aquashrimp.tasks.nursery_pond import NurseryPondEnvironment
        return NurseryPondEnvironment(seed=seed)
    elif task == "semi_intensive_farm":
        from aquashrimp.tasks.semi_intensive_farm import SemiIntensiveFarmEnvironment
        return SemiIntensiveFarmEnvironment(seed=seed)
    elif task == "commercial_grow_out":
        from aquashrimp.tasks.commercial_grow_out import CommercialGrowOutEnvironment
        return CommercialGrowOutEnvironment(seed=seed)
    raise ValueError(f"Unknown task: {task}")


def make_agent(agent: str, task_id: int, seed: int):
    if agent == "random":
        from aquashrimp.baselines.random_agent import RandomAgent
        return RandomAgent(task_id=task_id, seed=seed)
    elif agent == "rule":
        from aquashrimp.baselines.rule_based_agent import RuleBasedAgent
        return RuleBasedAgent(task_id=task_id)
    elif agent == "optimal":
        from aquashrimp.baselines.optimal_agent import OptimalAgent
        return OptimalAgent(task_id=task_id)
    raise ValueError(f"Unknown agent: {agent}")


def run_episode(env, agent) -> EpisodeResult:
    if hasattr(agent, "reset"):
        agent.reset()

    obs = env.reset()
    total_reward = 0.0
    steps = 0

    while not obs.done:
        action = agent.act(obs)
        obs = env.step(action)
        total_reward += obs.reward
        steps += 1

    if steps == 0:
        steps = 1

    reward_per_step = total_reward / steps
    grade = max(0.0, min(1.0, (reward_per_step + 1.0) / 2.0))

    if hasattr(obs, "survival_rate"):
        survival = obs.survival_rate
    elif hasattr(obs, "ponds"):
        active = [p for p in obs.ponds if p.survival_rate > 0]
        survival = statistics.mean(p.survival_rate for p in active) if active else 0.0
    else:
        survival = 0.0

    return EpisodeResult(reward_per_step=reward_per_step, grade=grade, survival=survival, steps=steps)


def run_combo(task: str, agent: str, episodes: int, seed: int, verbose: bool) -> dict:
    task_id = TASK_IDS[task]
    env = make_env(task, seed=seed)
    ag = make_agent(agent, task_id=task_id, seed=seed + 1)

    results = []
    for ep in range(episodes):
        env.seed = seed + ep
        r = run_episode(env, ag)
        results.append(r)
        if verbose:
            print(
                f"    ep {ep+1:3d}/{episodes}: "
                f"reward={r.reward_per_step:+.4f}  grade={r.grade:.4f}  "
                f"survival={r.survival:.1%}  steps={r.steps}"
            )

    rewards = [r.reward_per_step for r in results]
    grades = [r.grade for r in results]
    survivals = [r.survival for r in results]

    return {
        "task": task,
        "agent": agent,
        "episodes": episodes,
        "seed": seed,
        "mean_reward": statistics.mean(rewards),
        "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
        "mean_grade": statistics.mean(grades),
        "mean_survival": statistics.mean(survivals),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
    }


def print_table(all_results: list[dict]) -> None:
    """Print grade-based leaderboard table."""
    print("\n" + "=" * 72)
    print("AQUASHRIMP BASELINE BENCHMARK  (mean per-step reward | grade [0-1])")
    print("=" * 72)

    header = f"{'Agent':<10}  {'NurseryPond':>18}  {'SemiIntensive':>18}  {'CommercialGrowOut':>18}"
    print(header)
    print("-" * 72)

    by_agent: dict[str, dict] = {a: {} for a in AGENTS}
    for r in all_results:
        by_agent[r["agent"]][r["task"]] = r

    for agent in AGENTS:
        row = f"{agent:<10}"
        for task in TASKS:
            if task in by_agent[agent]:
                r = by_agent[agent][task]
                cell = f"{r['mean_reward']:+.3f} / {r['mean_grade']:.3f}"
            else:
                cell = "      —      "
            row += f"  {cell:>18}"
        print(row)

    print("-" * 72)
    print("Format: mean_per_step_reward / grade")
    print(f"Seeds: base={all_results[0]['seed']} (+ep offset)  Episodes: {all_results[0]['episodes']}")
    print("=" * 72)


def print_detailed(all_results: list[dict]) -> None:
    """Print per-combo detailed stats."""
    print("\n" + "=" * 72)
    print("DETAILED RESULTS")
    print("=" * 72)
    for r in all_results:
        print(
            f"  {TASK_DISPLAY[r['task']]:<20} / {r['agent']:<7}: "
            f"reward={r['mean_reward']:+.3f}±{r['std_reward']:.3f}  "
            f"grade={r['mean_grade']:.3f}  survival={r['mean_survival']:.1%}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="AquaShrimp reproducible benchmark — all agents × all tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per combo (default 5)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default 42)")
    parser.add_argument(
        "--task", choices=TASKS + ["all"], default="all",
        help="Task to benchmark (default: all)"
    )
    parser.add_argument(
        "--agent", choices=AGENTS + ["all"], default="all",
        help="Agent to benchmark (default: all)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-episode results")
    parser.add_argument("--no-table", action="store_true", help="Skip the summary table")
    args = parser.parse_args()

    tasks = TASKS if args.task == "all" else [args.task]
    agents = AGENTS if args.agent == "all" else [args.agent]
    combos = [(t, a) for t in tasks for a in agents]

    print(f"AquaShrimp Benchmark: {len(combos)} combos × {args.episodes} episodes  (seed={args.seed})")

    all_results = []
    t0 = time.time()

    for i, (task, agent) in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}] {TASK_DISPLAY[task]} / {agent} ...", flush=True)
        result = run_combo(task, agent, args.episodes, args.seed, args.verbose)
        all_results.append(result)
        print(
            f"  -> reward={result['mean_reward']:+.3f}±{result['std_reward']:.3f}  "
            f"grade={result['mean_grade']:.3f}  survival={result['mean_survival']:.1%}"
        )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s  ({elapsed/len(combos):.1f}s/combo)")

    if not args.no_table:
        print_table(all_results)
        print_detailed(all_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
