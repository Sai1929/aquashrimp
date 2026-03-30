"""Baseline runner script with reproducible score reporting.

Usage:
  python aquashrimp/baselines/run_baseline.py --agent random    --task nursery_pond     --seed 42 --episodes 100
  python aquashrimp/baselines/run_baseline.py --agent rule      --task nursery_pond     --seed 42 --episodes 100
  python aquashrimp/baselines/run_baseline.py --agent optimal   --task nursery_pond     --seed 42 --episodes 50
  python aquashrimp/baselines/run_baseline.py --agent rule      --task semi_intensive   --seed 42 --episodes 50
  python aquashrimp/baselines/run_baseline.py --agent rule      --task commercial_grow_out --seed 42 --episodes 20
"""
from __future__ import annotations
import argparse
import statistics
from typing import Any


def make_env(task: str, seed: int = 42) -> Any:
    if task in ("nursery_pond", "nursery", "1"):
        from aquashrimp.tasks.nursery_pond import NurseryPondEnvironment
        return NurseryPondEnvironment(seed=seed)
    elif task in ("semi_intensive", "semi_intensive_farm", "2"):
        from aquashrimp.tasks.semi_intensive_farm import SemiIntensiveFarmEnvironment
        return SemiIntensiveFarmEnvironment(seed=seed)
    elif task in ("commercial_grow_out", "commercial", "3"):
        from aquashrimp.tasks.commercial_grow_out import CommercialGrowOutEnvironment
        return CommercialGrowOutEnvironment(seed=seed)
    raise ValueError(f"Unknown task: {task}")


def make_agent(agent: str, task_id: int, seed: int = 0) -> Any:
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


def task_name_to_id(task: str) -> int:
    mapping = {
        "nursery_pond": 1, "nursery": 1, "1": 1,
        "semi_intensive": 2, "semi_intensive_farm": 2, "2": 2,
        "commercial_grow_out": 3, "commercial": 3, "3": 3,
    }
    return mapping.get(task, 1)


def run_episode(env, agent) -> tuple[float, int, float]:
    """Run a single episode. Returns (episode_reward, steps, final_survival)."""
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

    # Get final survival rate
    if hasattr(obs, "survival_rate"):
        survival = obs.survival_rate
    elif hasattr(obs, "ponds"):
        ponds = [p for p in obs.ponds if hasattr(p, "survival_rate")]
        survival = sum(p.survival_rate for p in ponds) / max(len(ponds), 1)
    else:
        survival = 0.0

    return total_reward, steps, survival


def main():
    parser = argparse.ArgumentParser(description="AquaShrimp baseline runner")
    parser.add_argument("--agent", choices=["random", "rule", "optimal"], default="rule")
    parser.add_argument("--task", default="nursery_pond",
                        choices=["nursery_pond", "nursery", "semi_intensive", "semi_intensive_farm",
                                 "commercial_grow_out", "commercial", "1", "2", "3"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    task_id = task_name_to_id(args.task)
    env = make_env(args.task, seed=args.seed)
    agent = make_agent(args.agent, task_id=task_id, seed=args.seed + 1)

    episode_rewards = []
    episode_survivals = []

    for ep in range(args.episodes):
        # Different seed per episode for variance
        env.seed = args.seed + ep
        reward, steps, survival = run_episode(env, agent)
        episode_rewards.append(reward)
        episode_survivals.append(survival)
        if args.episodes <= 20 or (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1:3d}/{args.episodes}: reward={reward:+.3f} | steps={steps} | survival={survival:.1%}")

    mean_r = statistics.mean(episode_rewards)
    std_r = statistics.stdev(episode_rewards) if len(episode_rewards) > 1 else 0.0
    min_r = min(episode_rewards)
    max_r = max(episode_rewards)
    mean_s = statistics.mean(episode_survivals)

    print(f"\n{'='*60}")
    print(f"Agent:   {args.agent}")
    print(f"Task:    {args.task} (task_id={task_id})")
    print(f"Episodes: {args.episodes} | Seed: {args.seed}")
    print(f"Reward:  mean={mean_r:+.3f} ± {std_r:.3f} | min={min_r:+.3f} | max={max_r:+.3f}")
    print(f"Survival: mean={mean_s:.1%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
