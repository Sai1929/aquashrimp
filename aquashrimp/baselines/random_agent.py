"""Random agent baseline for all three AquaShrimp tasks.

Expected scores (100 episodes, seed=42):
  NurseryPond:         -0.30 to -0.05
  SemiIntensiveFarm:   -0.40 to -0.05
  CommercialGrowOut:   -0.50 to -0.10

Random agent fails badly because:
- Random aeration → frequent nighttime DO crashes → shrimp die
- Random feeding → TAN spikes → ammonia poisoning
"""
from __future__ import annotations
import numpy as np
from aquashrimp.models.actions import (
    NurseryPondAction,
    SemiIntensiveFarmAction, PondFeedAction,
    CommercialGrowOutAction,
)


class RandomAgent:
    """Samples uniformly random valid actions."""

    def __init__(self, task_id: int, seed: int = 0):
        self.task_id = task_id
        self._rng = np.random.default_rng(seed)

    def act(self, obs) -> NurseryPondAction | SemiIntensiveFarmAction | CommercialGrowOutAction:
        if self.task_id == 1:
            return self._task1_action()
        elif self.task_id == 2:
            return self._task2_action()
        elif self.task_id == 3:
            return self._task3_action()
        raise ValueError(f"Unknown task_id={self.task_id}")

    def _task1_action(self) -> NurseryPondAction:
        return NurseryPondAction(
            feed_kg=float(self._rng.uniform(0.5, 30.0)),
            feeding_frequency=int(self._rng.integers(2, 7)),
            aeration_hours=float(self._rng.uniform(0.0, 24.0)),
            water_exchange_frac=float(self._rng.uniform(0.0, 0.15)),
            check_feeding_trays=bool(self._rng.random() > 0.5),
            lime_application_kg=float(self._rng.uniform(0.0, 15.0)),
        )

    def _task2_action(self) -> SemiIntensiveFarmAction:
        n_ponds = 4
        # Random aeration split summing to ≤ 1.0
        alloc = self._rng.dirichlet(np.ones(n_ponds))
        feeds = [
            PondFeedAction(
                pond_id=i,
                feed_kg=float(self._rng.uniform(2.0, 100.0)),
                frequency=int(self._rng.integers(2, 7)),
            )
            for i in range(n_ponds)
        ]
        return SemiIntensiveFarmAction(
            pond_feeds=feeds,
            aeration_allocation={i: float(alloc[i]) for i in range(n_ponds)},
            water_exchange={i: float(self._rng.uniform(0.0, 0.12)) for i in range(n_ponds)},
            check_trays={i: bool(self._rng.random() > 0.6) for i in range(n_ponds)},
            lime_per_pond={i: float(self._rng.uniform(0.0, 10.0)) for i in range(n_ponds)},
        )

    def _task3_action(self) -> CommercialGrowOutAction:
        n_ponds = 10
        feeds = [
            PondFeedAction(
                pond_id=i,
                feed_kg=float(self._rng.uniform(10.0, 500.0)),
                frequency=int(self._rng.integers(2, 7)),
            )
            for i in range(n_ponds)
        ]
        return CommercialGrowOutAction(
            pond_feeds=feeds,
            aeration_per_pond={i: float(self._rng.uniform(8.0, 24.0)) for i in range(n_ponds)},
            water_exchange={i: float(self._rng.uniform(0.0, 0.1)) for i in range(n_ponds)},
            check_trays={i: bool(self._rng.random() > 0.7) for i in range(n_ponds)},
        )
