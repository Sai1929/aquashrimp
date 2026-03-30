"""Rule-based agent baseline for all three AquaShrimp tasks.

Rule-based strategy (documented in plan):
  - Always aerate 20h/day
  - Feed at 4% biomass/day adjusted for temperature
  - Check trays every 2 days; reduce feed 30% if consumption < 60%
  - Apply lime when pH < 7.8
  - Apply probiotics if redness_score > 2 (Vibrio signal)
  - Emergency harvest if WSSV detected (tray consumption < 20%)

Expected scores (100 episodes, seed=42):
  NurseryPond:         +0.30 to +0.50
  SemiIntensiveFarm:   +0.20 to +0.40
  CommercialGrowOut:   +0.15 to +0.35
"""
from __future__ import annotations
from aquashrimp.models.actions import (
    NurseryPondAction,
    SemiIntensiveFarmAction, PondFeedAction,
    CommercialGrowOutAction, HarvestAction,
)
from aquashrimp.models.enums import HarvestType
from aquashrimp.simulation.shrimp_growth import estimate_feed_demand


class RuleBasedAgent:
    """Threshold-based rule agent."""

    def __init__(self, task_id: int):
        self.task_id = task_id
        self._step = 0
        self._last_tray: dict[int, float] = {}
        self._wssv_suspected: dict[int, bool] = {}
        self._wssv_detected_day: dict[int, int] = {}

    def reset(self):
        self._step = 0
        self._last_tray = {}
        self._wssv_suspected = {}
        self._wssv_detected_day = {}

    def act(self, obs) -> NurseryPondAction | SemiIntensiveFarmAction | CommercialGrowOutAction:
        self._step += 1
        if self.task_id == 1:
            return self._task1(obs)
        elif self.task_id == 2:
            return self._task2(obs)
        elif self.task_id == 3:
            return self._task3(obs)
        raise ValueError(f"Unknown task_id={self.task_id}")

    def _task1(self, obs) -> NurseryPondAction:
        biomass_kg = obs.biomass_kg
        mean_w = obs.mean_weight_g
        temp = obs.temperature_c

        # Base feed: 4% body weight / day, temperature adjusted
        base_feed = estimate_feed_demand(biomass_kg, mean_w, temp)
        feed_kg = base_feed * 1.05  # slight over-estimate

        # Check trays every 2 days
        check_trays = (self._step % 2 == 0)

        # Reduce feed if tray shows < 60% consumption
        if obs.tray_consumption_fraction is not None:
            self._last_tray[0] = obs.tray_consumption_fraction
        last = self._last_tray.get(0, 0.9)
        if last < 0.6:
            feed_kg *= 0.7  # reduce 30%
        elif last > 0.85:
            feed_kg *= 1.1  # slight increase

        # Lime if pH below threshold
        lime_kg = 5.0 if obs.ph < 7.8 else 0.0

        return NurseryPondAction(
            feed_kg=float(min(feed_kg, 50.0)),
            feeding_frequency=4,
            aeration_hours=20.0,
            water_exchange_frac=0.05 if obs.tan_mg_L > 0.15 else 0.02,
            check_feeding_trays=check_trays,
            lime_application_kg=lime_kg,
        )

    def _task2(self, obs) -> SemiIntensiveFarmAction:
        feeds = []
        aeration = {}
        water_ex = {}
        check_t = {}
        lime_p = {}
        probiotics = []
        antibiotics = []

        # Determine aeration weights based on DO
        do_scores = []
        for pond in obs.ponds:
            # Lower DO = needs more aeration
            do_score = max(0.01, 10.0 - pond.do_mg_L)
            do_scores.append(do_score)
        total_score = sum(do_scores)
        alloc_fracs = [s / total_score for s in do_scores]

        for i, pond in enumerate(obs.ponds):
            pid = pond.pond_id
            aeration[pid] = alloc_fracs[i]
            check_t[pid] = (self._step % 2 == 0)

            # Feed
            demand = estimate_feed_demand(pond.biomass_kg, pond.mean_weight_g, pond.temperature_c)
            feed_kg = demand * 1.05
            last_tray = self._last_tray.get(pid, 0.9)
            if pond.tray_consumption_fraction is not None:
                self._last_tray[pid] = pond.tray_consumption_fraction
                last_tray = pond.tray_consumption_fraction
            if last_tray < 0.6:
                feed_kg *= 0.7
            elif last_tray > 0.85:
                feed_kg *= 1.1
            feeds.append(PondFeedAction(pond_id=pid, feed_kg=float(min(feed_kg, 500.0)), frequency=4))

            # Water exchange
            water_ex[pid] = 0.07 if pond.tan_mg_L > 0.15 else 0.03

            # Lime
            lime_p[pid] = 5.0 if pond.ph < 7.8 else 0.0

            # Disease response
            if pond.redness_score > 2.0:
                probiotics.append(pid)
            # No antibiotics by default (export risk)

        return SemiIntensiveFarmAction(
            pond_feeds=feeds,
            aeration_allocation=aeration,
            water_exchange=water_ex,
            check_trays=check_t,
            lime_per_pond=lime_p,
            probiotic_ponds=probiotics,
            antibiotic_ponds=antibiotics,
        )

    def _task3(self, obs) -> CommercialGrowOutAction:
        feeds = []
        aeration = {}
        water_ex = {}
        check_t = {}
        lime_p = {}
        harvests = []
        inspections = []
        emergency_ponds = []
        disinfect = []
        report = False

        # Check if we should submit overdue report
        if obs.regulatory_report_overdue:
            report = True

        for pond in obs.ponds:
            pid = pond.pond_id
            check_t[pid] = (self._step % 2 == 0)
            aeration[pid] = 20.0  # default 20h/day
            water_ex[pid] = 0.03

            if pond.fallow_status.value != "active":
                continue

            # Feed
            demand = estimate_feed_demand(pond.biomass_kg, pond.mean_weight_g, pond.temperature_c)
            feed_kg = demand * 1.05
            last_tray = self._last_tray.get(pid, 0.9)
            if pond.tray_consumption_fraction is not None:
                self._last_tray[pid] = pond.tray_consumption_fraction
                last_tray = pond.tray_consumption_fraction

            # WSSV early detection via tray
            if last_tray < 0.20 and not self._wssv_suspected.get(pid, False):
                self._wssv_suspected[pid] = True
                self._wssv_detected_day[pid] = self._step
                inspections.append(pid)  # inspect to confirm

            if self._wssv_suspected.get(pid, False):
                days_since = self._step - self._wssv_detected_day.get(pid, self._step)
                if days_since <= 3:
                    # Emergency harvest within detection window
                    harvests.append(HarvestAction(
                        pond_id=pid, harvest_type=HarvestType.EMERGENCY
                    ))
                    disinfect.append(pid)
                    continue

            if last_tray < 0.6:
                feed_kg *= 0.7
            elif last_tray > 0.85:
                feed_kg *= 1.1

            feeds.append(PondFeedAction(pond_id=pid, feed_kg=float(min(feed_kg, 2000.0)), frequency=4))

            # Water exchange
            water_ex[pid] = 0.05 if pond.tan_mg_L > 0.15 else 0.02

            # Lime
            lime_p[pid] = 5.0 if pond.ph < 7.8 else 0.0

        return CommercialGrowOutAction(
            pond_feeds=feeds,
            aeration_per_pond=aeration,
            water_exchange=water_ex,
            check_trays=check_t,
            pond_inspections=inspections,
            lime_per_pond=lime_p,
            biosecurity_measure=obs.neighbor_outbreak,  # activate when neighbor outbreak
            treatments=[],
            harvests=harvests,
            disinfect_pond=disinfect,
            regulatory_report=report,
        )
