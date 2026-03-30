"""Near-optimal agent baseline (upper bound) for AquaShrimp.

This agent has access to the true simulation state and uses near-optimal
heuristics calibrated to the reward function.

Expected scores:
  NurseryPond:         +0.70 to +0.85
  SemiIntensiveFarm:   +0.60 to +0.72
  CommercialGrowOut:   +0.50 to +0.65

Strategy:
  - Feed exactly at demand estimate (minimizes FCR, avoids TAN spikes)
  - Aerate 22h/day (leaves 2h gap, close to optimum)
  - Water exchange 5% daily (maintains water quality without cost)
  - Check trays every day (full information)
  - Lime proactively to keep alkalinity >110 mg/L
  - At WSSV detection (day 1–2): immediately harvest + disinfect
  - Task 3: harvest at 22g (premium size, before price decline)
"""
from __future__ import annotations
from aquashrimp.models.actions import (
    NurseryPondAction,
    SemiIntensiveFarmAction, PondFeedAction, PartialHarvestAction,
    CommercialGrowOutAction, HarvestAction,
)
from aquashrimp.models.enums import HarvestType
from aquashrimp.simulation.shrimp_growth import estimate_feed_demand


class OptimalAgent:
    """Near-optimal heuristic agent with full information access."""

    def __init__(self, task_id: int):
        self.task_id = task_id
        self._step = 0
        self._wssv_detection: dict[int, int] = {}

    def reset(self):
        self._step = 0
        self._wssv_detection = {}

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
        # Optimal feed = exactly demand estimate (FCR stays near target)
        feed_kg = obs.feed_demand_estimate_kg * 1.0

        # Tray check every day for full information
        # If tray showed reduced consumption, adjust
        tray = obs.tray_consumption_fraction
        if tray is not None and tray < 0.7:
            feed_kg *= 0.8

        # Proactive lime: maintain alkalinity > 110
        lime_kg = 8.0 if obs.alkalinity_mg_L < 110.0 else 0.0

        # Increase exchange if TAN rising
        exchange = 0.08 if obs.tan_mg_L > 0.08 else 0.04

        return NurseryPondAction(
            feed_kg=float(min(feed_kg, 50.0)),
            feeding_frequency=5,       # 5 meals/day = better FCR
            aeration_hours=22.0,
            water_exchange_frac=exchange,
            check_feeding_trays=True,  # always check
            lime_application_kg=lime_kg,
        )

    def _task2(self, obs) -> SemiIntensiveFarmAction:
        feeds = []
        aeration = {}
        water_ex = {}
        check_t = {}
        lime_p = {}
        probiotics = []
        partial_harvest = None

        do_deficits = []
        for pond in obs.ponds:
            deficit = max(0, 5.0 - pond.do_mg_L)  # amount below 5 mg/L
            do_deficits.append(deficit + 0.1)  # small base allocation

        total_deficit = sum(do_deficits)
        alloc = [d / total_deficit for d in do_deficits]

        for i, pond in enumerate(obs.ponds):
            pid = pond.pond_id
            aeration[pid] = alloc[i]
            check_t[pid] = True

            demand = estimate_feed_demand(pond.biomass_kg, pond.mean_weight_g, pond.temperature_c)
            feed_kg = demand
            tray = pond.tray_consumption_fraction
            if tray is not None and tray < 0.65:
                feed_kg *= 0.75
            elif tray is not None and tray > 0.88:
                feed_kg *= 1.08

            feeds.append(PondFeedAction(pond_id=pid, feed_kg=float(min(feed_kg, 500.0)), frequency=5))

            water_ex[pid] = 0.06 if pond.tan_mg_L > 0.08 else 0.03
            lime_p[pid] = 6.0 if pond.alkalinity_mg_L < 110.0 else 0.0

            # Probiotics at first Vibrio sign (avoid antibiotics)
            if pond.redness_score > 1.5:
                probiotics.append(pid)

            # Partial harvest when density is high and shrimp > 8g
            if pond.mean_weight_g > 8.0 and pond.density_kg_m2 > 3.0 and partial_harvest is None:
                partial_harvest = PartialHarvestAction(
                    pond_id=pid, size_threshold_g=9.0, fraction=0.3
                )

        return SemiIntensiveFarmAction(
            pond_feeds=feeds,
            aeration_allocation=aeration,
            water_exchange=water_ex,
            check_trays=check_t,
            lime_per_pond=lime_p,
            probiotic_ponds=probiotics,
            antibiotic_ponds=[],  # never use antibiotics (export protection)
            partial_harvest=partial_harvest,
        )

    def _task3(self, obs) -> CommercialGrowOutAction:
        feeds = []
        aeration = {}
        water_ex = {}
        check_t = {}
        lime_p = {}
        harvests = []
        inspections = []
        disinfect = []
        report = obs.regulatory_report_overdue

        # Biosecurity when neighbor outbreak
        biosec = obs.neighbor_outbreak

        for pond in obs.ponds:
            pid = pond.pond_id
            check_t[pid] = True
            aeration[pid] = 22.0

            if pond.fallow_status.value != "active":
                continue

            # WSSV detection: tray < 20% = immediate response
            tray = pond.tray_consumption_fraction
            if tray is not None and tray < 0.20 and pid not in self._wssv_detection:
                self._wssv_detection[pid] = self._step
                inspections.append(pid)

            if pid in self._wssv_detection:
                days = self._step - self._wssv_detection[pid]
                if days <= 2:
                    harvests.append(HarvestAction(pond_id=pid, harvest_type=HarvestType.EMERGENCY))
                    disinfect.append(pid)
                    continue

            # WSSV confirmation
            if pond.white_spots_visible and pid not in self._wssv_detection:
                self._wssv_detection[pid] = self._step
                harvests.append(HarvestAction(pond_id=pid, harvest_type=HarvestType.EMERGENCY))
                disinfect.append(pid)
                report = True
                continue

            # Optimal harvest at 22g (premium size)
            if pond.mean_weight_g >= 22.0:
                harvests.append(HarvestAction(pond_id=pid, harvest_type=HarvestType.FULL))
                continue

            demand = estimate_feed_demand(pond.biomass_kg, pond.mean_weight_g, pond.temperature_c)
            feed_kg = demand
            if tray is not None and tray < 0.65:
                feed_kg *= 0.75
            elif tray is not None and tray > 0.88:
                feed_kg *= 1.08

            feeds.append(PondFeedAction(pond_id=pid, feed_kg=float(min(feed_kg, 2000.0)), frequency=5))
            water_ex[pid] = 0.05 if pond.tan_mg_L > 0.08 else 0.02
            lime_p[pid] = 6.0 if pond.alkalinity_mg_L < 110.0 else 0.0

        return CommercialGrowOutAction(
            pond_feeds=feeds,
            aeration_per_pond=aeration,
            water_exchange=water_ex,
            check_trays=check_t,
            pond_inspections=inspections,
            lime_per_pond=lime_p,
            biosecurity_measure=biosec,
            treatments=[],
            harvests=harvests,
            disinfect_pond=disinfect,
            regulatory_report=report,
        )
