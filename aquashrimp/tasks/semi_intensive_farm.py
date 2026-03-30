"""Task 2: SemiIntensiveFarm — 4-pond farm with shared aeration.

Difficulty: Medium | Max steps: 60 days
Key challenges: aeration allocation across 4 ponds, Vibrio outbreak (day 20–40),
antibiotic vs probiotic trade-off, partial harvest decisions.
"""
from __future__ import annotations
import numpy as np
from aquashrimp.models.actions import SemiIntensiveFarmAction, PondFeedAction, PartialHarvestAction
from aquashrimp.models.observations import SemiIntensiveFarmObs, PondObs, RewardBreakdown
from aquashrimp.models.state import PondState, EpisodeState
from aquashrimp.models.enums import WeatherEvent, DiseaseType, FallowStatus
from aquashrimp.simulation import shrimp_growth as sg
from aquashrimp.simulation import water_quality as wq
from aquashrimp.simulation.feeding_trays import true_consumption_fraction, observe_tray, uneaten_feed
from aquashrimp.simulation.weather import seasonal_temperature
from aquashrimp.simulation.disease import (
    vibrio_trigger_day, step_vibrio_severity,
    vibrio_mortality_count, vibrio_observable_signals
)
from aquashrimp.simulation.market import (
    compute_daily_costs, daily_revenue_accrual, update_price,
    compute_harvest_revenue, VANNAMEI_PRICE_MEAN
)
from aquashrimp.rewards.reward_calculator import (
    compute_total_reward, growth_reward, water_quality_reward,
    economic_reward, biosecurity_reward, expected_weight, compute_grade
)

# ── Farm configuration ────────────────────────────────────────────────────────
NUM_PONDS = 4
POND_AREA_M2 = 5000.0   # 0.5 ha each
POND_VOLUME_M3 = 7500.0
INITIAL_STOCKING = 400_000  # per pond
INITIAL_WEIGHT_G = 0.05
NUM_AERATORS = 8
MAX_AERATION_HOURS = 24.0
TOTAL_AERATION_CAPACITY = NUM_AERATORS * MAX_AERATION_HOURS  # 192 aerator-hours/day


class SemiIntensiveFarmEnvironment:
    """Task 2: 4-pond semi-intensive farm environment."""

    task_id = 2
    max_steps = 60

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._state: EpisodeState | None = None
        self._vibrio_treatment: dict[int, dict] = {}  # pond_id → treatment state
        self._cumulative_reward: float = 0.0

    def reset(self) -> SemiIntensiveFarmObs:
        self._rng = np.random.default_rng(self.seed)
        self._vibrio_treatment = {}

        ponds = []
        for i in range(NUM_PONDS):
            # Stagger stocking ages slightly
            age_offset = i * 5  # ponds stocked 5 days apart
            ponds.append(PondState(
                pond_id=i,
                area_m2=POND_AREA_M2,
                volume_m3=POND_VOLUME_M3,
                n_shrimp=INITIAL_STOCKING,
                mean_weight_g=INITIAL_WEIGHT_G + 0.01 * age_offset,  # slightly older
                initial_stocking=INITIAL_STOCKING,
                temperature_c=27.0,
                do_mg_L=7.5,
                tan_mg_L=0.02,
                ph=8.1,
                salinity_ppt=15.0,
                alkalinity_mg_L=120.0,
            ))
            self._vibrio_treatment[i] = {
                "probiotic_active": False,
                "antibiotic_active": False,
                "probiotic_days": 0,
            }

        self._state = EpisodeState(
            task_id=2,
            day=0,
            max_steps=self.max_steps,
            done=False,
            seed=self.seed,
            ponds=ponds,
            vannamei_price_usd_per_kg=VANNAMEI_PRICE_MEAN,
            vibrio_trigger_day=vibrio_trigger_day(self._rng),
        )
        self._cumulative_reward = 0.0

        return self._make_obs(
            reward=0.0,
            breakdown=RewardBreakdown(0.0, 0.0, 0.0, 0.0, 0.0),
            costs={"total": 0.0, "feed": 0.0, "energy": 0.0, "treatment": 0.0},
        )

    def step(self, action: SemiIntensiveFarmAction) -> SemiIntensiveFarmObs:
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._state.done:
            raise RuntimeError("Episode done — call reset()")

        s = self._state
        s.day += 1

        # Build lookup maps
        feed_map = {pf.pond_id: pf for pf in action.pond_feeds}
        aeration_map = action.aeration_allocation  # pond_id → fraction (sum ≤ 1.0)

        total_cost = 0.0
        total_revenue = 0.0
        pond_rewards = []

        # Check if Vibrio triggers today
        if s.day == s.vibrio_trigger_day and s.vibrio_pond_id == -1:
            s.vibrio_pond_id = int(self._rng.integers(0, NUM_PONDS))

        # Process each pond
        for pond in s.ponds:
            pid = pond.pond_id
            pf = feed_map.get(pid, PondFeedAction(pond_id=pid, feed_kg=5.0, frequency=4))
            aeration_frac = aeration_map.get(pid, 0.25)  # default equal share
            aeration_hours = aeration_frac * TOTAL_AERATION_CAPACITY / NUM_AERATORS

            if pond.fallow_status != FallowStatus.ACTIVE:
                pond.fallow_days_remaining -= 1
                if pond.fallow_days_remaining <= 0:
                    pond.fallow_status = FallowStatus.READY
                continue

            # Update temperature
            pond.temperature_c = seasonal_temperature(s.day, self._rng)

            # Disease: Vibrio
            vib = self._vibrio_treatment[pid]
            if pid == s.vibrio_pond_id and pond.disease_type == DiseaseType.NONE:
                if s.day >= s.vibrio_trigger_day:
                    pond.disease_type = DiseaseType.VIBRIO
                    pond.disease_severity = 0.05

            # Apply treatments
            if pid in action.probiotic_ponds:
                vib["probiotic_active"] = True
                vib["probiotic_days"] = 0
            if pid in action.antibiotic_ponds:
                vib["antibiotic_active"] = True
                s.antibiotic_used = True
                pond.export_compliance_flag = True

            # Advance Vibrio
            if pond.disease_type == DiseaseType.VIBRIO:
                pond.disease_severity = step_vibrio_severity(
                    pond.disease_severity,
                    vib["probiotic_active"],
                    vib["antibiotic_active"],
                    vib["probiotic_days"],
                )
                pond.days_since_infection += 1
                if vib["probiotic_active"]:
                    vib["probiotic_days"] += 1
                dead_vib = vibrio_mortality_count(pond.n_shrimp, pond.disease_severity, self._rng)
                pond.n_shrimp = max(0, pond.n_shrimp - dead_vib)

                # Cure if severity near zero
                if pond.disease_severity < 0.01:
                    pond.disease_type = DiseaseType.NONE
                    pond.disease_severity = 0.0

            # Stress + growth
            stress = sg.compute_stress_index(
                pond.do_mg_L, pond.tan_mg_L, pond.ph, pond.salinity_ppt, pond.alkalinity_mg_L
            )
            true_frac = true_consumption_fraction(
                pond.biomass_kg, pf.feed_kg, pond.disease_severity, stress
            )
            check = action.check_trays.get(pid, False)
            uneaten = uneaten_feed(pf.feed_kg, true_frac)

            demand = sg.estimate_feed_demand(pond.biomass_kg, pond.mean_weight_g, pond.temperature_c)
            feed_ratio = pf.feed_kg / max(demand, 0.01)
            W_old = pond.mean_weight_g
            W_new, _ = sg.daily_weight_gain_full(
                W_old, pond.temperature_c, feed_ratio,
                pond.biomass_kg, pond.area_m2, pond.disease_severity, stress
            )
            pond.mean_weight_g = W_new

            # Molting
            molt, pond.days_since_molt = sg.check_molt_event(
                pond.days_since_molt, pond.temperature_c, self._rng
            )
            if molt:
                pond.n_shrimp = max(0, pond.n_shrimp - sg.cannibalism_mortality(
                    pond.n_shrimp, pond.density_kg_m2
                ))

            # Partial harvest
            if action.partial_harvest and action.partial_harvest.pond_id == pid:
                ph_action = action.partial_harvest
                if pond.mean_weight_g >= ph_action.size_threshold_g:
                    harvest_n = int(pond.n_shrimp * ph_action.fraction)
                    harvest_kg = harvest_n * pond.mean_weight_g / 1000.0
                    rev = harvest_kg * s.vannamei_price_usd_per_kg * 1.0
                    pond.cumulative_revenue_usd += rev
                    pond.cumulative_biomass_harvested_kg += harvest_kg
                    total_revenue += rev
                    pond.n_shrimp -= harvest_n

            # Water quality
            lime_kg = action.lime_per_pond.get(pid, 0.0)
            exchange = action.water_exchange.get(pid, 0.05)
            pond.do_mg_L = wq.step_do(
                pond.do_mg_L, pond.temperature_c, pond.salinity_ppt,
                pond.biomass_kg, aeration_hours, pond.volume_m3,
                water_exchange_frac=exchange
            )
            pond.tan_mg_L = wq.step_tan(
                pond.tan_mg_L, pf.feed_kg, uneaten, pond.biomass_kg, pond.volume_m3, exchange
            )
            pond.alkalinity_mg_L = wq.step_alkalinity(
                pond.alkalinity_mg_L, pond.tan_mg_L,
                lime_kg_per_ha=lime_kg, pond_area_ha=pond.area_m2 / 10000.0,
                water_exchange_frac=exchange
            )
            pond.ph = wq.step_ph(pond.ph, pond.alkalinity_mg_L, pond.tan_mg_L)
            pond.salinity_ppt = wq.step_salinity(pond.salinity_ppt, pond.temperature_c)
            pond.h2s_risk = wq.step_h2s_risk(
                pond.h2s_risk, pf.feed_kg, pond.biomass_kg, pond.area_m2,
                pond.days_since_bottom_cleaned
            )
            pond.days_since_bottom_cleaned += 1
            pond.cumulative_feed_kg += pf.feed_kg

            # Economics
            pond_rev = daily_revenue_accrual(W_new, W_old, pond.n_shrimp, s.vannamei_price_usd_per_kg)
            total_revenue += pond_rev

            probiotic_count = 1 if pid in action.probiotic_ponds else 0
            antibiotic_count = 1 if pid in action.antibiotic_ponds else 0
            pond_costs = compute_daily_costs(
                pf.feed_kg, aeration_hours, lime_kg,
                probiotic_ponds=probiotic_count, antibiotic_ponds=antibiotic_count,
                tray_checks=1 if check else 0
            )
            total_cost += pond_costs["total"]

            # Per-pond reward
            mortality_rate = 0.0
            pond_r_growth = growth_reward(
                W_new, expected_weight(INITIAL_WEIGHT_G, s.day, pond.temperature_c), mortality_rate
            )
            pond_r_water = water_quality_reward(pond.do_mg_L, pond.tan_mg_L, pond.ph, pond.alkalinity_mg_L)
            pond_r_econ = economic_reward(pond_rev, pond_costs["total"], pond.fcr)
            pond_rewards.append((pond_r_growth, pond_r_water, pond_r_econ))

        # Farm-level reward
        n_active = len([p for p in s.ponds if p.fallow_status == FallowStatus.ACTIVE])
        if n_active > 0:
            avg_growth = sum(r[0] for r in pond_rewards) / n_active
            avg_water = sum(r[1] for r in pond_rewards) / n_active
            avg_econ = sum(r[2] for r in pond_rewards) / n_active
        else:
            avg_growth = avg_water = avg_econ = -1.0

        r_bio = biosecurity_reward(
            1.0 - min(1.0, s.ponds[s.vibrio_pond_id].disease_severity if s.vibrio_pond_id >= 0 else 0.0),
            s.antibiotic_used, 0, 0
        )

        s.cumulative_cost_usd += total_cost
        s.cumulative_revenue_usd += total_revenue
        s.vannamei_price_usd_per_kg = update_price(s.vannamei_price_usd_per_kg, s.day, self._rng, 60)

        total_r, breakdown = compute_total_reward(2, avg_growth, avg_water, avg_econ, r_bio)
        self._cumulative_reward += total_r

        # Episode end
        all_dead = all(p.n_shrimp == 0 for p in s.ponds)
        if s.day >= s.max_steps or all_dead:
            s.done = True

        return self._make_obs(
            total_r, breakdown,
            {"total": total_cost, "feed": total_cost * 0.6, "energy": total_cost * 0.2, "treatment": total_cost * 0.2},
        )

    def _make_obs(self, reward, breakdown, costs) -> SemiIntensiveFarmObs:
        s = self._state
        pond_obs = []
        for p in s.ponds:
            vib_signals = {}
            if p.disease_type == DiseaseType.VIBRIO:
                vib_signals = vibrio_observable_signals(p.disease_severity, p.days_since_infection)

            pond_obs.append(PondObs(
                pond_id=p.pond_id,
                n_shrimp=p.n_shrimp,
                mean_weight_g=round(p.mean_weight_g, 4),
                biomass_kg=round(p.biomass_kg, 3),
                survival_rate=round(p.survival_rate, 4),
                mortality_today=0,
                molt_event_today=False,
                temperature_c=round(p.temperature_c, 2),
                do_mg_L=round(p.do_mg_L, 3),
                tan_mg_L=round(p.tan_mg_L, 4),
                ph=round(p.ph, 3),
                salinity_ppt=round(p.salinity_ppt, 2),
                alkalinity_mg_L=round(p.alkalinity_mg_L, 1),
                h2s_risk_score=round(p.h2s_risk, 4),
                secchi_depth_cm=round(p.secchi_depth_cm if hasattr(p, 'secchi_depth_cm') else 35.0, 1),
                tray_consumption_fraction=None,
                fcr_cumulative=round(p.fcr, 3),
                feed_demand_estimate_kg=round(
                    sg.estimate_feed_demand(p.biomass_kg, p.mean_weight_g, p.temperature_c), 3
                ),
                density_kg_m2=round(p.density_kg_m2, 4),
                disease_type=p.disease_type,
                redness_score=vib_signals.get("redness_score", 0.0),
                export_compliance_flag=p.export_compliance_flag,
            ))

        total_biomass = sum(p.biomass_kg for p in s.ponds)
        return SemiIntensiveFarmObs(
            day=s.day,
            done=s.done,
            reward=reward,
            reward_breakdown=breakdown,
            grade=compute_grade(self._cumulative_reward, s.day),
            ponds=pond_obs,
            aeration_capacity_remaining=0.0,
            total_biomass_kg=round(total_biomass, 2),
            feed_cost_today_usd=round(costs.get("feed", 0.0), 2),
            energy_cost_today_usd=round(costs.get("energy", 0.0), 2),
            treatment_cost_today_usd=round(costs.get("treatment", 0.0), 2),
            cumulative_cost_usd=round(s.cumulative_cost_usd, 2),
            vannamei_price_usd_per_kg=round(s.vannamei_price_usd_per_kg, 3),
            antibiotic_used_this_episode=s.antibiotic_used,
        )

    @property
    def state(self) -> EpisodeState | None:
        return self._state

    @property
    def grade(self) -> float:
        return compute_grade(self._cumulative_reward, self._state.day if self._state else 0)
