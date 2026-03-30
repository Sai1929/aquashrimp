"""Task 1: NurseryPond — single 0.1 ha L. vannamei nursery pond.

Difficulty: Easy | Max steps: 30 days
Key challenges: nighttime DO crash, feed tray monitoring, ammonia management.
Target: grow shrimp 0.05g → 12–15g in 30 days with FCR < 1.6.
"""
from __future__ import annotations
import numpy as np
from aquashrimp.models.actions import NurseryPondAction
from aquashrimp.models.observations import NurseryPondObs, RewardBreakdown
from aquashrimp.models.state import PondState, EpisodeState
from aquashrimp.models.enums import WeatherEvent, DiseaseType
from aquashrimp.simulation import shrimp_growth as sg
from aquashrimp.simulation import water_quality as wq
from aquashrimp.simulation.feeding_trays import (
    true_consumption_fraction, observe_tray, uneaten_feed
)
from aquashrimp.simulation.weather import seasonal_temperature
from aquashrimp.simulation.market import (
    compute_daily_costs, daily_revenue_accrual, update_price,
    FEED_PRICE_USD_PER_KG, VANNAMEI_PRICE_MEAN
)
from aquashrimp.rewards.reward_calculator import (
    compute_total_reward, growth_reward, water_quality_reward,
    economic_reward, biosecurity_reward, expected_weight, compute_grade
)

# ── Pond configuration ────────────────────────────────────────────────────────
POND_AREA_M2 = 1000.0
POND_VOLUME_M3 = 1500.0  # ~1.5m depth
INITIAL_STOCKING = 100_000
INITIAL_WEIGHT_G = 0.05   # PL15 post-larvae


class NurseryPondEnvironment:
    """Task 1: Single nursery pond environment."""

    task_id = 1
    max_steps = 30

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._state: EpisodeState | None = None
        self._pond: PondState | None = None
        self._cumulative_reward: float = 0.0

    def reset(self) -> NurseryPondObs:
        """Reset episode with deterministic seed."""
        self._rng = np.random.default_rng(self.seed)

        pond = PondState(
            pond_id=0,
            area_m2=POND_AREA_M2,
            volume_m3=POND_VOLUME_M3,
            n_shrimp=INITIAL_STOCKING,
            mean_weight_g=INITIAL_WEIGHT_G,
            initial_stocking=INITIAL_STOCKING,
            temperature_c=27.0,
            do_mg_L=7.5,
            tan_mg_L=0.02,
            ph=8.1,
            salinity_ppt=15.0,
            alkalinity_mg_L=120.0,
            h2s_risk=0.0,
            secchi_depth_cm=35.0,
        )

        self._state = EpisodeState(
            task_id=1,
            day=0,
            max_steps=self.max_steps,
            done=False,
            seed=self.seed,
            ponds=[pond],
            vannamei_price_usd_per_kg=VANNAMEI_PRICE_MEAN,
            cumulative_cost_usd=0.0,
            cumulative_revenue_usd=0.0,
        )
        self._pond = pond
        self._cumulative_reward = 0.0

        return self._make_obs(
            reward=0.0,
            reward_breakdown=RewardBreakdown(0.0, 0.0, 0.0, 0.0, 0.0),
            tray_fraction=None,
            costs={"total": 0.0, "feed": 0.0, "energy": 0.0},
        )

    def step(self, action: NurseryPondAction) -> NurseryPondObs:
        """Execute one day of farm management."""
        if self._state is None or self._pond is None:
            raise RuntimeError("Call reset() before step()")
        if self._state.done:
            raise RuntimeError("Episode done — call reset()")

        # Validate and clip action
        action = self._clip_action(action)

        s = self._state
        p = self._pond
        s.day += 1

        # 1. Update temperature (seasonal)
        p.temperature_c = seasonal_temperature(s.day, self._rng)

        # 2. Compute stress index
        stress = sg.compute_stress_index(
            p.do_mg_L, p.tan_mg_L, p.ph, p.salinity_ppt, p.alkalinity_mg_L
        )

        # 3. Feeding tray
        true_frac = true_consumption_fraction(
            p.biomass_kg, action.feed_kg, 0.0, stress
        )
        tray_obs = observe_tray(true_frac, self._rng) if action.check_feeding_trays else None
        uneaten = uneaten_feed(action.feed_kg, true_frac)

        # 4. Feed ratio for growth
        demand = sg.estimate_feed_demand(p.biomass_kg, p.mean_weight_g, p.temperature_c)
        feed_ratio = action.feed_kg / max(demand, 0.01)

        # 5. Shrimp growth
        W_old = p.mean_weight_g
        W_new, dwg = sg.daily_weight_gain_full(
            W_old, p.temperature_c, feed_ratio,
            p.biomass_kg, p.area_m2, 0.0, stress
        )
        p.mean_weight_g = W_new

        # 6. Molting
        molt, p.days_since_molt = sg.check_molt_event(p.days_since_molt, p.temperature_c, self._rng)
        if molt:
            dead_molt = sg.cannibalism_mortality(p.n_shrimp, p.density_kg_m2)
        else:
            dead_molt = 0

        # 7. Background mortality (DO + ammonia stress)
        dead_stress = self._compute_stress_mortality(p, stress)
        dead_today = dead_molt + dead_stress
        p.n_shrimp = max(0, p.n_shrimp - dead_today)
        p.days_since_bottom_cleaned += 1

        # 8. Water quality update
        p.do_mg_L = wq.step_do(
            p.do_mg_L, p.temperature_c, p.salinity_ppt,
            p.biomass_kg, action.aeration_hours, p.volume_m3,
            is_daytime=True, water_exchange_frac=action.water_exchange_frac
        )
        p.tan_mg_L = wq.step_tan(
            p.tan_mg_L, action.feed_kg, uneaten,
            p.biomass_kg, p.volume_m3, action.water_exchange_frac
        )
        p.alkalinity_mg_L = wq.step_alkalinity(
            p.alkalinity_mg_L, p.tan_mg_L,
            lime_kg_per_ha=action.lime_application_kg,
            pond_area_ha=p.area_m2 / 10000.0,
            water_exchange_frac=action.water_exchange_frac
        )
        p.ph = wq.step_ph(
            p.ph, p.alkalinity_mg_L, p.tan_mg_L,
            is_daytime=True, water_exchange_frac=action.water_exchange_frac
        )
        p.salinity_ppt = wq.step_salinity(
            p.salinity_ppt, p.temperature_c,
            water_exchange_frac=action.water_exchange_frac
        )
        p.h2s_risk = wq.step_h2s_risk(
            p.h2s_risk, action.feed_kg, p.biomass_kg,
            p.area_m2, p.days_since_bottom_cleaned
        )
        p.secchi_depth_cm = wq.compute_secchi_depth(
            p.tan_mg_L, p.salinity_ppt, p.biomass_kg, p.volume_m3, action.feed_kg
        )
        p.cumulative_feed_kg += action.feed_kg

        # 9. Economics
        s.vannamei_price_usd_per_kg = update_price(
            s.vannamei_price_usd_per_kg, s.day, self._rng, self.max_steps
        )
        costs = compute_daily_costs(
            action.feed_kg,
            action.aeration_hours,
            action.lime_application_kg * p.area_m2 / 10000.0,
            tray_checks=1 if action.check_feeding_trays else 0,
        )
        s.cumulative_cost_usd += costs["total"]
        revenue = daily_revenue_accrual(W_new, W_old, p.n_shrimp, s.vannamei_price_usd_per_kg)
        s.cumulative_revenue_usd += revenue

        # 10. Reward
        mortality_rate = dead_today / max(p.initial_stocking, 1)
        r_growth = growth_reward(
            W_new,
            expected_weight(INITIAL_WEIGHT_G, s.day, p.temperature_c),
            mortality_rate
        )
        r_water = water_quality_reward(p.do_mg_L, p.tan_mg_L, p.ph, p.alkalinity_mg_L)
        r_econ = economic_reward(revenue, costs["total"], p.fcr)
        r_bio = biosecurity_reward(1.0, False, 0, 0)
        total_r, breakdown = compute_total_reward(1, r_growth, r_water, r_econ, r_bio)
        self._cumulative_reward += total_r

        # 11. Check episode end
        if s.day >= s.max_steps or p.survival_rate < 0.10:
            s.done = True

        return self._make_obs(total_r, breakdown, tray_obs, costs)

    def _clip_action(self, action: NurseryPondAction) -> NurseryPondAction:
        return NurseryPondAction(
            feed_kg=float(np.clip(action.feed_kg, 0.0, 50.0)),
            feeding_frequency=int(np.clip(action.feeding_frequency, 2, 6)),
            aeration_hours=float(np.clip(action.aeration_hours, 0.0, 24.0)),
            water_exchange_frac=float(np.clip(action.water_exchange_frac, 0.0, 0.15)),
            check_feeding_trays=bool(action.check_feeding_trays),
            lime_application_kg=float(np.clip(action.lime_application_kg, 0.0, 20.0)),
        )

    def _compute_stress_mortality(self, p: PondState, stress: float) -> int:
        """Background mortality from water quality stress."""
        base_rate = 0.0
        if p.do_mg_L < 2.0:
            base_rate += 0.05  # severe hypoxia
        elif p.do_mg_L < 3.0:
            base_rate += 0.02
        elif p.do_mg_L < 4.0:
            base_rate += 0.005
        if p.tan_mg_L > 0.5:
            base_rate += 0.015
        elif p.tan_mg_L > 0.2:
            base_rate += 0.005
        if stress > 0.5:
            base_rate += 0.01
        n_dead = int(p.n_shrimp * base_rate)
        n_dead += self._rng.poisson(max(0, p.n_shrimp * base_rate * 0.1))
        return min(n_dead, p.n_shrimp)

    def _make_obs(
        self,
        reward: float,
        reward_breakdown: RewardBreakdown,
        tray_fraction: float | None,
        costs: dict,
    ) -> NurseryPondObs:
        s = self._state
        p = self._pond
        demand = sg.estimate_feed_demand(p.biomass_kg, p.mean_weight_g, p.temperature_c)
        return NurseryPondObs(
            day=s.day,
            done=s.done,
            reward=reward,
            reward_breakdown=reward_breakdown,
            grade=compute_grade(self._cumulative_reward, s.day),
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
            secchi_depth_cm=round(p.secchi_depth_cm, 1),
            tray_consumption_fraction=round(tray_fraction, 3) if tray_fraction is not None else None,
            fcr_cumulative=round(p.fcr, 3),
            feed_demand_estimate_kg=round(demand, 3),
            feed_cost_today_usd=round(costs.get("feed", 0.0), 2),
            energy_cost_today_usd=round(costs.get("energy", 0.0), 2),
            cumulative_cost_usd=round(s.cumulative_cost_usd, 2),
        )

    @property
    def state(self) -> EpisodeState | None:
        return self._state

    @property
    def grade(self) -> float:
        return compute_grade(self._cumulative_reward, self._state.day if self._state else 0)
