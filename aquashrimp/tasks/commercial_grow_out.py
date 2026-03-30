"""Task 3: CommercialGrowOut — 10 ponds across 2 farm sites.

Difficulty: Hard | Max steps: 90 days
Key challenges:
  - WSSV outbreak (no treatment, containment only)
  - Market price timing (20g vs 25g harvest window)
  - Export compliance (antibiotics = export ban risk)
  - Biosecurity (shared water source at Site A)
  - Weather events (rain/heat waves)
  - Regulatory reporting (WSSV within 48h)
"""
from __future__ import annotations
import numpy as np
from aquashrimp.models.actions import CommercialGrowOutAction, PondFeedAction, HarvestAction
from aquashrimp.models.observations import CommercialGrowOutObs, PondObs, RewardBreakdown
from aquashrimp.models.state import PondState, EpisodeState
from aquashrimp.models.enums import WeatherEvent, DiseaseType, FallowStatus, HarvestType, TreatmentType
from aquashrimp.simulation import shrimp_growth as sg
from aquashrimp.simulation import water_quality as wq
from aquashrimp.simulation.feeding_trays import (
    true_consumption_fraction, observe_tray, uneaten_feed, wssv_early_warning_fraction
)
from aquashrimp.simulation.weather import determine_weather_event, weather_impacts
from aquashrimp.simulation.disease import (
    wssv_trigger_day, wssv_consumption_signal, wssv_mortality_count,
    wssv_spread_check, emergency_harvest_recovery, neighbor_outbreak_trigger,
    vibrio_trigger_day, step_vibrio_severity, vibrio_mortality_count,
    vibrio_observable_signals
)
from aquashrimp.simulation.market import (
    compute_daily_costs, daily_revenue_accrual, compute_harvest_revenue,
    update_price, VANNAMEI_PRICE_MEAN
)
from aquashrimp.rewards.reward_calculator import (
    compute_total_reward, growth_reward, water_quality_reward,
    economic_reward, biosecurity_reward, expected_weight, compute_grade
)

# ── Farm configuration ────────────────────────────────────────────────────────
NUM_PONDS = 10
SITE_A_PONDS = [0, 1, 2, 3, 4]   # coastal, salinity 15 ppt, shared water source
SITE_B_PONDS = [5, 6, 7, 8, 9]   # inland, salinity 5 ppt, separate water
POND_AREA_M2 = 20_000.0            # 2 ha each
POND_VOLUME_M3 = 30_000.0
INITIAL_STOCKING = 3_000_000       # 150 PL/m² × 20,000 m²
INITIAL_WEIGHT_G = 0.05
SITE_A_SALINITY = 15.0
SITE_B_SALINITY = 5.0
FALLOW_PERIOD_DAYS = 30


class CommercialGrowOutEnvironment:
    """Task 3: Commercial grow-out farm environment."""

    task_id = 3
    max_steps = 90

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._state: EpisodeState | None = None
        self._vib_treatment: dict[int, dict] = {}
        self._heat_wave_active = False
        self._heat_wave_remaining = 0
        self._cumulative_reward: float = 0.0

    def reset(self) -> CommercialGrowOutObs:
        self._rng = np.random.default_rng(self.seed)
        self._heat_wave_active = False
        self._heat_wave_remaining = 0
        self._vib_treatment = {}

        ponds = []
        for i in range(NUM_PONDS):
            salinity = SITE_A_SALINITY if i in SITE_A_PONDS else SITE_B_SALINITY
            ponds.append(PondState(
                pond_id=i,
                area_m2=POND_AREA_M2,
                volume_m3=POND_VOLUME_M3,
                n_shrimp=INITIAL_STOCKING,
                mean_weight_g=INITIAL_WEIGHT_G,
                initial_stocking=INITIAL_STOCKING,
                temperature_c=27.0,
                do_mg_L=7.5,
                tan_mg_L=0.02,
                ph=8.1,
                salinity_ppt=salinity,
                alkalinity_mg_L=120.0,
            ))
            self._vib_treatment[i] = {
                "probiotic_active": False,
                "antibiotic_active": False,
                "probiotic_days": 0,
            }

        self._state = EpisodeState(
            task_id=3,
            day=0,
            max_steps=self.max_steps,
            done=False,
            seed=self.seed,
            ponds=ponds,
            vannamei_price_usd_per_kg=VANNAMEI_PRICE_MEAN,
            wssv_trigger_day=wssv_trigger_day(self._rng, self.max_steps),
            neighbor_outbreak=False,
            regulatory_report_submitted=False,
        )
        self._cumulative_reward = 0.0

        return self._make_obs(
            0.0, RewardBreakdown(0.0, 0.0, 0.0, 0.0, 0.0),
            {"total": 0.0, "feed": 0.0, "energy": 0.0, "treatment": 0.0, "inspection": 0.0, "biosecurity": 0.0},
            weather=WeatherEvent.NONE, wssv_confirmed=[],
        )

    def step(self, action: CommercialGrowOutAction) -> CommercialGrowOutObs:
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._state.done:
            raise RuntimeError("Episode done — call reset()")

        s = self._state
        s.day += 1

        # ── Weather ──────────────────────────────────────────────────────────
        weather_event, rainfall_cm, heat_spike, self._heat_wave_active, self._heat_wave_remaining = (
            determine_weather_event(s.day, self._rng, self._heat_wave_active, self._heat_wave_remaining, 3)
        )
        w_impacts = weather_impacts(weather_event, rainfall_cm, heat_spike)

        # ── Neighbor outbreak ─────────────────────────────────────────────────
        if not s.neighbor_outbreak and neighbor_outbreak_trigger(s.day, self._rng):
            s.neighbor_outbreak = True

        # ── WSSV primary trigger ──────────────────────────────────────────────
        wssv_source_pond = None
        for p in s.ponds:
            if (p.fallow_status == FallowStatus.ACTIVE
                    and p.disease_type == DiseaseType.NONE
                    and s.day == s.wssv_trigger_day):
                p.disease_type = DiseaseType.WSSV
                p.disease_severity = 0.01
                p.days_since_infection = 1
                wssv_source_pond = p.pond_id
                break

        # ── Regulatory report overdue ──────────────────────────────────────────
        overdue_days = 0
        if s.regulatory_report_due_day > 0 and not s.regulatory_report_submitted:
            if s.day > s.regulatory_report_due_day:
                overdue_days = s.day - s.regulatory_report_due_day

        if action.regulatory_report and not s.regulatory_report_submitted:
            s.regulatory_report_submitted = True

        # ── Build lookup ──────────────────────────────────────────────────────
        feed_map = {pf.pond_id: pf for pf in action.pond_feeds}
        treatment_map = {t.pond_id: t for t in action.treatments}
        harvest_map = {h.pond_id: h for h in action.harvests}

        total_cost = 0.0
        total_revenue = 0.0
        pond_rewards = []
        wssv_confirmed_ponds = []
        wssv_spread_count = 0

        # ── Per-pond processing ──────────────────────────────────────────────
        for pond in s.ponds:
            pid = pond.pond_id

            # Fallow tracking
            if pond.fallow_status == FallowStatus.FALLOWING:
                pond.fallow_days_remaining -= 1
                if pond.fallow_days_remaining <= 0:
                    pond.fallow_status = FallowStatus.READY
                continue

            if pond.fallow_status == FallowStatus.READY:
                continue

            # ── Pond disinfection (post-WSSV) ────────────────────────────────
            if pid in action.disinfect_pond and pond.disease_type == DiseaseType.WSSV:
                pond.fallow_status = FallowStatus.FALLOWING
                pond.fallow_days_remaining = FALLOW_PERIOD_DAYS
                pond.n_shrimp = 0
                pond.disease_type = DiseaseType.NONE
                from aquashrimp.simulation.market import DISINFECTION_COST_PER_POND
                total_cost += DISINFECTION_COST_PER_POND
                continue

            # ── WSSV spread check ─────────────────────────────────────────────
            if pond.disease_type == DiseaseType.NONE:
                water_from_infected = False
                if pid in SITE_A_PONDS and wssv_source_pond in SITE_A_PONDS:
                    exch = action.water_exchange.get(pid, 0.05)
                    water_from_infected = exch > 0.02

                spread = wssv_spread_check(
                    self._rng, water_from_infected, action.biosecurity_measure, s.neighbor_outbreak
                )
                if spread:
                    pond.disease_type = DiseaseType.WSSV
                    pond.disease_severity = 0.01
                    pond.days_since_infection = 1
                    wssv_spread_count += 1

            # ── WSSV confirmation ─────────────────────────────────────────────
            if pond.disease_type == DiseaseType.WSSV:
                if pid in action.pond_inspections:
                    pond.wssv_confirmed = True
                    # Set report due within 2 days
                    if s.regulatory_report_due_day < 0:
                        s.regulatory_report_due_day = s.day + 2
                if pond.wssv_confirmed:
                    wssv_confirmed_ponds.append(pid)

            # ── Temperature + weather ─────────────────────────────────────────
            base_temp = sg.f_temp.__module__  # just for import check
            from aquashrimp.simulation.weather import seasonal_temperature
            pond.temperature_c = seasonal_temperature(s.day, self._rng) + w_impacts["temp_delta"]

            # ── Treatment ─────────────────────────────────────────────────────
            vib = self._vib_treatment[pid]
            if pid in action.pond_inspections:
                pass  # already handled above
            if pid in [t.pond_id for t in action.treatments]:
                t = treatment_map[pid]
                if t.treatment == TreatmentType.PROBIOTIC:
                    vib["probiotic_active"] = True
                    vib["probiotic_days"] = 0
                elif t.treatment == TreatmentType.ANTIBIOTIC:
                    vib["antibiotic_active"] = True
                    s.antibiotic_used = True
                    pond.export_compliance_flag = True

            # ── WSSV mortality (accelerating) ────────────────────────────────
            if pond.disease_type == DiseaseType.WSSV:
                dead_wssv = wssv_mortality_count(pond.n_shrimp, pond.days_since_infection, self._rng)
                pond.n_shrimp = max(0, pond.n_shrimp - dead_wssv)
                pond.days_since_infection += 1
                pond.disease_severity = min(1.0, 0.2 * pond.days_since_infection)

            # ── Emergency harvest ─────────────────────────────────────────────
            if pid in harvest_map:
                h = harvest_map[pid]
                if h.harvest_type == HarvestType.EMERGENCY:
                    save_frac, rev_penalty = emergency_harvest_recovery(pond.days_since_infection)
                    harvest_kg = pond.biomass_kg * save_frac
                    rev = compute_harvest_revenue(
                        harvest_kg, pond.mean_weight_g, s.vannamei_price_usd_per_kg, emergency=True
                    )
                    pond.cumulative_revenue_usd += rev
                    pond.cumulative_biomass_harvested_kg += harvest_kg
                    total_revenue += rev
                    pond.n_shrimp = 0
                    pond.fallow_status = FallowStatus.FALLOWING
                    pond.fallow_days_remaining = FALLOW_PERIOD_DAYS
                    continue

                elif h.harvest_type == HarvestType.FULL:
                    rev = compute_harvest_revenue(
                        pond.biomass_kg, pond.mean_weight_g, s.vannamei_price_usd_per_kg
                    )
                    pond.cumulative_revenue_usd += rev
                    total_revenue += rev
                    pond.n_shrimp = 0
                    pond.fallow_status = FallowStatus.FALLOWING
                    pond.fallow_days_remaining = FALLOW_PERIOD_DAYS
                    continue

            # ── Stress and growth ─────────────────────────────────────────────
            stress = sg.compute_stress_index(
                pond.do_mg_L, pond.tan_mg_L, pond.ph, pond.salinity_ppt, pond.alkalinity_mg_L
            )
            stress = min(1.0, stress + (1.0 - w_impacts["appetite_factor"]) * 0.3)

            pf = feed_map.get(pid, PondFeedAction(pond_id=pid, feed_kg=50.0, frequency=4))
            true_frac = true_consumption_fraction(
                pond.biomass_kg, pf.feed_kg, pond.disease_severity, stress,
                appetite_factor=w_impacts["appetite_factor"]
            )

            # WSSV overrides consumption signal
            if pond.disease_type == DiseaseType.WSSV:
                true_frac = wssv_consumption_signal(pond.days_since_infection, true_frac)

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

            # Weather osmotic stress mortality
            if weather_event == WeatherEvent.HEAVY_RAIN and pond.salinity_ppt < 5.0:
                rain_dead = int(pond.n_shrimp * float(self._rng.uniform(0.005, 0.02)))
                pond.n_shrimp = max(0, pond.n_shrimp - rain_dead)

            # Water quality
            source_salinity = SITE_A_SALINITY if pid in SITE_A_PONDS else SITE_B_SALINITY
            lime_kg = action.lime_per_pond.get(pid, 0.0)
            exchange = action.water_exchange.get(pid, 0.03)
            aeration_hrs = action.aeration_per_pond.get(pid, 20.0)

            pond.do_mg_L = wq.step_do(
                pond.do_mg_L, pond.temperature_c, pond.salinity_ppt,
                pond.biomass_kg, aeration_hrs, pond.volume_m3,
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
            pond.ph = wq.step_ph(
                pond.ph, pond.alkalinity_mg_L, pond.tan_mg_L,
                rainfall_cm=rainfall_cm, water_exchange_frac=exchange
            )
            pond.salinity_ppt = wq.step_salinity(
                pond.salinity_ppt, pond.temperature_c,
                rainfall_cm=rainfall_cm,
                water_exchange_frac=exchange,
                source_salinity_ppt=source_salinity
            )
            pond.h2s_risk = wq.step_h2s_risk(
                pond.h2s_risk, pf.feed_kg, pond.biomass_kg, pond.area_m2,
                pond.days_since_bottom_cleaned
            )
            pond.days_since_bottom_cleaned += 1
            pond.cumulative_feed_kg += pf.feed_kg

            # Economics
            pond_rev = daily_revenue_accrual(W_new, W_old, pond.n_shrimp, s.vannamei_price_usd_per_kg)
            total_revenue += pond_rev

            trt = treatment_map.get(pid)
            probiotic_c = 1 if (trt and trt.treatment == TreatmentType.PROBIOTIC) else 0
            antibiotic_c = 1 if (trt and trt.treatment == TreatmentType.ANTIBIOTIC) else 0
            pond_costs = compute_daily_costs(
                pf.feed_kg, aeration_hrs, lime_kg,
                probiotic_ponds=probiotic_c, antibiotic_ponds=antibiotic_c,
                inspection_ponds=1 if pid in action.pond_inspections else 0,
                tray_checks=1 if check else 0,
            )
            total_cost += pond_costs["total"]

            pond_r_growth = growth_reward(
                W_new, expected_weight(INITIAL_WEIGHT_G, s.day, pond.temperature_c), 0.0
            )
            pond_r_water = water_quality_reward(pond.do_mg_L, pond.tan_mg_L, pond.ph, pond.alkalinity_mg_L)
            pond_r_econ = economic_reward(pond_rev, pond_costs["total"], pond.fcr)
            pond_rewards.append((pond_r_growth, pond_r_water, pond_r_econ))

        # Biosecurity cost
        if action.biosecurity_measure:
            from aquashrimp.simulation.market import BIOSECURITY_DAILY_COST
            total_cost += BIOSECURITY_DAILY_COST

        # Farm reward
        n_active = max(1, len(pond_rewards))
        avg_growth = sum(r[0] for r in pond_rewards) / n_active
        avg_water = sum(r[1] for r in pond_rewards) / n_active
        avg_econ = sum(r[2] for r in pond_rewards) / n_active
        r_bio = biosecurity_reward(
            max(0.0, 1.0 - wssv_spread_count * 0.4),
            s.antibiotic_used, wssv_spread_count, overdue_days
        )

        s.cumulative_cost_usd += total_cost
        s.cumulative_revenue_usd += total_revenue
        s.vannamei_price_usd_per_kg = update_price(s.vannamei_price_usd_per_kg, s.day, self._rng, 90)

        total_r, breakdown = compute_total_reward(3, avg_growth, avg_water, avg_econ, r_bio)
        self._cumulative_reward += total_r

        all_done = all(
            p.n_shrimp == 0 or p.fallow_status != FallowStatus.ACTIVE
            for p in s.ponds
        )
        if s.day >= s.max_steps or all_done:
            s.done = True

        return self._make_obs(
            total_r, breakdown,
            {"total": total_cost, "feed": total_cost * 0.55, "energy": total_cost * 0.2,
             "treatment": total_cost * 0.1, "inspection": total_cost * 0.05,
             "biosecurity": total_cost * 0.1},
            weather=weather_event,
            wssv_confirmed=wssv_confirmed_ponds,
        )

    def _make_obs(self, reward, breakdown, costs, weather, wssv_confirmed) -> CommercialGrowOutObs:
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
                secchi_depth_cm=round(getattr(p, 'secchi_depth_cm', 35.0), 1),
                tray_consumption_fraction=None,
                fcr_cumulative=round(p.fcr, 3),
                feed_demand_estimate_kg=round(
                    sg.estimate_feed_demand(p.biomass_kg, p.mean_weight_g, p.temperature_c), 3
                ),
                density_kg_m2=round(p.density_kg_m2, 4),
                disease_type=p.disease_type,
                redness_score=vib_signals.get("redness_score", 0.0),
                swimming_behavior_score=max(0.0, 1.0 - p.disease_severity) if p.disease_type == DiseaseType.WSSV else 1.0,
                white_spots_visible=p.wssv_confirmed and p.pond_id in wssv_confirmed,
                disease_severity=round(p.disease_severity, 3),
                fallow_status=p.fallow_status,
                fallow_days_remaining=p.fallow_days_remaining,
                export_compliance_flag=p.export_compliance_flag,
            ))

        overdue_days = 0
        if s.regulatory_report_due_day > 0 and not s.regulatory_report_submitted:
            overdue_days = max(0, s.day - s.regulatory_report_due_day)

        return CommercialGrowOutObs(
            day=s.day,
            done=s.done,
            reward=reward,
            reward_breakdown=breakdown,
            grade=compute_grade(self._cumulative_reward, s.day),
            ponds=pond_obs,
            total_biomass_kg=round(sum(p.biomass_kg for p in s.ponds), 2),
            feed_cost_today_usd=round(costs.get("feed", 0.0), 2),
            energy_cost_today_usd=round(costs.get("energy", 0.0), 2),
            treatment_cost_today_usd=round(costs.get("treatment", 0.0), 2),
            inspection_cost_today_usd=round(costs.get("inspection", 0.0), 2),
            biosecurity_cost_today_usd=round(costs.get("biosecurity", 0.0), 2),
            cumulative_cost_usd=round(s.cumulative_cost_usd, 2),
            cumulative_revenue_usd=round(s.cumulative_revenue_usd, 2),
            vannamei_price_usd_per_kg=round(s.vannamei_price_usd_per_kg, 3),
            feed_price_usd_per_kg=1.2,
            neighbor_outbreak=s.neighbor_outbreak,
            wssv_confirmed_ponds=wssv_confirmed,
            regulatory_report_overdue=overdue_days > 0,
            report_overdue_days=overdue_days,
            antibiotic_used_this_episode=s.antibiotic_used,
            export_ban_risk=s.antibiotic_used,
            weather_event=weather,
        )

    @property
    def state(self) -> EpisodeState | None:
        return self._state

    @property
    def grade(self) -> float:
        return compute_grade(self._cumulative_reward, self._state.day if self._state else 0)
