"""Shrimp growth model for Litopenaeus vannamei.

Uses Daily Weight Gain (DWG) — NOT the fish TGC model.
All parameter values are calibrated to published L. vannamei literature.
"""
from __future__ import annotations
import math
import numpy as np

# ── Growth constants ─────────────────────────────────────────────────────────
DWG_MAX_G_PER_DAY = 0.25   # Maximum daily weight gain under ideal conditions
FCR_TARGET = 1.4            # Industry benchmark for L. vannamei

# ── Temperature factor (Gaussian, peak at 27°C) ───────────────────────────────
T_OPT = 27.0     # °C optimal temperature
T_SIGMA = 3.0    # °C width of Gaussian

# ── Molting constants ─────────────────────────────────────────────────────────
MOLT_BASE_PERIOD = 7.5   # days at optimal temperature
MOLT_TEMP_COEFF = 0.15   # days reduction per °C
SOFT_SHELL_HOURS = 9.0   # average soft-shell vulnerability window (hours)
CANNIBALISM_RATE = 0.001  # fraction per molt event × density_factor


def f_temp(temperature_c: float) -> float:
    """Temperature factor: Gaussian centered at 27°C."""
    return math.exp(-0.5 * ((temperature_c - T_OPT) / T_SIGMA) ** 2)


def f_feed(feed_ratio: float) -> float:
    """Feeding factor based on actual vs. estimated demand.

    feed_ratio = feed_fed_kg / estimated_feed_demand_kg
    """
    return min(1.0, feed_ratio / 0.8) ** 0.6


def f_density(biomass_kg: float, area_m2: float) -> float:
    """Density-dependent competition factor."""
    density_kg_m2 = biomass_kg / area_m2
    return max(0.4, 1.0 - 0.003 * (density_kg_m2 - 2.0))


def f_health(disease_severity: float, stress_index: float) -> float:
    """Health factor from disease and environmental stress."""
    return max(0.0, 1.0 - 0.3 * disease_severity - 0.2 * stress_index)


def daily_weight_gain(
    W_old: float,
    temp: float,
    feed_ratio: float,
    density_kg_m2: float,
    disease_severity: float = 0.0,
    stress_index: float = 0.0,
) -> float:
    """Compute new mean weight after one day.

    Args:
        W_old: Current mean shrimp weight (grams)
        temp: Water temperature (°C)
        feed_ratio: feed_fed_kg / estimated_feed_demand_kg
        density_kg_m2: Current biomass density (kg/m²)
        disease_severity: Disease severity index (0–1)
        stress_index: Environmental stress index (0–1, from water quality)

    Returns:
        W_new: New mean weight (grams)
    """
    # Compute area from density — if density provided directly, create dummy area
    area_dummy = 1.0
    biomass_dummy = density_kg_m2 * area_dummy

    dwg = (
        DWG_MAX_G_PER_DAY
        * f_temp(temp)
        * f_feed(feed_ratio)
        * f_density(biomass_dummy, area_dummy)
        * f_health(disease_severity, stress_index)
    )
    return W_old + max(0.0, dwg)


def daily_weight_gain_full(
    W_old: float,
    temp: float,
    feed_ratio: float,
    biomass_kg: float,
    area_m2: float,
    disease_severity: float = 0.0,
    stress_index: float = 0.0,
) -> tuple[float, float]:
    """Compute new mean weight and the DWG value.

    Args:
        W_old: Current mean weight (grams)
        temp: Water temperature (°C)
        feed_ratio: feed_fed / demand
        biomass_kg: Total biomass
        area_m2: Pond area
        disease_severity: 0–1
        stress_index: 0–1

    Returns:
        (W_new, dwg_today) — new weight and daily weight gain achieved
    """
    dwg = (
        DWG_MAX_G_PER_DAY
        * f_temp(temp)
        * f_feed(feed_ratio)
        * f_density(biomass_kg, area_m2)
        * f_health(disease_severity, stress_index)
    )
    dwg = max(0.0, dwg)
    return W_old + dwg, dwg


def estimate_feed_demand(biomass_kg: float, mean_weight_g: float, temp: float) -> float:
    """Estimate daily feed demand based on FCR target and expected growth.

    This is the formula agents can use to plan feed amounts.
    Returns kg of feed needed per day.
    """
    if mean_weight_g <= 0 or biomass_kg <= 0:
        return 0.1
    temp_factor = f_temp(temp)
    return FCR_TARGET * biomass_kg * DWG_MAX_G_PER_DAY * temp_factor / mean_weight_g


def inter_molt_period(temperature_c: float) -> float:
    """Inter-molt period in days (temperature dependent)."""
    return max(3.0, MOLT_BASE_PERIOD - MOLT_TEMP_COEFF * temperature_c)


def check_molt_event(
    days_since_molt: float,
    temperature_c: float,
    rng: np.random.Generator,
) -> tuple[bool, float]:
    """Determine if molting occurs today.

    Returns:
        (molt_occurred, new_days_since_molt)
    """
    period = inter_molt_period(temperature_c)
    if days_since_molt >= period:
        # Add slight stochasticity (±1 day)
        jitter = rng.uniform(-0.5, 0.5)
        if days_since_molt >= period + jitter:
            return True, 0.0
    return False, days_since_molt + 1.0


def cannibalism_mortality(n_shrimp: int, density_kg_m2: float) -> int:
    """Mortality from cannibalism during post-molt soft-shell window."""
    density_factor = min(2.0, density_kg_m2 / 2.0)
    rate = CANNIBALISM_RATE * density_factor
    return max(0, int(n_shrimp * rate))


def disease_mortality(
    n_shrimp: int,
    disease_type: str,
    severity: float,
    days_since_infection: int,
    rng: np.random.Generator,
) -> int:
    """Compute daily mortality from disease.

    Returns number of shrimp that die today.
    """
    if severity <= 0:
        return 0

    if disease_type == "vibrio":
        base_rate = 0.008 * severity ** 2
        n_dead = int(n_shrimp * base_rate)
        # Add stochastic component
        n_dead += rng.poisson(max(0, n_shrimp * base_rate * 0.1))
        return min(n_dead, n_shrimp)

    elif disease_type == "wssv":
        # 20% per day base, accelerating
        accel = min(3.0, 1.0 + 0.3 * days_since_infection)
        base_rate = min(0.95, 0.20 * accel)
        n_dead = int(n_shrimp * base_rate)
        n_dead += rng.poisson(max(0, n_shrimp * base_rate * 0.05))
        return min(n_dead, n_shrimp)

    return 0


def compute_stress_index(
    do_mg_L: float,
    tan_mg_L: float,
    ph: float,
    salinity_ppt: float,
    alkalinity_mg_L: float,
) -> float:
    """Composite environmental stress index (0–1) for L. vannamei.

    Higher = more stressed = reduced growth.
    """
    stress = 0.0

    # DO stress (most critical)
    if do_mg_L < 3.0:
        stress += 0.5
    elif do_mg_L < 4.0:
        stress += 0.3
    elif do_mg_L < 5.0:
        stress += 0.1

    # TAN stress (shrimp very sensitive)
    if tan_mg_L > 0.5:
        stress += 0.4
    elif tan_mg_L > 0.2:
        stress += 0.2
    elif tan_mg_L > 0.1:
        stress += 0.1

    # pH stress
    if ph < 7.0 or ph > 9.0:
        stress += 0.3
    elif ph < 7.5 or ph > 8.8:
        stress += 0.1

    # Salinity stress (optimal 10–25 ppt)
    if salinity_ppt < 5.0 or salinity_ppt > 35.0:
        stress += 0.3
    elif salinity_ppt < 10.0 or salinity_ppt > 28.0:
        stress += 0.1

    # Alkalinity stress (critical for molting)
    if alkalinity_mg_L < 50.0:
        stress += 0.2
    elif alkalinity_mg_L < 100.0:
        stress += 0.1

    return min(1.0, stress)
