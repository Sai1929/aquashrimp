"""Market simulation: shrimp price random walk and cost tables.

Shrimp prices follow seasonal patterns with stochastic random walk.
Harvest timing (20g vs 25g) can significantly affect revenue.
"""
from __future__ import annotations
import numpy as np


# ── Price parameters ──────────────────────────────────────────────────────────
VANNAMEI_PRICE_MEAN = 5.0        # USD/kg baseline
VANNAMEI_PRICE_MIN = 2.5         # USD/kg floor
VANNAMEI_PRICE_MAX = 9.0         # USD/kg ceiling
PRICE_SEASONAL_AMPLITUDE = 0.8   # USD/kg seasonal variation
PRICE_RANDOM_WALK_STD = 0.15     # USD/kg daily random walk step

# ── Feed cost table (by quality/type) ────────────────────────────────────────
FEED_PRICE_USD_PER_KG = 1.20     # Standard commercial pellet

# ── Operating costs ───────────────────────────────────────────────────────────
ENERGY_COST_PER_AERATOR_HOUR = 0.05   # USD per aerator-hour
PROBIOTIC_COST_PER_POND_DAY = 2.0    # USD per pond per day
ANTIBIOTIC_COST_PER_POND_DAY = 8.0   # USD per pond per treatment
LIME_COST_PER_KG = 0.15              # USD per kg Ca(OH)2
INSPECTION_COST_PER_POND = 5.0       # USD per pond inspection (WSSV check)
BIOSECURITY_DAILY_COST = 20.0        # USD per day for water inlet disinfection
DISINFECTION_COST_PER_POND = 150.0   # USD for chlorine disinfection (post-WSSV)
LABOR_COST_PER_DAY = 15.0            # USD base labor per day (per farm, not per pond)
TRAY_CHECK_LABOR_COST = 0.5          # USD per tray check (30-min labor)

# ── Size premium table (price premium by market size) ────────────────────────
SIZE_PREMIUM = {
    # (min_g, max_g): price_multiplier
    (0, 5): 0.5,
    (5, 10): 0.7,
    (10, 15): 0.9,
    (15, 20): 1.0,       # Standard market size
    (20, 25): 1.15,      # Premium size
    (25, 999): 1.25,     # Super premium (30/40 count)
}


def size_price_multiplier(mean_weight_g: float) -> float:
    """Price multiplier based on shrimp size at harvest."""
    for (min_g, max_g), mult in SIZE_PREMIUM.items():
        if min_g <= mean_weight_g < max_g:
            return mult
    return 1.25  # very large shrimp = premium


def update_price(
    current_price: float,
    day: int,
    rng: np.random.Generator,
    total_days: int = 90,
) -> float:
    """Update shrimp price with seasonal trend + random walk.

    Price tends higher at end of season (lower supply from other farms).
    """
    # Seasonal component: prices slightly higher mid-season
    seasonal = PRICE_SEASONAL_AMPLITUDE * np.sin(np.pi * day / total_days)

    # Mean reversion + random walk
    mean_reversion = 0.05 * (VANNAMEI_PRICE_MEAN - current_price)
    random_step = rng.normal(0.0, PRICE_RANDOM_WALK_STD)

    new_price = current_price + mean_reversion + random_step + seasonal * 0.02
    return float(np.clip(new_price, VANNAMEI_PRICE_MEAN - 2.0, VANNAMEI_PRICE_MEAN + 2.0))


def compute_daily_costs(
    feed_kg: float,
    aeration_hours: float,
    lime_kg: float = 0.0,
    probiotic_ponds: int = 0,
    antibiotic_ponds: int = 0,
    inspection_ponds: int = 0,
    biosecurity_active: bool = False,
    tray_checks: int = 0,
    disinfection_ponds: int = 0,
) -> dict[str, float]:
    """Compute all daily operating costs.

    Returns itemized cost dictionary.
    """
    costs = {
        "feed": feed_kg * FEED_PRICE_USD_PER_KG,
        "energy": aeration_hours * ENERGY_COST_PER_AERATOR_HOUR,
        "lime": lime_kg * LIME_COST_PER_KG,
        "probiotic": probiotic_ponds * PROBIOTIC_COST_PER_POND_DAY,
        "antibiotic": antibiotic_ponds * ANTIBIOTIC_COST_PER_POND_DAY,
        "inspection": inspection_ponds * INSPECTION_COST_PER_POND,
        "biosecurity": BIOSECURITY_DAILY_COST if biosecurity_active else 0.0,
        "tray_labor": tray_checks * TRAY_CHECK_LABOR_COST,
        "disinfection": disinfection_ponds * DISINFECTION_COST_PER_POND,
        "labor": LABOR_COST_PER_DAY,
    }
    costs["total"] = sum(v for k, v in costs.items() if k != "total")
    return costs


def compute_harvest_revenue(
    biomass_kg: float,
    mean_weight_g: float,
    price_usd_per_kg: float,
    harvest_type: str = "full",
    emergency: bool = False,
) -> float:
    """Compute revenue from a harvest.

    Emergency harvest applies 15% penalty.
    """
    size_mult = size_price_multiplier(mean_weight_g)
    effective_price = price_usd_per_kg * size_mult

    if emergency:
        effective_price *= (1.0 - 0.15)  # 15% revenue penalty

    return biomass_kg * effective_price


def daily_revenue_accrual(
    w_new: float,
    w_old: float,
    n_shrimp: int,
    price_usd_per_kg: float,
) -> float:
    """Daily revenue accrual from biomass gain (not actual harvest).

    Used for dense reward signal calculation.
    """
    biomass_gain_kg = (w_new - w_old) * n_shrimp / 1000.0
    return max(0.0, biomass_gain_kg * price_usd_per_kg)
