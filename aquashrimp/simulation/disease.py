"""Disease models for shrimp aquaculture.

Vibrio bacterial infection (Task 2): treatable, but antibiotics risk export compliance.
WSSV (White Spot Syndrome Virus) (Task 3): no treatment — containment only.

The WSSV 3-day detection window is the most consequential mechanic:
- Day 1–2: tray consumption drops (ONLY early warning)
- Day 3+: visible mortality, too late for full recovery
"""
from __future__ import annotations
import numpy as np
from aquashrimp.models.enums import DiseaseType


# ── Vibrio constants ──────────────────────────────────────────────────────────
VIBRIO_TRIGGER_WINDOW = (20, 40)   # stochastic trigger day range (Task 2)
VIBRIO_GROWTH_RATE = 0.15          # severity growth per day without treatment
VIBRIO_PROBIOTIC_EFFICACY = 0.5    # treatment efficacy
VIBRIO_ANTIBIOTIC_EFFICACY = 0.9   # higher efficacy, but export risk
VIBRIO_PROBIOTIC_ONSET_DAYS = 3    # days before probiotic takes effect
VIBRIO_MORTALITY_COEFF = 0.008     # base λ per shrimp × severity²

# ── WSSV constants ────────────────────────────────────────────────────────────
WSSV_TRIGGER_DAY_MIN = 45          # cannot trigger before this day (Task 3)
WSSV_SPREAD_PROB_WATER = 0.20      # P(spread per day) if water exchanged from infected pond
WSSV_SPREAD_PROB_NO_WATER = 0.02   # P(spread per day) via aerosol/birds
WSSV_EXTERNAL_PROB = 0.05          # P(external contamination per day) with neighbor outbreak
WSSV_EXTERNAL_BIOSEC = 0.01        # P(external) with biosecurity measures active
WSSV_EMERGENCY_HARVEST_CUTOFF = 3  # days after detection; after this, >50% losses
WSSV_EMERGENCY_REVENUE_PENALTY = 0.15  # 15% revenue penalty for emergency harvest


def vibrio_trigger_day(rng: np.random.Generator) -> int:
    """Draw stochastic Vibrio infection trigger day."""
    return int(rng.integers(VIBRIO_TRIGGER_WINDOW[0], VIBRIO_TRIGGER_WINDOW[1] + 1))


def step_vibrio_severity(
    severity: float,
    probiotic_active: bool,
    antibiotic_active: bool,
    probiotic_days_elapsed: int,
) -> float:
    """Update Vibrio infection severity.

    Severity grows each day unless treated.
    Probiotic has 3-day onset; antibiotic is immediate.
    """
    if antibiotic_active:
        efficacy = VIBRIO_ANTIBIOTIC_EFFICACY
    elif probiotic_active and probiotic_days_elapsed >= VIBRIO_PROBIOTIC_ONSET_DAYS:
        efficacy = VIBRIO_PROBIOTIC_EFFICACY
    elif probiotic_active:
        # Probiotic still in onset period — partial efficacy
        efficacy = VIBRIO_PROBIOTIC_EFFICACY * (probiotic_days_elapsed / VIBRIO_PROBIOTIC_ONSET_DAYS)
    else:
        efficacy = 0.0

    growth = VIBRIO_GROWTH_RATE * (1.0 - efficacy)
    new_severity = severity * (1.0 + growth)
    return min(1.0, new_severity)


def vibrio_mortality_count(n_shrimp: int, severity: float, rng: np.random.Generator) -> int:
    """Daily mortality from Vibrio infection."""
    if severity <= 0:
        return 0
    base_rate = VIBRIO_MORTALITY_COEFF * severity ** 2
    n_dead = int(n_shrimp * base_rate)
    n_dead += rng.poisson(max(0, n_shrimp * base_rate * 0.1))
    return min(n_dead, n_shrimp)


def vibrio_observable_signals(severity: float, days_since_infection: int) -> dict:
    """Observable signals from Vibrio infection.

    Returns:
        {redness_score: 0–5, tray_drop_factor: 0–1}
    """
    # Redness appears after day 2 of infection
    if days_since_infection < 2:
        redness = 0.0
    else:
        redness = min(5.0, severity * 5.0 * (days_since_infection / 5.0))

    # Tray consumption drops to 20–40% of normal during Vibrio
    tray_drop = max(0.2, 1.0 - 0.8 * severity)

    return {"redness_score": round(redness, 1), "tray_drop_factor": round(tray_drop, 2)}


def wssv_trigger_day(rng: np.random.Generator, max_day: int = 90) -> int:
    """Draw stochastic WSSV trigger day (cannot trigger before day 45)."""
    return int(rng.integers(WSSV_TRIGGER_DAY_MIN, max_day))


def wssv_consumption_signal(days_since_infection: int, base_fraction: float = 0.9) -> float:
    """Tray consumption fraction under WSSV infection.

    This is the primary EARLY WARNING signal — drops before mortality appears.
    Days 1–2: consumption falls sharply → agent can detect before clinical signs
    Day 3+: near-zero consumption
    """
    from aquashrimp.simulation.feeding_trays import wssv_early_warning_fraction
    return wssv_early_warning_fraction(days_since_infection, base_fraction)


def wssv_mortality_count(
    n_shrimp: int,
    days_since_infection: int,
    rng: np.random.Generator,
) -> int:
    """Daily WSSV mortality — accelerating progression.

    Day 1–2: no mortality (feeding tray drop is the only signal)
    Day 3–4: 20% per day
    Day 5–7: 50–95% per day
    """
    if days_since_infection <= 2:
        return 0  # Pre-clinical: CRITICAL early detection window
    elif days_since_infection <= 4:
        accel = 1.0 + 0.3 * (days_since_infection - 2)
        base_rate = min(0.25, 0.20 * accel)
    else:
        accel = min(4.0, 1.0 + 0.5 * (days_since_infection - 2))
        base_rate = min(0.95, 0.20 * accel)

    n_dead = int(n_shrimp * base_rate)
    n_dead += rng.poisson(max(0, n_shrimp * base_rate * 0.05))
    return min(n_dead, n_shrimp)


def wssv_spread_check(
    rng: np.random.Generator,
    water_exchange_from_infected: bool,
    biosecurity_active: bool,
    neighbor_outbreak: bool,
) -> bool:
    """Check if WSSV spreads to this pond today.

    Args:
        water_exchange_from_infected: True if pond received water from infected pond
        biosecurity_active: True if water inlet disinfection active
        neighbor_outbreak: External pressure from neighboring farm

    Returns:
        True if spread occurs
    """
    spread_prob = 0.0

    if water_exchange_from_infected:
        spread_prob = max(spread_prob, WSSV_SPREAD_PROB_WATER)

    # Background spread via aerosol, birds, shared equipment
    spread_prob = max(spread_prob, WSSV_SPREAD_PROB_NO_WATER)

    if neighbor_outbreak:
        external_prob = WSSV_EXTERNAL_BIOSEC if biosecurity_active else WSSV_EXTERNAL_PROB
        spread_prob = max(spread_prob, external_prob)

    if biosecurity_active:
        spread_prob *= 0.2  # 80% reduction from biosecurity

    return bool(rng.random() < spread_prob)


def emergency_harvest_recovery(days_since_detection: int) -> tuple[float, float]:
    """Biomass recovery fraction and revenue penalty for emergency harvest.

    Returns:
        (biomass_fraction_saved, revenue_penalty)
    """
    if days_since_detection <= WSSV_EMERGENCY_HARVEST_CUTOFF:
        # Days 1–3: save 60–80% of biomass
        save_frac = 0.80 - 0.07 * (days_since_detection - 1)
        return max(0.60, save_frac), WSSV_EMERGENCY_REVENUE_PENALTY
    else:
        # Day 4+: too late, <50% recovery
        save_frac = max(0.10, 0.50 - 0.15 * (days_since_detection - 3))
        return save_frac, WSSV_EMERGENCY_REVENUE_PENALTY * 1.5


def neighbor_outbreak_trigger(day: int, rng: np.random.Generator) -> bool:
    """Stochastic trigger for neighboring farm WSSV outbreak (from day 40)."""
    if day < 40:
        return False
    return bool(rng.random() < 0.02)  # 2% chance per day after day 40
