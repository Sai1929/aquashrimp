"""Feeding tray monitoring system — partial observability mechanic.

Shrimp are bottom feeders; feed trays reveal actual consumption rate.
This is the primary observability tool unique to shrimp aquaculture.

If the agent checks trays: sees noisy consumption fraction.
If not: only sees lagging mortality + FCR proxy.
"""
from __future__ import annotations
import numpy as np


TRAY_NOISE_STD = 0.05  # 5% Gaussian noise on observed consumption


def true_consumption_fraction(
    biomass_kg: float,
    feed_fed_kg: float,
    disease_severity: float,
    stress_index: float,
    appetite_factor: float = 1.0,
) -> float:
    """Compute true tray consumption fraction (not visible to agent unless checked).

    Args:
        biomass_kg: Current shrimp biomass
        feed_fed_kg: Feed placed in pond
        disease_severity: 0–1 (disease reduces appetite)
        stress_index: 0–1 (stress reduces appetite)
        appetite_factor: Baseline appetite multiplier (affected by weather etc.)

    Returns:
        true_fraction: 0.0–1.0 fraction of feed consumed
    """
    if feed_fed_kg <= 0 or biomass_kg <= 0:
        return 0.0

    # Maximum consumption based on biomass (shrimp eat ~4–6% body weight/day)
    max_consume_kg = biomass_kg * 0.05 * appetite_factor

    # Disease and stress reduce appetite
    appetite_reduction = 1.0 - 0.8 * disease_severity - 0.4 * stress_index
    actual_consume_kg = max_consume_kg * max(0.0, appetite_reduction)

    fraction = min(1.0, actual_consume_kg / feed_fed_kg)
    return max(0.0, fraction)


def observe_tray(
    true_fraction: float,
    rng: np.random.Generator,
) -> float:
    """Observed tray consumption with 5% Gaussian noise.

    Returns observed fraction (clipped to [0, 1]).
    """
    noise = rng.normal(0.0, TRAY_NOISE_STD)
    observed = true_fraction + noise
    return max(0.0, min(1.0, observed))


def wssv_early_warning_fraction(
    days_since_infection: int,
    base_fraction: float = 0.9,
) -> float:
    """Simulate WSSV-driven feed consumption drop.

    WSSV causes feeding to drop before visible mortality:
    - Days 1–2 after infection: fraction drops to 0.1–0.2 (early warning signal)
    - Day 3+: essentially zero appetite

    This creates the critical early detection window for the agent.
    """
    if days_since_infection <= 0:
        return base_fraction
    elif days_since_infection == 1:
        return base_fraction * 0.25  # ~75% drop on day 1
    elif days_since_infection == 2:
        return base_fraction * 0.10  # ~90% drop on day 2
    else:
        return base_fraction * 0.02  # minimal feeding after day 3


def vibrio_appetite_factor(severity: float) -> float:
    """Vibrio infection reduces appetite proportional to severity."""
    return max(0.1, 1.0 - 0.9 * severity)


def interpret_tray_signal(fraction: float) -> str:
    """Human-readable interpretation of tray consumption fraction."""
    if fraction > 0.85:
        return "HUNGRY — increase feed"
    elif fraction > 0.65:
        return "GOOD — maintain feed"
    elif fraction > 0.45:
        return "REDUCED — monitor closely"
    elif fraction > 0.20:
        return "LOW — possible disease or overfeeding"
    else:
        return "CRITICAL — disease or severe stress suspected"


def uneaten_feed(feed_fed_kg: float, consumption_fraction: float) -> float:
    """Compute uneaten feed (drives TAN/ammonia accumulation)."""
    return feed_fed_kg * (1.0 - consumption_fraction)
