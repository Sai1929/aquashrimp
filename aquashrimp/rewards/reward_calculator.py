"""4-component reward calculator for all AquaShrimp tasks.

Components:
  R_total = w_growth × R_growth + w_water × R_water + w_econ × R_econ + w_biosecurity × R_biosecurity

Each component in range [−1.0, +1.0]. R_total also in [−1.0, +1.0].

Weights per task:
  Task 1 (NurseryPond):       growth=0.35, water=0.35, econ=0.20, bio=0.10
  Task 2 (SemiIntensive):     growth=0.30, water=0.25, econ=0.30, bio=0.15
  Task 3 (CommercialGrowOut): growth=0.25, water=0.20, econ=0.35, bio=0.20
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from aquashrimp.models.observations import RewardBreakdown

# Task weights
TASK_WEIGHTS = {
    1: {"growth": 0.35, "water": 0.35, "economic": 0.20, "biosecurity": 0.10},
    2: {"growth": 0.30, "water": 0.25, "economic": 0.30, "biosecurity": 0.15},
    3: {"growth": 0.25, "water": 0.20, "economic": 0.35, "biosecurity": 0.20},
}


# ── Component: Growth ─────────────────────────────────────────────────────────

def growth_reward(
    W_actual: float,
    W_expected: float,
    mortality_rate_today: float,
) -> float:
    """Growth component reward.

    R_growth = tanh(3 × deviation) − 2 × mortality_rate_today
    Clipped to [−1, +1].
    """
    if W_expected <= 0:
        deviation = 0.0
    else:
        deviation = (W_actual - W_expected) / max(W_expected, 0.01)

    r = math.tanh(3.0 * deviation) - 2.0 * mortality_rate_today
    return max(-1.0, min(1.0, r))


def expected_weight(
    W_start: float,
    day: int,
    temp_mean: float = 27.0,
) -> float:
    """Benchmark expected weight at ideal conditions on given day."""
    from aquashrimp.simulation.shrimp_growth import DWG_MAX_G_PER_DAY, f_temp
    return W_start + DWG_MAX_G_PER_DAY * f_temp(temp_mean) * day


# ── Component: Water Quality ──────────────────────────────────────────────────

def do_score(do_mg_L: float) -> float:
    """DO quality score."""
    if do_mg_L >= 5.0:
        return 1.0
    elif do_mg_L >= 3.0:
        return (do_mg_L - 3.0) / 2.0  # linear 3–5
    else:
        return -1.0


def tan_score(tan_mg_L: float) -> float:
    """TAN quality score. Shrimp more sensitive than fish (<0.1 mg/L critical)."""
    if tan_mg_L <= 0.05:
        return 1.0
    elif tan_mg_L <= 0.5:
        return 1.0 - (tan_mg_L - 0.05) / 0.45 * 2.0  # linear decline to −1.0 at 0.5
    else:
        return -1.0


def ph_score(ph: float) -> float:
    """pH quality score. Optimal 7.8–8.3."""
    if 7.8 <= ph <= 8.3:
        return 1.0
    elif 7.5 <= ph < 7.8:
        return (ph - 7.5) / 0.3
    elif 8.3 < ph <= 8.8:
        return 1.0 - (ph - 8.3) / 0.5
    elif 7.0 <= ph < 7.5:
        return -0.5 * (7.5 - ph) / 0.5
    else:
        return -1.0


def alkalinity_score(alkalinity_mg_L: float) -> float:
    """Alkalinity quality score. Critical for molting (>100 mg/L CaCO3)."""
    if alkalinity_mg_L >= 100.0:
        return 1.0
    elif alkalinity_mg_L >= 50.0:
        return (alkalinity_mg_L - 50.0) / 50.0
    else:
        return -0.5


def water_quality_reward(
    do_mg_L: float,
    tan_mg_L: float,
    ph: float,
    alkalinity_mg_L: float,
) -> float:
    """Composite water quality reward.

    Weights: DO=50%, TAN=25%, pH=15%, Alkalinity=10%.
    """
    r = (
        0.50 * do_score(do_mg_L)
        + 0.25 * tan_score(tan_mg_L)
        + 0.15 * ph_score(ph)
        + 0.10 * alkalinity_score(alkalinity_mg_L)
    )
    return max(-1.0, min(1.0, r))


# ── Component: Economic ───────────────────────────────────────────────────────

def economic_reward(
    revenue_accrual_usd: float,
    daily_cost_usd: float,
    fcr_actual: float,
    fcr_target: float = 1.4,
) -> float:
    """Economic component reward.

    R_econ = tanh(3 × profit_margin) − FCR_overage_penalty
    """
    margin_denominator = max(revenue_accrual_usd, 0.01)
    margin = (revenue_accrual_usd - daily_cost_usd) / margin_denominator

    r = math.tanh(3.0 * margin)

    # FCR overage penalty
    if fcr_actual > 0 and fcr_target > 0:
        fcr_penalty = max(0.0, (fcr_actual / fcr_target) - 1.0) * 0.3
        r -= fcr_penalty

    return max(-1.0, min(1.0, r))


# ── Component: Biosecurity ────────────────────────────────────────────────────

def biosecurity_reward(
    disease_containment_score: float,
    antibiotic_used_episode: bool,
    wssv_spread_ponds: int,
    overdue_report_days: int,
) -> float:
    """Biosecurity and compliance reward component.

    R_biosecurity = disease_containment_score
                  − antibiotic_penalty (−0.3 if used)
                  − WSSV spread penalty (−0.4 per pond)
                  − 0.2 × overdue_report_days
    """
    r = disease_containment_score

    if antibiotic_used_episode:
        r -= 0.3

    r -= 0.4 * wssv_spread_ponds
    r -= 0.2 * overdue_report_days

    return max(-1.0, min(1.0, r))


# ── Total Reward ──────────────────────────────────────────────────────────────

def compute_total_reward(
    task_id: int,
    r_growth: float,
    r_water: float,
    r_econ: float,
    r_biosecurity: float,
) -> tuple[float, RewardBreakdown]:
    """Compute weighted total reward and breakdown.

    Args:
        task_id: 1, 2, or 3
        r_growth: Growth component (−1 to +1)
        r_water: Water quality component (−1 to +1)
        r_econ: Economic component (−1 to +1)
        r_biosecurity: Biosecurity component (−1 to +1)

    Returns:
        (total_reward, RewardBreakdown)
    """
    weights = TASK_WEIGHTS[task_id]
    total = (
        weights["growth"] * r_growth
        + weights["water"] * r_water
        + weights["economic"] * r_econ
        + weights["biosecurity"] * r_biosecurity
    )
    # Clamp to [-1, +1] (theoretically always in range, but floating point safety)
    total = max(-1.0, min(1.0, total))

    breakdown = RewardBreakdown(
        total=total,
        growth=r_growth,
        water_quality=r_water,
        economic=r_econ,
        biosecurity=r_biosecurity,
    )
    return total, breakdown


def compute_grade(cumulative_reward: float, steps: int) -> float:
    """Normalize cumulative episode reward to agent grade [0.0, 1.0].

    Maps mean per-step reward linearly: -1.0→0.0, 0.0→0.5, +1.0→1.0.
    Returns 0.5 (neutral) before any steps are taken.
    """
    if steps <= 0:
        return 0.5
    mean_reward = cumulative_reward / steps
    return max(0.0, min(1.0, (mean_reward + 1.0) / 2.0))


def compute_nursery_pond_reward(
    W_actual: float,
    W_start: float,
    day: int,
    temp: float,
    mortality_rate_today: float,
    do_mg_L: float,
    tan_mg_L: float,
    ph: float,
    alkalinity_mg_L: float,
    revenue_accrual: float,
    daily_cost: float,
    fcr_actual: float,
    disease_containment_score: float = 1.0,
    antibiotic_used: bool = False,
) -> tuple[float, RewardBreakdown]:
    """Convenience function for Task 1 reward."""
    r_growth = growth_reward(W_actual, expected_weight(W_start, day, temp), mortality_rate_today)
    r_water = water_quality_reward(do_mg_L, tan_mg_L, ph, alkalinity_mg_L)
    r_econ = economic_reward(revenue_accrual, daily_cost, fcr_actual)
    r_bio = biosecurity_reward(disease_containment_score, antibiotic_used, 0, 0)
    return compute_total_reward(1, r_growth, r_water, r_econ, r_bio)
