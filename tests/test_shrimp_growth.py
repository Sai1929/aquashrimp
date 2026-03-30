"""Tests for shrimp growth model."""
import pytest
import numpy as np
from aquashrimp.simulation.shrimp_growth import (
    daily_weight_gain,
    daily_weight_gain_full,
    estimate_feed_demand,
    f_temp,
    f_feed,
    f_density,
    f_health,
    inter_molt_period,
    check_molt_event,
    cannibalism_mortality,
    compute_stress_index,
    DWG_MAX_G_PER_DAY,
    T_OPT,
)


class TestDWGModel:
    def test_optimal_conditions_growth(self):
        """Shrimp should grow ~0.2g/day under ideal conditions."""
        w_new = daily_weight_gain(
            W_old=5.0, temp=27.0, feed_ratio=1.0,
            density_kg_m2=1.5, disease_severity=0.0
        )
        assert 5.0 < w_new < 5.4, f"Expected 5.1–5.3g at optimal conditions, got {w_new:.3f}g"

    def test_plan_verification_sanity(self):
        """Verify the exact assertion from the plan verification script."""
        w_new = daily_weight_gain(
            W_old=5.0, temp=27.0, feed_ratio=1.0,
            density_kg_m2=1.5, disease_severity=0.0
        )
        assert 5.0 < w_new < 5.4, f"Expected ~5.2g, got {w_new:.2f}g"

    def test_growth_monotone_with_feed(self):
        """More feed → more growth (up to satiation)."""
        w_low = daily_weight_gain(5.0, 27.0, 0.3, 1.5, 0.0)
        w_mid = daily_weight_gain(5.0, 27.0, 0.8, 1.5, 0.0)
        w_high = daily_weight_gain(5.0, 27.0, 1.5, 1.5, 0.0)
        assert w_low < w_mid <= w_high, "Growth should increase with feed ratio up to satiation"

    def test_disease_reduces_growth(self):
        """Disease should reduce growth."""
        w_healthy = daily_weight_gain(5.0, 27.0, 1.0, 1.5, 0.0)
        w_sick = daily_weight_gain(5.0, 27.0, 1.0, 1.5, 0.8)
        assert w_sick < w_healthy, "Disease should reduce growth"
        assert w_sick > 5.0, "Growth should still be positive under mild disease"

    def test_temperature_effect(self):
        """Growth peaks at 27°C and drops at extremes."""
        w_cold = daily_weight_gain(5.0, 18.0, 1.0, 1.5, 0.0)
        w_opt = daily_weight_gain(5.0, 27.0, 1.0, 1.5, 0.0)
        w_hot = daily_weight_gain(5.0, 35.0, 1.0, 1.5, 0.0)
        assert w_opt > w_cold, "Optimal temperature should give more growth than cold"
        assert w_opt > w_hot, "Optimal temperature should give more growth than hot"

    def test_density_reduces_growth(self):
        """High density should reduce growth."""
        w_low = daily_weight_gain(5.0, 27.0, 1.0, 1.0, 0.0)
        w_high = daily_weight_gain(5.0, 27.0, 1.0, 5.0, 0.0)
        assert w_high <= w_low, "High density should reduce growth"

    def test_no_negative_growth(self):
        """Weight should never decrease."""
        w_new = daily_weight_gain(1.0, 15.0, 0.0, 10.0, 1.0)
        assert w_new >= 1.0, "Weight should never decrease"

    def test_f_temp_peak(self):
        """f_temp should peak near optimal temperature."""
        f_opt = f_temp(T_OPT)
        f_cold = f_temp(20.0)
        assert abs(f_opt - 1.0) < 0.01, f"f_temp at optimum should ≈ 1.0, got {f_opt:.4f}"
        assert f_cold < f_opt

    def test_f_feed_capped_at_one(self):
        """f_feed should not exceed 1.0."""
        assert f_feed(10.0) <= 1.0, "f_feed should cap at 1.0"
        assert f_feed(0.79) < 1.0  # just below satiation threshold
        assert f_feed(0.0) == 0.0

    def test_f_density_floor(self):
        """f_density should have a minimum floor of 0.4."""
        very_dense = f_density(1000.0, 100.0)  # 10 kg/m²
        assert very_dense >= 0.4, f"f_density floor is 0.4, got {very_dense}"


class TestMolting:
    def test_molt_period_temperature_dependent(self):
        """Warmer temperature = shorter molt period."""
        p_cold = inter_molt_period(22.0)
        p_warm = inter_molt_period(28.0)
        assert p_warm < p_cold, "Warmer water → shorter molt period"
        assert p_warm >= 3.0, "Molt period should not drop below 3 days"

    def test_molt_occurs_after_period(self):
        """Molting should occur when days_since_molt >= inter-molt period."""
        rng = np.random.default_rng(42)
        # At 27°C, molt period ≈ 3.45 days
        molt, new_days = check_molt_event(10.0, 27.0, rng)
        assert molt, "Should molt after exceeding inter-molt period"
        assert new_days == 0.0, "Days since molt resets to 0"

    def test_no_molt_before_period(self):
        """Should not molt before inter-molt period."""
        rng = np.random.default_rng(42)
        molt, _ = check_molt_event(0.5, 27.0, rng)
        assert not molt, "Should not molt on day 0.5"

    def test_cannibalism_scales_with_density(self):
        """Cannibalism mortality increases with density."""
        low_density_dead = cannibalism_mortality(10000, 0.5)
        high_density_dead = cannibalism_mortality(10000, 5.0)
        assert high_density_dead >= low_density_dead


class TestFeedDemand:
    def test_feed_demand_positive(self):
        """Feed demand should always be positive for live shrimp."""
        demand = estimate_feed_demand(100.0, 10.0, 27.0)
        assert demand > 0

    def test_feed_demand_scales_with_biomass(self):
        """More biomass = more feed needed."""
        d_small = estimate_feed_demand(50.0, 10.0, 27.0)
        d_large = estimate_feed_demand(200.0, 10.0, 27.0)
        assert d_large > d_small


class TestStressIndex:
    def test_optimal_conditions_low_stress(self):
        """Optimal water quality should produce near-zero stress."""
        stress = compute_stress_index(
            do_mg_L=7.0, tan_mg_L=0.02, ph=8.1,
            salinity_ppt=15.0, alkalinity_mg_L=120.0
        )
        assert stress < 0.2, f"Optimal conditions should have low stress, got {stress:.3f}"

    def test_low_do_high_stress(self):
        """DO crash should cause high stress."""
        stress = compute_stress_index(
            do_mg_L=2.0, tan_mg_L=0.05, ph=8.0,
            salinity_ppt=15.0, alkalinity_mg_L=120.0
        )
        assert stress > 0.4, f"DO crash should cause high stress, got {stress:.3f}"

    def test_high_tan_stress(self):
        """High ammonia should cause stress."""
        stress = compute_stress_index(
            do_mg_L=7.0, tan_mg_L=0.8, ph=8.0,
            salinity_ppt=15.0, alkalinity_mg_L=120.0
        )
        assert stress > 0.3
