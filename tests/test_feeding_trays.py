"""Tests for feeding tray partial observability system."""
import pytest
import numpy as np
from aquashrimp.simulation.feeding_trays import (
    true_consumption_fraction,
    observe_tray,
    uneaten_feed,
    wssv_early_warning_fraction,
    vibrio_appetite_factor,
    interpret_tray_signal,
)


class TestFeedingTrays:
    def test_healthy_shrimp_high_consumption(self):
        """Healthy, well-fed shrimp should consume most of their feed."""
        frac = true_consumption_fraction(
            biomass_kg=100.0, feed_fed_kg=4.0,
            disease_severity=0.0, stress_index=0.0
        )
        assert frac > 0.7, f"Healthy shrimp should eat >70% of appropriate feed, got {frac:.2f}"

    def test_disease_reduces_consumption(self):
        """Disease should reduce feed consumption."""
        healthy = true_consumption_fraction(100.0, 4.0, 0.0, 0.0)
        sick = true_consumption_fraction(100.0, 4.0, 0.8, 0.0)
        assert sick < healthy, "Disease should reduce consumption"
        assert sick < 0.5, "Severe disease should drop consumption below 50%"

    def test_overfeeding_reduces_fraction(self):
        """Overfeeding (relative to appetite) should reduce consumption fraction."""
        appropriate = true_consumption_fraction(100.0, 5.0, 0.0, 0.0)
        overfed = true_consumption_fraction(100.0, 20.0, 0.0, 0.0)
        assert overfed < appropriate, "Overfeeding should reduce consumption fraction"

    def test_observe_adds_noise(self):
        """Observation should add 5% Gaussian noise."""
        rng = np.random.default_rng(42)
        observations = [observe_tray(0.8, rng) for _ in range(1000)]
        mean_obs = np.mean(observations)
        std_obs = np.std(observations)
        assert abs(mean_obs - 0.8) < 0.05, f"Mean observation should be ≈0.8, got {mean_obs:.3f}"
        assert 0.02 < std_obs < 0.12, f"Std should be ≈0.05, got {std_obs:.3f}"

    def test_observe_clipped_to_valid_range(self):
        """Observed fraction should always be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            obs = observe_tray(0.0, rng)
            assert 0.0 <= obs <= 1.0
            obs = observe_tray(1.0, rng)
            assert 0.0 <= obs <= 1.0

    def test_uneaten_feed_calculation(self):
        """Uneaten feed should equal fed × (1 - fraction consumed)."""
        uneaten = uneaten_feed(10.0, 0.8)
        assert abs(uneaten - 2.0) < 0.01, f"Uneaten should be 2.0 kg, got {uneaten}"

        uneaten_zero = uneaten_feed(10.0, 1.0)
        assert abs(uneaten_zero) < 0.01, "Perfect consumption = 0 uneaten"

    def test_wssv_early_warning_day1(self):
        """WSSV day 1: significant consumption drop (the early warning signal)."""
        base = 0.9
        frac_day0 = wssv_early_warning_fraction(0, base)
        frac_day1 = wssv_early_warning_fraction(1, base)
        frac_day2 = wssv_early_warning_fraction(2, base)
        frac_day3 = wssv_early_warning_fraction(3, base)

        assert frac_day0 == base, "Day 0: no disease effect"
        assert frac_day1 < 0.3, f"Day 1: should drop below 0.3, got {frac_day1:.2f}"
        assert frac_day2 < 0.15, f"Day 2: should drop below 0.15, got {frac_day2:.2f}"
        assert frac_day3 < 0.05, f"Day 3: near zero consumption"
        assert frac_day1 > frac_day2 > frac_day3, "Consumption should decrease each day"

    def test_vibrio_appetite_factor(self):
        """Vibrio severity should reduce appetite factor."""
        full = vibrio_appetite_factor(0.0)
        partial = vibrio_appetite_factor(0.5)
        severe = vibrio_appetite_factor(1.0)
        assert full == 1.0
        assert partial < full
        assert severe == 0.1  # floor

    def test_interpret_tray_signal(self):
        """Tray signal interpretation should match documented thresholds."""
        assert "HUNGRY" in interpret_tray_signal(0.95)
        assert "GOOD" in interpret_tray_signal(0.75)
        assert "REDUCED" in interpret_tray_signal(0.55)
        assert "LOW" in interpret_tray_signal(0.30)
        assert "CRITICAL" in interpret_tray_signal(0.10)
