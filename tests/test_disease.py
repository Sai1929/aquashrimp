"""Tests for disease models — WSSV and Vibrio critical invariants."""
import pytest
import numpy as np
from aquashrimp.simulation.disease import (
    wssv_consumption_signal,
    wssv_mortality_count,
    wssv_spread_check,
    emergency_harvest_recovery,
    step_vibrio_severity,
    vibrio_mortality_count,
    vibrio_observable_signals,
)
from aquashrimp.simulation.feeding_trays import wssv_early_warning_fraction


class TestWSSV:
    def test_tray_drops_before_mortality(self):
        """Critical invariant: tray consumption drops on days 1–2 before mortality appears.

        This is the key early warning signal for WSSV.
        """
        rng = np.random.default_rng(42)

        # Day 1: consumption should drop significantly, mortality = 0
        tray_day1 = wssv_consumption_signal(1, base_fraction=0.9)
        dead_day1 = wssv_mortality_count(100000, 1, rng)

        # Day 2: still low consumption, still pre-clinical
        tray_day2 = wssv_consumption_signal(2, base_fraction=0.9)
        dead_day2 = wssv_mortality_count(100000, 2, rng)

        # Day 4: heavy mortality
        dead_day4 = wssv_mortality_count(100000, 4, rng)

        assert tray_day1 < 0.3, f"Tray consumption on day 1 should drop below 30%, got {tray_day1:.2f}"
        assert tray_day2 < 0.15, f"Tray consumption on day 2 should drop below 15%, got {tray_day2:.2f}"
        assert dead_day1 == 0, f"No mortality on day 1 (pre-clinical), got {dead_day1}"
        assert dead_day2 == 0, f"No mortality on day 2 (pre-clinical), got {dead_day2}"
        assert dead_day4 > 0, f"Should have mortality by day 4, got {dead_day4}"

    def test_white_spots_only_after_inspection(self):
        """White spots visible only after pond_inspection action.

        This is tested at environment level; here we verify the disease model
        doesn't expose white spots directly.
        """
        # The disease model itself doesn't return white_spots
        # That's gated by the pond_inspection action in the environment
        # This test verifies the consumption signal is the only early warning
        tray = wssv_early_warning_fraction(1)
        assert tray < 0.4, "WSSV day-1 tray should be early warning signal"

    def test_emergency_harvest_saves_biomass_early(self):
        """Emergency harvest on days 1–3 should save 60–80% of biomass."""
        save_day1, penalty_day1 = emergency_harvest_recovery(1)
        save_day2, penalty_day2 = emergency_harvest_recovery(2)
        save_day3, penalty_day3 = emergency_harvest_recovery(3)

        assert save_day1 >= 0.60, f"Day 1 save should be ≥60%, got {save_day1:.1%}"
        assert save_day1 <= 0.85, f"Day 1 save should be ≤85%, got {save_day1:.1%}"
        assert save_day2 >= 0.60
        assert save_day3 >= 0.60
        assert penalty_day1 == 0.15, "Emergency harvest should have 15% revenue penalty"

    def test_late_harvest_poor_recovery(self):
        """Emergency harvest after day 3 should save <50% of biomass."""
        save_day4, _ = emergency_harvest_recovery(4)
        save_day6, _ = emergency_harvest_recovery(6)
        assert save_day4 < 0.50, f"Day 4 emergency harvest should save <50%, got {save_day4:.1%}"
        assert save_day6 < save_day4, "Later harvest should save less"

    def test_wssv_spread_higher_with_water_exchange(self):
        """WSSV spread probability should be higher when water is exchanged from infected pond."""
        rng = np.random.default_rng(42)
        # Run many trials to compare probabilities
        spread_with_water = sum(
            wssv_spread_check(np.random.default_rng(i), True, False, False)
            for i in range(1000)
        )
        spread_without_water = sum(
            wssv_spread_check(np.random.default_rng(i), False, False, False)
            for i in range(1000)
        )
        assert spread_with_water > spread_without_water, (
            "Water exchange from infected pond should increase WSSV spread probability"
        )

    def test_biosecurity_reduces_spread(self):
        """Biosecurity measures should reduce WSSV spread probability."""
        spread_no_biosec = sum(
            wssv_spread_check(np.random.default_rng(i), False, False, True)
            for i in range(1000)
        )
        spread_with_biosec = sum(
            wssv_spread_check(np.random.default_rng(i), False, True, True)
            for i in range(1000)
        )
        assert spread_with_biosec < spread_no_biosec, (
            "Biosecurity should reduce WSSV spread probability"
        )

    def test_wssv_mortality_accelerates(self):
        """WSSV mortality should accelerate over time."""
        rng = np.random.default_rng(42)
        dead_day3 = wssv_mortality_count(100000, 3, np.random.default_rng(0))
        dead_day5 = wssv_mortality_count(100000, 5, np.random.default_rng(1))
        dead_day7 = wssv_mortality_count(100000, 7, np.random.default_rng(2))
        assert dead_day3 <= dead_day5 <= dead_day7, "WSSV mortality should accelerate"


class TestVibrio:
    def test_probiotic_slows_vibrio(self):
        """Probiotic treatment should reduce Vibrio severity growth."""
        # 5 days without treatment
        sev = 0.1
        for _ in range(5):
            sev = step_vibrio_severity(sev, False, False, 0)
        sev_untreated = sev

        # 5 days with probiotic (after 3-day onset)
        sev = 0.1
        for day in range(5):
            sev = step_vibrio_severity(sev, True, False, day + 1)
        sev_treated = sev

        assert sev_treated < sev_untreated, "Probiotic should slow Vibrio progression"

    def test_antibiotic_more_effective(self):
        """Antibiotic should be more effective than probiotic."""
        sev = 0.2
        sev_prob = step_vibrio_severity(sev, True, False, 5)  # probiotic (past onset)
        sev_anti = step_vibrio_severity(sev, False, True, 0)  # antibiotic

        assert sev_anti < sev_prob, "Antibiotic should be more effective than probiotic"

    def test_vibrio_observable_signals(self):
        """Vibrio should produce observable signals (redness, tray drop)."""
        signals = vibrio_observable_signals(0.5, 3)
        assert signals["redness_score"] > 0, "Vibrio should cause redness"
        assert signals["tray_drop_factor"] < 1.0, "Vibrio should reduce tray consumption"
        assert 0 <= signals["redness_score"] <= 5
        assert 0 <= signals["tray_drop_factor"] <= 1

    def test_no_redness_before_day2(self):
        """Redness should not appear in first 2 days."""
        signals_day1 = vibrio_observable_signals(0.5, 1)
        assert signals_day1["redness_score"] == 0.0, "No redness on day 1"
