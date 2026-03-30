"""Tests for water quality dynamics — critical invariants from the plan."""
import pytest
from aquashrimp.simulation.water_quality import (
    WQState, step_do, step_tan, step_alkalinity, step_ph, step_salinity,
    step_water_quality, do_saturation,
)


class TestDODynamics:
    def test_do_crash_without_aeration(self):
        """Critical invariant: nighttime DO crash without aeration.

        Plan says: DO drops ≥1.5 mg/L per 6 hours at biomass 150 kg/pond.
        """
        wq = WQState(
            do_mg_L=8.0, tan_mg_L=0.05, ph=8.0, salinity_ppt=15.0,
            alkalinity_mg_L=120.0, temp_c=28.0, volume_m3=1000.0
        )
        initial_do = wq.do_mg_L

        # 6 nighttime hours without aeration, 150 kg biomass
        for _ in range(6):
            wq = step_water_quality(
                wq, feed_fed_kg=0, uneaten_kg=0, biomass_kg=150,
                aerator_on=False, exchange_frac=0, is_night_hour=True
            )

        do_drop = initial_do - wq.do_mg_L
        assert do_drop >= 1.5, (
            f"DO should drop ≥1.5 mg/L after 6 night hours without aeration, "
            f"got drop={do_drop:.2f} mg/L (DO={wq.do_mg_L:.2f})"
        )
        assert wq.do_mg_L < 5.0, f"DO should crash below 5.0, got {wq.do_mg_L:.2f}"

    def test_aeration_recovers_do(self):
        """Aeration should improve DO toward saturation."""
        do_low = step_do(
            do_mg_L=4.0, temp_c=27.0, salinity_ppt=15.0,
            biomass_kg=50.0, aerator_hours=20.0, volume_m3=1500.0
        )
        do_none = step_do(
            do_mg_L=4.0, temp_c=27.0, salinity_ppt=15.0,
            biomass_kg=50.0, aerator_hours=0.0, volume_m3=1500.0
        )
        assert do_low > do_none, "Aeration should improve DO"

    def test_do_capped_at_saturation(self):
        """DO should not exceed saturation."""
        do_sat = do_saturation(27.0, 15.0)
        do_result = step_do(
            do_mg_L=do_sat - 0.1, temp_c=27.0, salinity_ppt=15.0,
            biomass_kg=10.0, aerator_hours=24.0, volume_m3=1500.0
        )
        assert do_result <= do_sat + 0.6, f"DO should not exceed saturation by much, got {do_result:.2f}"

    def test_do_never_negative(self):
        """DO should never go below zero."""
        do_result = step_do(
            do_mg_L=0.1, temp_c=30.0, salinity_ppt=20.0,
            biomass_kg=500.0, aerator_hours=0.0, volume_m3=1000.0
        )
        assert do_result >= 0.0

    def test_do_saturation_decreases_with_temperature(self):
        """Hotter water holds less dissolved oxygen."""
        do_cold = do_saturation(20.0)
        do_warm = do_saturation(30.0)
        assert do_cold > do_warm, "DO saturation should decrease with temperature"


class TestTANDynamics:
    def test_overfeeding_causes_tan_spike(self):
        """Critical invariant: 3× overfeeding with no exchange → TAN > 0.5 mg/L in 5 days."""
        tan = 0.02
        volume_m3 = 1500.0
        demand_kg = 5.0
        feed_kg = demand_kg * 3  # 3× overfeeding

        for _ in range(5):
            uneaten = feed_kg * 0.5  # assume 50% uneaten when overfed
            tan = step_tan(
                tan, feed_fed_kg=feed_kg, uneaten_kg=uneaten,
                biomass_kg=100.0, volume_m3=volume_m3, water_exchange_frac=0.0
            )

        assert tan > 0.5, (
            f"3× overfeeding with no exchange should spike TAN > 0.5 mg/L in 5 days, got {tan:.3f}"
        )

    def test_water_exchange_reduces_tan(self):
        """Water exchange should dilute TAN."""
        tan_no_exchange = step_tan(0.3, 5.0, 1.0, 100.0, 1500.0, water_exchange_frac=0.0)
        tan_with_exchange = step_tan(0.3, 5.0, 1.0, 100.0, 1500.0, water_exchange_frac=0.1)
        assert tan_with_exchange < tan_no_exchange, "Water exchange should reduce TAN"

    def test_tan_never_negative(self):
        """TAN should never go negative."""
        tan = step_tan(0.01, 0.0, 0.0, 0.0, 1500.0, water_exchange_frac=0.1)
        assert tan >= 0.0


class TestAlkalinityDynamics:
    def test_lime_increases_alkalinity(self):
        """Lime application should increase alkalinity."""
        alk_no_lime = step_alkalinity(100.0, 0.1, lime_kg_per_ha=0.0)
        alk_lime = step_alkalinity(100.0, 0.1, lime_kg_per_ha=10.0)
        assert alk_lime > alk_no_lime, "Lime should increase alkalinity"

    def test_nitrification_reduces_alkalinity(self):
        """Nitrification should consume alkalinity."""
        alk_no_tan = step_alkalinity(120.0, 0.0)
        alk_with_tan = step_alkalinity(120.0, 0.5)
        assert alk_with_tan <= alk_no_tan, "High TAN nitrification should reduce alkalinity"

    def test_alkalinity_floor(self):
        """Alkalinity should have a minimum floor."""
        alk = step_alkalinity(25.0, 2.0, lime_kg_per_ha=0.0)
        assert alk >= 20.0, "Alkalinity should not drop below 20 mg/L floor"


class TestSalinityDynamics:
    def test_rainfall_reduces_salinity(self):
        """Rainfall should decrease salinity."""
        sal_dry = step_salinity(15.0, 27.0, rainfall_cm=0.0)
        sal_rain = step_salinity(15.0, 27.0, rainfall_cm=5.0)
        assert sal_rain < sal_dry, "Rainfall should reduce salinity"
        # 5 cm rain → −2.5 ppt drop
        drop = sal_dry - sal_rain
        assert drop >= 2.0, f"5cm rain should drop salinity ≥2 ppt, got {drop:.2f}"

    def test_evaporation_increases_salinity(self):
        """Hot weather evaporation should increase salinity."""
        sal_warm = step_salinity(15.0, 30.0, rainfall_cm=0.0)
        sal_cool = step_salinity(15.0, 22.0, rainfall_cm=0.0)
        assert sal_warm >= sal_cool, "Warm temperature should concentrate salinity via evaporation"

    def test_salinity_bounds(self):
        """Salinity should always be in valid range."""
        sal = step_salinity(15.0, 27.0, rainfall_cm=100.0)
        assert sal >= 0.1, "Salinity should not drop below 0.1"
        assert sal <= 50.0, "Salinity should not exceed 50 ppt"
