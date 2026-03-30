"""Weather and seasonal temperature simulation for shrimp ponds.

Covers: seasonal temperature baseline, heavy rainfall events, heat waves.
Calibrated for tropical/subtropical shrimp production regions
(India, Vietnam, Thailand, Ecuador).
"""
from __future__ import annotations
import numpy as np
from aquashrimp.models.enums import WeatherEvent


# Seasonal temperature profile — tropical average
# Baseline centered on 27°C with gentle sinusoidal variation
TEMP_MEAN = 27.0      # °C annual mean
TEMP_AMPLITUDE = 2.5  # °C seasonal amplitude
TEMP_NOISE_STD = 0.5  # °C daily noise

# Monsoon/rainfall window (day 30–70 of a 90-day grow-out)
MONSOON_START_DAY = 30
MONSOON_END_DAY = 70
RAIN_PROB_BASE = 0.03           # P(heavy rain) per day during monsoon
RAIN_PROB_OFF_SEASON = 0.005    # P(heavy rain) outside monsoon

# Heat wave parameters
HEAT_WAVE_PROB = 0.02           # P(heat wave start) days 45–75
HEAT_WAVE_DURATION = (2, 5)     # days range
HEAT_WAVE_TEMP_SPIKE = (2.0, 4.0)  # °C range


def seasonal_temperature(day: int, rng: np.random.Generator) -> float:
    """Compute daily temperature with seasonal variation and noise.

    Day 0 = start of grow-out cycle (typically post-monsoon planting).
    """
    # Slight warming trend during tropical grow-out
    seasonal = TEMP_MEAN + TEMP_AMPLITUDE * np.sin(2 * np.pi * day / 365.0)
    noise = rng.normal(0.0, TEMP_NOISE_STD)
    return round(float(seasonal + noise), 2)


def check_rainfall_event(
    day: int,
    rng: np.random.Generator,
    task_id: int = 3,
) -> tuple[bool, float]:
    """Determine if a heavy rain event occurs today.

    Returns:
        (rain_occurred, rainfall_cm)
    """
    if task_id < 3:
        # Tasks 1 and 2 have no rainfall events
        return False, 0.0

    in_monsoon = MONSOON_START_DAY <= day <= MONSOON_END_DAY
    prob = RAIN_PROB_BASE if in_monsoon else RAIN_PROB_OFF_SEASON

    if rng.random() < prob:
        rainfall_cm = float(rng.uniform(2.0, 10.0))  # 2–10 cm event
        return True, rainfall_cm

    return False, 0.0


def salinity_drop_from_rain(rainfall_cm: float) -> float:
    """Salinity drop from heavy rainfall: −0.5 ppt per cm."""
    return rainfall_cm * 0.5


def check_heat_wave(
    day: int,
    rng: np.random.Generator,
    current_heat_wave_active: bool = False,
    heat_wave_days_remaining: int = 0,
    task_id: int = 3,
) -> tuple[bool, int, float]:
    """Determine heat wave status.

    Returns:
        (heat_wave_active, days_remaining, temp_spike_c)
    """
    if task_id < 3:
        return False, 0, 0.0

    if current_heat_wave_active:
        remaining = heat_wave_days_remaining - 1
        if remaining <= 0:
            return False, 0, 0.0
        spike = rng.uniform(*HEAT_WAVE_TEMP_SPIKE)
        return True, remaining, float(spike)

    # New heat wave possible days 45–75
    if 45 <= day <= 75:
        if rng.random() < HEAT_WAVE_PROB:
            duration = int(rng.integers(*HEAT_WAVE_DURATION))
            spike = float(rng.uniform(*HEAT_WAVE_TEMP_SPIKE))
            return True, duration, spike

    return False, 0, 0.0


def weather_impacts(
    event: WeatherEvent,
    rainfall_cm: float = 0.0,
    heat_spike_c: float = 0.0,
) -> dict:
    """Return impact parameters for a weather event."""
    impacts = {
        "temp_delta": 0.0,
        "rainfall_cm": 0.0,
        "salinity_delta": 0.0,
        "ph_delta": 0.0,
        "osmotic_stress_mortality_frac": 0.0,
        "appetite_factor": 1.0,
    }

    if event == WeatherEvent.HEAVY_RAIN:
        impacts["rainfall_cm"] = rainfall_cm
        impacts["salinity_delta"] = -salinity_drop_from_rain(rainfall_cm)
        impacts["ph_delta"] = float(-0.3 * min(rainfall_cm / 3.0, 1.0))
        # Osmotic stress mortality: 0.5–2% if salinity drops below 5 ppt threshold
        impacts["osmotic_stress_mortality_frac"] = 0.01  # 1% (actual value computed per-pond)
        impacts["appetite_factor"] = 0.8

    elif event == WeatherEvent.HEAT_WAVE:
        impacts["temp_delta"] = heat_spike_c
        impacts["appetite_factor"] = max(0.5, 1.0 - heat_spike_c * 0.15)

    return impacts


def determine_weather_event(
    day: int,
    rng: np.random.Generator,
    heat_wave_active: bool,
    heat_wave_remaining: int,
    task_id: int = 3,
) -> tuple[WeatherEvent, float, float, bool, int]:
    """Single function to determine today's weather state.

    Returns:
        (event, rainfall_cm, heat_spike_c, heat_wave_active, heat_wave_remaining)
    """
    rain, rainfall_cm = check_rainfall_event(day, rng, task_id)
    heat_wave_active, heat_wave_remaining, heat_spike = check_heat_wave(
        day, rng, heat_wave_active, heat_wave_remaining, task_id
    )

    if rain:
        event = WeatherEvent.HEAVY_RAIN
    elif heat_wave_active:
        event = WeatherEvent.HEAT_WAVE
    else:
        event = WeatherEvent.NONE

    return event, rainfall_cm, heat_spike, heat_wave_active, heat_wave_remaining
