"""Water quality dynamics for L. vannamei shrimp ponds.

Includes: DO diurnal cycle, TAN/ammonia, pH, salinity, alkalinity, H2S.
Shrimp parameters differ significantly from fish (especially TAN sensitivity and salinity).
"""
from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class WQState:
    """Water quality state for one pond."""
    do_mg_L: float
    tan_mg_L: float
    ph: float
    salinity_ppt: float
    alkalinity_mg_L: float
    temp_c: float
    volume_m3: float
    h2s_risk: float = 0.0
    nitrite_mg_L: float = 0.0
    secchi_depth_cm: float = 35.0


def do_saturation(temp_c: float, salinity_ppt: float = 0.0) -> float:
    """DO saturation value (mg/L) — decreases with temperature and salinity.

    Uses simplified empirical formula valid for aquaculture range.
    """
    # Base saturation at given temperature (freshwater)
    do_sat = 14.62 - 0.3898 * temp_c + 0.006969 * temp_c ** 2 - 0.00005896 * temp_c ** 3
    # Salinity correction (~0.03 mg/L reduction per ppt)
    do_sat -= 0.03 * salinity_ppt
    return max(6.0, do_sat)


def step_do(
    do_mg_L: float,
    temp_c: float,
    salinity_ppt: float,
    biomass_kg: float,
    aerator_hours: float,
    volume_m3: float,
    is_daytime: bool = True,
    water_exchange_frac: float = 0.0,
    source_do: float = 7.0,
) -> float:
    """Update DO for one daily step.

    Args:
        do_mg_L: Current DO
        temp_c: Water temperature
        salinity_ppt: Salinity
        biomass_kg: Total shrimp biomass
        aerator_hours: Hours of aeration today (0–24)
        volume_m3: Pond volume
        is_daytime: Daytime = algae photosynthesis produces O2
        water_exchange_frac: Fraction exchanged with source water
        source_do: Source water DO (typically ~7 mg/L)

    Returns:
        New DO (mg/L)
    """
    do_sat = do_saturation(temp_c, salinity_ppt)

    # Nighttime respiration consumption (shrimp + algae)
    # Simplified daily model: 12h day / 12h night cycle assumed
    night_hours = 24.0 - aerator_hours  # hours when aerators may be off
    night_hours = max(0.0, min(12.0, 24.0 - aerator_hours))

    # DO drop from shrimp respiration (exponential oxygen demand with temperature)
    respiration_rate = 0.004 * (1.02 ** temp_c)  # mg/L per kg biomass per hour
    do_consumed = biomass_kg * respiration_rate * 24.0 / volume_m3

    # Algae respiration at night adds ~40% extra consumption
    do_consumed_night_extra = do_consumed * 0.4 * (night_hours / 12.0)
    total_do_consumed = do_consumed + do_consumed_night_extra

    # Aerator recovery: each aerator-hour recovers oxygen proportional to deficit
    aeration_recovery = aerator_hours * 0.8 * (do_sat - do_mg_L) / volume_m3 * 1000.0
    aeration_recovery = max(0.0, aeration_recovery)

    # Daytime algae photosynthesis adds DO
    if is_daytime:
        # Secchi-dependent; assume moderate algae bloom (phytoplankton density)
        photosynthesis = 0.5 * (do_sat - do_mg_L) * 0.3  # simplified
    else:
        photosynthesis = 0.0

    new_do = do_mg_L - total_do_consumed + aeration_recovery + photosynthesis

    # Water exchange: mix with source water
    if water_exchange_frac > 0:
        new_do = new_do * (1 - water_exchange_frac) + source_do * water_exchange_frac

    return max(0.0, min(do_sat + 0.5, new_do))


def step_tan(
    tan_mg_L: float,
    feed_fed_kg: float,
    uneaten_kg: float,
    biomass_kg: float,
    volume_m3: float,
    water_exchange_frac: float = 0.0,
    nitrification_rate: float = 0.15,
    source_tan: float = 0.02,
) -> float:
    """Update total ammonia nitrogen (TAN).

    Shrimp excretion + uneaten feed decomposition - nitrification - exchange.
    """
    # Ammonia production from shrimp excretion and uneaten feed
    tan_produced = (0.025 * feed_fed_kg + 0.008 * uneaten_kg) * 1000.0 / volume_m3

    # Nitrification removes TAN (converts to nitrate; consuming alkalinity)
    nitrification_removed = tan_mg_L * nitrification_rate

    # Water exchange
    exchange_effect = (source_tan - tan_mg_L) * water_exchange_frac

    new_tan = tan_mg_L + tan_produced - nitrification_removed + exchange_effect
    return max(0.0, new_tan)


def step_alkalinity(
    alkalinity_mg_L: float,
    tan_mg_L: float,
    nitrification_rate: float = 0.15,
    lime_kg_per_ha: float = 0.0,
    pond_area_ha: float = 0.1,
    water_exchange_frac: float = 0.0,
    source_alkalinity: float = 120.0,
) -> float:
    """Update alkalinity (mg/L as CaCO3).

    Nitrification consumes alkalinity; lime addition restores it.
    """
    # Each g NH4-N nitrified consumes 7.1 g CaCO3
    tan_nitrified = tan_mg_L * nitrification_rate
    alk_consumed = tan_nitrified * 7.1

    # Lime application restores alkalinity
    # 10 kg Ca(OH)2/ha adds ~10 mg/L alkalinity
    lime_boost = (lime_kg_per_ha / 10.0) * 10.0

    # Water exchange
    exchange_effect = (source_alkalinity - alkalinity_mg_L) * water_exchange_frac

    new_alk = alkalinity_mg_L - alk_consumed + lime_boost + exchange_effect
    return max(20.0, new_alk)


def step_ph(
    ph: float,
    alkalinity_mg_L: float,
    tan_mg_L: float,
    is_daytime: bool = True,
    rainfall_cm: float = 0.0,
    water_exchange_frac: float = 0.0,
    source_ph: float = 8.0,
) -> float:
    """Update pH.

    Alkalinity buffers pH; algae photosynthesis raises pH during day;
    nitrification and rainfall lower pH.
    """
    # Algae photosynthesis raises pH in daytime (CO2 removal)
    if is_daytime:
        ph_change = +0.1
    else:
        # Respiration lowers pH at night (CO2 production)
        ph_change = -0.05

    # Low alkalinity = less buffering = more pH swing
    buffering = min(1.0, alkalinity_mg_L / 100.0)
    ph_change *= buffering

    # Nitrification acidifies water
    if tan_mg_L > 0.1:
        ph_change -= 0.02 * (tan_mg_L / 0.1)

    # Rainfall acidifies (CO2 from rainwater)
    if rainfall_cm > 0:
        ph_change -= 0.3 * min(rainfall_cm, 3.0)

    # Water exchange
    if water_exchange_frac > 0:
        ph_change += (source_ph - ph) * water_exchange_frac

    new_ph = ph + ph_change
    return max(5.0, min(10.0, new_ph))


def step_salinity(
    salinity_ppt: float,
    temp_c: float,
    rainfall_cm: float = 0.0,
    water_exchange_frac: float = 0.0,
    source_salinity_ppt: float = 15.0,
    evaporation_factor: float = 1.0,
) -> float:
    """Update salinity (ppt).

    Evaporation concentrates; rainfall dilutes; water exchange toward source.
    """
    # Evaporation: +0.05 ppt per degree above 20°C per day
    if temp_c > 20.0:
        evap_increase = 0.05 * (temp_c - 20.0) * evaporation_factor
    else:
        evap_increase = 0.0

    # Rainfall dilution
    rainfall_decrease = rainfall_cm * 0.5  # −0.5 ppt per cm

    # Water exchange toward source
    exchange_effect = (source_salinity_ppt - salinity_ppt) * water_exchange_frac

    new_sal = salinity_ppt + evap_increase - rainfall_decrease + exchange_effect
    return max(0.1, min(50.0, new_sal))


def step_h2s_risk(
    h2s_risk: float,
    feed_fed_kg: float,
    biomass_kg: float,
    area_m2: float,
    days_since_cleaned: int,
    bottom_aeration: bool = False,
) -> float:
    """Update H2S risk score (0–1).

    Bottom sediment accumulates organic matter from uneaten feed + feces.
    H2S produced by anaerobic decomposition.
    """
    feed_rate_factor = min(2.0, (feed_fed_kg / (area_m2 * 0.01)) if area_m2 > 0 else 1.0)

    # H2S builds up slowly over time
    accumulation = 0.001 * max(0, days_since_cleaned - 30) * feed_rate_factor
    accumulation += 0.0005 * (biomass_kg / area_m2) if area_m2 > 0 else 0.0

    # Aeration breaks down H2S
    if bottom_aeration:
        decay = h2s_risk * 0.3
    else:
        decay = 0.0

    new_risk = h2s_risk + accumulation - decay
    return max(0.0, min(1.0, new_risk))


def compute_secchi_depth(
    tan_mg_L: float,
    salinity_ppt: float,
    biomass_kg: float,
    volume_m3: float,
    feed_fed_kg: float,
) -> float:
    """Estimate Secchi depth (cm) as proxy for phytoplankton density.

    Optimal range: 30–40 cm for L. vannamei.
    Too clear (<25 cm) = too much algae = DO crash at night.
    Too turbid (>50 cm) = poor water quality.
    """
    biomass_density = biomass_kg / volume_m3 if volume_m3 > 0 else 0
    nutrient_load = tan_mg_L + (feed_fed_kg / volume_m3 * 100 if volume_m3 > 0 else 0)

    # High nutrients = more algae = lower Secchi depth
    base_secchi = 45.0 - nutrient_load * 5.0 - biomass_density * 10.0
    return max(10.0, min(80.0, base_secchi))


def step_water_quality(
    wq: WQState,
    feed_fed_kg: float,
    uneaten_kg: float,
    biomass_kg: float,
    aerator_on: bool,
    exchange_frac: float = 0.0,
    is_night_hour: bool = False,
    lime_kg_per_ha: float = 0.0,
    pond_area_ha: float = 0.1,
    rainfall_cm: float = 0.0,
    source_salinity: float = 15.0,
) -> WQState:
    """Hourly water quality step (for DO crash test verification).

    Used in tests to verify nighttime DO crash behavior.
    """
    do_sat = do_saturation(wq.temp_c, wq.salinity_ppt)
    # Plan formula: DO_night_drop = biomass_kg × 0.004 × 1.02^T [mg/L per hour]
    respiration_rate = 0.004 * (1.02 ** wq.temp_c)
    do_consumed = biomass_kg * respiration_rate  # mg/L per hour (plan formula, no volume division)

    if is_night_hour:
        do_consumed *= 1.4  # Night: algae also respire

    if aerator_on:
        aeration = 0.8 * (do_sat - wq.do_mg_L)
    else:
        aeration = 0.0

    new_do = wq.do_mg_L - do_consumed + aeration
    if exchange_frac > 0:
        new_do = new_do * (1 - exchange_frac) + 7.0 * exchange_frac
    new_do = max(0.0, min(do_sat + 0.5, new_do))

    # Return updated state
    from copy import deepcopy
    new_wq = deepcopy(wq)
    new_wq.do_mg_L = new_do
    return new_wq
