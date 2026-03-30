"""Observation dataclasses returned to the agent after each step."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from aquashrimp.models.enums import WeatherEvent, DiseaseType, FallowStatus


@dataclass
class RewardBreakdown:
    """Per-component reward breakdown (dense signal every step)."""
    total: float
    growth: float
    water_quality: float
    economic: float
    biosecurity: float


@dataclass
class NurseryPondObs:
    """Task 1 observation — single nursery pond."""
    # Episode metadata
    day: int
    done: bool
    reward: float
    reward_breakdown: RewardBreakdown

    # Shrimp population
    n_shrimp: int
    mean_weight_g: float
    biomass_kg: float
    survival_rate: float        # fraction of original stock surviving
    mortality_today: int        # shrimp died today
    molt_event_today: bool      # molting occurred today (cannibalism risk)

    # Water quality (observable)
    temperature_c: float
    do_mg_L: float              # dissolved oxygen
    tan_mg_L: float             # total ammonia nitrogen
    ph: float
    salinity_ppt: float
    alkalinity_mg_L: float
    h2s_risk_score: float       # 0–1, proxy for bottom sediment gas risk
    secchi_depth_cm: float      # 0 = turbid, 60 = clear

    # Feeding
    tray_consumption_fraction: Optional[float]  # None if check_feeding_trays=False
    fcr_cumulative: float                       # cumulative feed conversion ratio
    feed_demand_estimate_kg: float              # estimated daily demand for planning

    # Economics
    feed_cost_today_usd: float
    energy_cost_today_usd: float
    cumulative_cost_usd: float

    # Weather / Grade
    weather_event: WeatherEvent = WeatherEvent.NONE
    grade: float = 0.5              # normalized episode grade [0.0, 1.0]


@dataclass
class PondObs:
    """Per-pond observation for multi-pond tasks."""
    pond_id: int
    n_shrimp: int
    mean_weight_g: float
    biomass_kg: float
    survival_rate: float
    mortality_today: int
    molt_event_today: bool

    # Water quality
    temperature_c: float
    do_mg_L: float
    tan_mg_L: float
    ph: float
    salinity_ppt: float
    alkalinity_mg_L: float
    h2s_risk_score: float
    secchi_depth_cm: float

    # Feeding
    tray_consumption_fraction: Optional[float]
    fcr_cumulative: float
    feed_demand_estimate_kg: float

    # Derived
    density_kg_m2: float = 0.0              # biomass density (kg/m²)

    # Disease signals
    disease_type: DiseaseType = DiseaseType.NONE
    redness_score: float = 0.0          # Vibrio signal (0–5)
    swimming_behavior_score: float = 1.0  # 1.0 = normal, <0.5 = abnormal (WSSV signal)
    white_spots_visible: bool = False   # only True after pond_inspection action
    disease_severity: float = 0.0      # internal-ish severity proxy (0–1)

    # Status
    fallow_status: FallowStatus = FallowStatus.ACTIVE
    fallow_days_remaining: int = 0

    # Export
    export_compliance_flag: bool = False


@dataclass
class SemiIntensiveFarmObs:
    """Task 2 observation — 4-pond semi-intensive farm."""
    day: int
    done: bool
    reward: float
    reward_breakdown: RewardBreakdown

    ponds: list[PondObs]
    aeration_capacity_remaining: float  # fraction of total aeration unused

    # Farm-level economics
    total_biomass_kg: float
    feed_cost_today_usd: float
    energy_cost_today_usd: float
    treatment_cost_today_usd: float
    cumulative_cost_usd: float

    # Market
    vannamei_price_usd_per_kg: float

    # Weather / Grade / Compliance
    weather_event: WeatherEvent = WeatherEvent.NONE
    grade: float = 0.5              # normalized episode grade [0.0, 1.0]
    antibiotic_used_this_episode: bool = False


@dataclass
class CommercialGrowOutObs:
    """Task 3 observation — 10-pond commercial farm."""
    day: int
    done: bool
    reward: float
    reward_breakdown: RewardBreakdown

    ponds: list[PondObs]

    # Farm-level economics
    total_biomass_kg: float
    feed_cost_today_usd: float
    energy_cost_today_usd: float
    treatment_cost_today_usd: float
    inspection_cost_today_usd: float
    biosecurity_cost_today_usd: float
    cumulative_cost_usd: float
    cumulative_revenue_usd: float

    # Market
    vannamei_price_usd_per_kg: float
    feed_price_usd_per_kg: float

    # Disease pressure / Grade
    grade: float = 0.5              # normalized episode grade [0.0, 1.0]
    neighbor_outbreak: bool = False   # external WSSV pressure from neighboring farm
    wssv_confirmed_ponds: list[int] = field(default_factory=list)
    regulatory_report_overdue: bool = False
    report_overdue_days: int = 0

    # Compliance
    antibiotic_used_this_episode: bool = False
    export_ban_risk: bool = False

    # Weather
    weather_event: WeatherEvent = WeatherEvent.NONE
