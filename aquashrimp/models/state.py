"""Internal episode state (hidden from agent — ground truth simulation state)."""
from __future__ import annotations
from dataclasses import dataclass, field
from aquashrimp.models.enums import DiseaseType, FallowStatus


@dataclass
class PondState:
    """Full internal state of a single pond."""
    pond_id: int
    area_m2: float
    volume_m3: float

    # Shrimp population
    n_shrimp: int
    mean_weight_g: float
    initial_stocking: int

    # Water quality (true values)
    temperature_c: float = 27.0
    do_mg_L: float = 7.0
    tan_mg_L: float = 0.05
    ph: float = 8.0
    salinity_ppt: float = 15.0
    alkalinity_mg_L: float = 120.0
    h2s_risk: float = 0.0      # 0–1 risk score
    secchi_depth_cm: float = 35.0

    # Molting
    days_since_molt: float = 0.0
    inter_molt_period: float = 7.5  # recalculated each step

    # Disease
    disease_type: DiseaseType = DiseaseType.NONE
    disease_severity: float = 0.0   # 0–1
    days_since_infection: int = 0
    wssv_detected: bool = False     # true if tray consumption signal crossed threshold
    wssv_confirmed: bool = False    # true only after pond_inspection
    export_compliance_flag: bool = False  # True if antibiotics used

    # Fallow
    fallow_status: FallowStatus = FallowStatus.ACTIVE
    fallow_days_remaining: int = 0
    days_since_bottom_cleaned: int = 0

    # Economics tracking
    cumulative_feed_kg: float = 0.0
    cumulative_biomass_harvested_kg: float = 0.0
    cumulative_revenue_usd: float = 0.0

    @property
    def biomass_kg(self) -> float:
        return self.n_shrimp * self.mean_weight_g / 1000.0

    @property
    def density_kg_m2(self) -> float:
        return self.biomass_kg / self.area_m2

    @property
    def survival_rate(self) -> float:
        if self.initial_stocking == 0:
            return 0.0
        return self.n_shrimp / self.initial_stocking

    @property
    def fcr(self) -> float:
        gained_kg = self.cumulative_biomass_harvested_kg + self.biomass_kg - (
            self.initial_stocking * 0.05 / 1000.0
        )
        if gained_kg <= 0:
            return 0.0
        return self.cumulative_feed_kg / max(gained_kg, 0.001)


@dataclass
class EpisodeState:
    """Full episode state including all ponds and farm-level variables."""
    task_id: int          # 1, 2, or 3
    day: int = 0
    max_steps: int = 30
    done: bool = False
    seed: int = 42

    ponds: list[PondState] = field(default_factory=list)

    # Economics
    vannamei_price_usd_per_kg: float = 5.0
    feed_price_usd_per_kg: float = 1.2
    energy_cost_per_aerator_hour: float = 0.05
    cumulative_cost_usd: float = 0.0
    cumulative_revenue_usd: float = 0.0

    # Disease / biosecurity
    neighbor_outbreak: bool = False
    wssv_trigger_day: int = -1
    vibrio_trigger_day: int = -1
    vibrio_pond_id: int = -1
    regulatory_report_submitted: bool = False
    regulatory_report_due_day: int = -1
    antibiotic_used: bool = False

    # Weather
    current_weather: str = "none"
    weather_duration_days: int = 0
    monsoon_active: bool = False  # Task 3

    # Shared aeration capacity (Task 2)
    total_aeration_capacity_hours: float = 192.0  # 8 aerators × 24h
