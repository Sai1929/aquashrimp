"""Action dataclasses for all three AquaShrimp tasks."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from aquashrimp.models.enums import TreatmentType, HarvestType


@dataclass
class NurseryPondAction:
    """Task 1: Single nursery pond daily management action."""
    feed_kg: float = 5.0               # daily feed amount (0–50 kg for 1,000 m² pond)
    feeding_frequency: int = 4         # meals per day (2–6)
    aeration_hours: float = 20.0       # paddle wheel hours per day (0–24)
    water_exchange_frac: float = 0.05  # fraction of pond volume exchanged per day (0–0.15)
    check_feeding_trays: bool = False   # reveals tray_consumption_fraction
    lime_application_kg: float = 0.0   # Ca(OH)2 per hectare (0–20 kg/ha)

    def validate(self) -> None:
        if not (0.0 <= self.feed_kg <= 50.0):
            raise ValueError(f"feed_kg must be 0–50, got {self.feed_kg}")
        if not (2 <= self.feeding_frequency <= 6):
            raise ValueError(f"feeding_frequency must be 2–6, got {self.feeding_frequency}")
        if not (0.0 <= self.aeration_hours <= 24.0):
            raise ValueError(f"aeration_hours must be 0–24, got {self.aeration_hours}")
        if not (0.0 <= self.water_exchange_frac <= 0.15):
            raise ValueError(f"water_exchange_frac must be 0–0.15, got {self.water_exchange_frac}")
        if not (0.0 <= self.lime_application_kg <= 20.0):
            raise ValueError(f"lime_application_kg must be 0–20, got {self.lime_application_kg}")


@dataclass
class PondFeedAction:
    """Per-pond feeding specification."""
    pond_id: int
    feed_kg: float = 5.0
    frequency: int = 4  # meals per day

    def validate(self) -> None:
        if not (0.0 <= self.feed_kg <= 500.0):
            raise ValueError(f"feed_kg must be 0–500, got {self.feed_kg}")
        if not (2 <= self.frequency <= 6):
            raise ValueError(f"frequency must be 2–6, got {self.frequency}")


@dataclass
class PartialHarvestAction:
    """Selective harvest of large-sized shrimp from a pond."""
    pond_id: int
    size_threshold_g: float  # harvest shrimp above this weight (grams)
    fraction: float = 0.5    # fraction of eligible shrimp to harvest (0–1)

    def validate(self) -> None:
        if not (0.0 < self.fraction <= 1.0):
            raise ValueError(f"fraction must be (0, 1], got {self.fraction}")
        if self.size_threshold_g <= 0:
            raise ValueError(f"size_threshold_g must be positive, got {self.size_threshold_g}")


@dataclass
class SemiIntensiveFarmAction:
    """Task 2: 4-pond farm with shared aeration capacity."""
    pond_feeds: list[PondFeedAction] = field(default_factory=list)
    aeration_allocation: dict[int, float] = field(default_factory=dict)  # sum ≤ 1.0
    water_exchange: dict[int, float] = field(default_factory=dict)        # per pond (0–0.15)
    check_trays: dict[int, bool] = field(default_factory=dict)
    lime_per_pond: dict[int, float] = field(default_factory=dict)         # kg/ha per pond
    probiotic_ponds: list[int] = field(default_factory=list)
    antibiotic_ponds: list[int] = field(default_factory=list)             # triggers export flag!
    partial_harvest: Optional[PartialHarvestAction] = None

    def validate(self) -> None:
        total_aeration = sum(self.aeration_allocation.values())
        if total_aeration > 1.001:
            raise ValueError(f"aeration_allocation sum must be ≤ 1.0, got {total_aeration:.3f}")
        for pond_id, frac in self.water_exchange.items():
            if not (0.0 <= frac <= 0.15):
                raise ValueError(f"water_exchange[{pond_id}] must be 0–0.15, got {frac}")
        for pf in self.pond_feeds:
            pf.validate()
        if self.partial_harvest:
            self.partial_harvest.validate()


@dataclass
class PondTreatmentAction:
    """Treatment application to a specific pond."""
    pond_id: int
    treatment: TreatmentType


@dataclass
class HarvestAction:
    """Harvest specification for a pond."""
    pond_id: int
    harvest_type: HarvestType
    size_threshold_g: Optional[float] = None  # for PARTIAL
    fraction: Optional[float] = None          # for PARTIAL


@dataclass
class CommercialGrowOutAction:
    """Task 3: 10-pond commercial farm with full disease/market management."""
    pond_feeds: list[PondFeedAction] = field(default_factory=list)
    aeration_per_pond: dict[int, float] = field(default_factory=dict)  # hours per day (0–24)
    water_exchange: dict[int, float] = field(default_factory=dict)
    check_trays: dict[int, bool] = field(default_factory=dict)
    pond_inspections: list[int] = field(default_factory=list)  # WSSV white-spot check ($5/pond)
    lime_per_pond: dict[int, float] = field(default_factory=dict)
    biosecurity_measure: bool = False  # water inlet disinfection ($20/day)
    treatments: list[PondTreatmentAction] = field(default_factory=list)
    harvests: list[HarvestAction] = field(default_factory=list)
    disinfect_pond: list[int] = field(default_factory=list)  # post-WSSV chlorine + fallowing
    regulatory_report: bool = False  # required within 48h of WSSV confirmation

    def validate(self) -> None:
        for pond_id, hours in self.aeration_per_pond.items():
            if not (0.0 <= hours <= 24.0):
                raise ValueError(f"aeration_per_pond[{pond_id}] must be 0–24, got {hours}")
        for pond_id, frac in self.water_exchange.items():
            if not (0.0 <= frac <= 0.15):
                raise ValueError(f"water_exchange[{pond_id}] must be 0–0.15, got {frac}")
        for pf in self.pond_feeds:
            pf.validate()
