"""Enumerations for the AquaShrimp environment."""
from enum import Enum


class TreatmentType(str, Enum):
    PROBIOTIC = "probiotic"        # Efficacy 0.5, no regulatory risk, 3-day onset
    ANTIBIOTIC = "antibiotic"      # Efficacy 0.9, immediate, triggers export_compliance_flag
    LIME = "lime"                  # Alkalinity + pH stabilization
    DISINFECTION = "disinfection"  # Post-WSSV chlorine pond treatment


class HarvestType(str, Enum):
    FULL = "full"            # Complete pond harvest
    PARTIAL = "partial"      # Size-graded partial harvest
    EMERGENCY = "emergency"  # Emergency harvest (WSSV containment, 15% revenue penalty)


class WeatherEvent(str, Enum):
    NONE = "none"
    HEAVY_RAIN = "heavy_rain"   # Salinity drop -2 to -5 ppt, pH drop
    HEAT_WAVE = "heat_wave"     # Temp +2 to +4°C, DO saturation drop


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class DiseaseType(str, Enum):
    NONE = "none"
    VIBRIO = "vibrio"    # Bacterial, Task 2 — treatable
    WSSV = "wssv"        # Viral, Task 3 — no treatment, containment only


class FallowStatus(str, Enum):
    ACTIVE = "active"      # Pond in production
    FALLOWING = "fallowing"  # Post-WSSV disinfection, 30-day rest
    READY = "ready"        # Ready for restocking
