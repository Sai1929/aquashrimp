"""AquaShrimp: Precision shrimp (prawn) aquaculture operations OpenEnv environment.

Primary species: Litopenaeus vannamei (Whiteleg shrimp) — 80% of global production.

Tasks:
  1. NurseryPond    (Easy,   30 steps) — single pond, DO management, feed tray monitoring
  2. SemiIntensive  (Medium, 60 steps) — 4 ponds, shared aerators, Vibrio outbreak
  3. CommercialGrowOut (Hard, 90 steps) — 10 ponds, WSSV, market timing, export compliance
"""

__version__ = "1.0.0"

from aquashrimp.tasks.nursery_pond import NurseryPondEnvironment
from aquashrimp.tasks.semi_intensive_farm import SemiIntensiveFarmEnvironment
from aquashrimp.tasks.commercial_grow_out import CommercialGrowOutEnvironment

__all__ = [
    "NurseryPondEnvironment",
    "SemiIntensiveFarmEnvironment",
    "CommercialGrowOutEnvironment",
]
