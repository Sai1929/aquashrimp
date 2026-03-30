---
title: AquaShrimp OpenEnv
emoji: 🦐
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# AquaShrimp: Prawn/Shrimp Aquaculture Operations OpenEnv Environment

**The first OpenEnv environment for shrimp aquaculture** — simulating *Litopenaeus vannamei* (Whiteleg shrimp) farm operations for AI agent training.

**Impact**: Shrimp aquaculture is the world's most traded seafood commodity — a $40B/year industry employing 12 million people. A single WSSV outbreak can wipe out an entire pond in 3–10 days. AI-driven farm management could save billions in disease losses and reduce feed waste by 20–30%.

---

## Install

```bash
pip install -e .
```

---

## Three Tasks

| Task | Difficulty | Steps | Ponds | Key Challenge |
|------|-----------|-------|-------|---------------|
| `NurseryPond` | Easy | 30 | 1 (0.1 ha) | Nighttime DO crash, feed tray monitoring |
| `SemiIntensiveFarm` | Medium | 60 | 4 (0.5 ha each) | Shared aerators, Vibrio outbreak, partial harvest |
| `CommercialGrowOut` | Hard | 90 | 10 (2 ha each, 2 sites) | WSSV outbreak, market timing, export compliance |

---

## Quick Start

```python
from aquashrimp.tasks.nursery_pond import NurseryPondEnvironment
from aquashrimp.models.actions import NurseryPondAction

env = NurseryPondEnvironment(seed=42)
obs = env.reset()

for day in range(30):
    action = NurseryPondAction(
        feed_kg=obs.feed_demand_estimate_kg,
        aeration_hours=20.0,
        water_exchange_frac=0.05,
        check_feeding_trays=(day % 2 == 0),
        lime_application_kg=5.0 if obs.ph < 7.8 else 0.0,
    )
    obs = env.step(action)
    print(f"Day {obs.day}: weight={obs.mean_weight_g:.2f}g | DO={obs.do_mg_L:.2f} | reward={obs.reward:+.3f} | grade={obs.grade:.3f}")
    if obs.done:
        break
```

---

## Action Space

### Task 1 — NurseryPond (`NurseryPondAction`)

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `feed_kg` | float | 0–50 | Daily feed amount (kg) |
| `feeding_frequency` | int | 2–6 | Meals per day |
| `aeration_hours` | float | 0–24 | Aeration duration (hours/day) |
| `water_exchange_frac` | float | 0–0.15 | Fraction of pond volume exchanged |
| `check_feeding_trays` | bool | — | Observe tray consumption (+5% noise) |
| `lime_application_kg` | float | 0–20 | Lime added (kg/ha equivalent) |

### Task 2 — SemiIntensiveFarm (`SemiIntensiveFarmAction`)

| Field | Type | Description |
|-------|------|-------------|
| `pond_feeds` | list[PondFeedAction] | Per-pond feed (kg) and frequency |
| `aeration_allocation` | dict[pond_id, float] | Fraction of shared 192 aerator-h/day; sum ≤ 1.0 |
| `water_exchange` | dict[pond_id, float] | Per-pond water exchange fraction (0–0.15) |
| `check_trays` | dict[pond_id, bool] | Per-pond tray observation |
| `lime_per_pond` | dict[pond_id, float] | Per-pond lime (kg) |
| `probiotic_ponds` | list[int] | Apply probiotic treatment (50% efficacy, 3-day onset, no export penalty) |
| `antibiotic_ponds` | list[int] | Apply antibiotic treatment (90% efficacy, immediate, sets export flag) |
| `partial_harvest` | PartialHarvestAction \| None | Size-graded harvest of a single pond |

### Task 3 — CommercialGrowOut (`CommercialGrowOutAction`)

| Field | Type | Description |
|-------|------|-------------|
| `pond_feeds` | list[PondFeedAction] | Per-pond feed (kg) and frequency |
| `aeration_per_pond` | dict[pond_id, float] | Per-pond aeration hours/day (0–24) |
| `water_exchange` | dict[pond_id, float] | Per-pond exchange fraction (0–0.15) |
| `check_trays` | dict[pond_id, bool] | Per-pond tray observation |
| `pond_inspections` | list[int] | Physical inspection — reveals `white_spots_visible` if WSSV |
| `lime_per_pond` | dict[pond_id, float] | Per-pond lime (kg) |
| `biosecurity_measure` | bool | Activate biosecurity ($20/day; reduces neighbor WSSV spread 5%→1%) |
| `treatments` | list[TreatmentAction] | Per-pond treatment (PROBIOTIC / ANTIBIOTIC) |
| `harvests` | list[HarvestAction] | Per-pond harvest (FULL / PARTIAL / EMERGENCY) |
| `disinfect_pond` | list[int] | Disinfect after harvest ($150/pond; required post-WSSV) |
| `regulatory_report` | bool | Submit WSSV regulatory report (required within 48h of confirmation) |

---

## Observation Space

### Task 1 — NurseryPond (`NurseryPondObs`)

| Field | Type | Description |
|-------|------|-------------|
| `day` | int | Current simulation day (1–30) |
| `done` | bool | Episode ended |
| `reward` | float | Per-step reward (−1 to +1) |
| `reward_breakdown` | RewardBreakdown | growth / water / economic / biosecurity components |
| `grade` | float | Normalized episode grade [0.0, 1.0] — (mean_reward+1)/2 |
| `n_shrimp` | int | Current shrimp count |
| `mean_weight_g` | float | Mean individual weight (g) |
| `biomass_kg` | float | Total biomass (kg) |
| `survival_rate` | float | Fraction of original stock alive |
| `mortality_today` | int | Deaths today |
| `molt_event_today` | bool | Molting occurred (cannibalism risk) |
| `temperature_c` | float | Water temperature (°C) |
| `do_mg_L` | float | Dissolved oxygen (mg/L) — critical if < 3 |
| `tan_mg_L` | float | Total ammonia nitrogen (mg/L) — critical if > 0.5 |
| `ph` | float | pH — optimal 7.8–8.3 |
| `salinity_ppt` | float | Salinity (ppt) |
| `alkalinity_mg_L` | float | Alkalinity (mg/L CaCO₃) — critical if < 50 for molting |
| `h2s_risk_score` | float | H₂S risk proxy (0–1) |
| `secchi_depth_cm` | float | Water clarity proxy (0 turbid → 60 clear) |
| `tray_consumption_fraction` | float \| None | Feed tray consumption (None if not checked) |
| `fcr_cumulative` | float | Feed conversion ratio to date |
| `feed_demand_estimate_kg` | float | Estimated daily feed requirement |
| `feed_cost_today_usd` | float | Feed cost today ($) |
| `energy_cost_today_usd` | float | Aeration energy cost today ($) |
| `cumulative_cost_usd` | float | Total costs so far ($) |
| `weather_event` | WeatherEvent | NONE / HEAVY_RAIN / HEAT_WAVE |

### Task 2 & 3 — Per-pond (`PondObs`) additional fields

| Field | Type | Description |
|-------|------|-------------|
| `pond_id` | int | Pond identifier |
| `density_kg_m2` | float | Biomass density (kg/m²) |
| `disease_type` | DiseaseType | NONE / VIBRIO / WSSV |
| `redness_score` | float | Vibrio signal (0–5); > 1.5 = treatment needed |
| `swimming_behavior_score` | float | 1.0 = normal; < 0.5 = WSSV behavioral sign |
| `white_spots_visible` | bool | WSSV confirmed — only True after `pond_inspections` action |
| `disease_severity` | float | Severity proxy (0–1) |
| `fallow_status` | FallowStatus | ACTIVE / FALLOWING / READY |
| `fallow_days_remaining` | int | Days until pond can be restocked |
| `export_compliance_flag` | bool | True if antibiotics used this episode |

---

## What Makes Shrimp Aquaculture Unique

### Feeding Tray Monitoring (Partial Observability)
Shrimp are bottom feeders. Feed trays reveal actual consumption — the primary early warning for disease.

```python
obs = env.step(NurseryPondAction(check_feeding_trays=True))
if obs.tray_consumption_fraction < 0.2:
    print("ALERT: WSSV suspected — emergency harvest window open!")
```

### WSSV Disease Model (No Treatment — Containment Only)
- **Day 1–2** after infection: only tray consumption drops 75–90% (pre-clinical early warning)
- **Day 3**: clinical mortality begins at 20%/day, accelerating to 50–95% by day 5+
- **Emergency harvest days 1–3**: saves 60–80% of biomass
- **Emergency harvest day 4+**: saves <50%

```python
# Task 3: respond to WSSV signal
from aquashrimp.models.actions import CommercialGrowOutAction, HarvestAction
from aquashrimp.models.enums import HarvestType

action = CommercialGrowOutAction(
    harvests=[HarvestAction(pond_id=3, harvest_type=HarvestType.EMERGENCY)],
    disinfect_pond=[3],
    regulatory_report=True,
)
```

### Antibiotic Trade-off (Vibrio vs Export Compliance)
Antibiotics cure Vibrio (90% efficacy, immediate) but set `export_compliance_flag=True` episode-long (−0.3 biosecurity penalty). Probiotics are safer (50% efficacy, 3-day onset, no penalty).

---

## Simulation Physics

### Shrimp Growth (DWG Model — NOT the fish TGC model)
```
DWG = 0.25 g/day × f_temp × f_feed × f_density × f_health

f_temp    = exp(−0.5 × ((T − 27) / 3)²)     [Gaussian, peak 27°C]
f_feed    = min(1.0, feed_ratio / 0.8)^0.6
f_density = max(0.4, 1.0 − 0.003 × (kg/m² − 2.0))
f_health  = max(0.0, 1.0 − 0.3 × disease − 0.2 × stress)
```

### Water Quality (L. vannamei Optimal Ranges)
| Parameter | Optimal | Critical Threshold |
|-----------|---------|-------------------|
| Temperature | 26–28°C | <20°C or >32°C |
| DO | >5 mg/L | <3 mg/L (mass mortality) |
| TAN (NH₃) | <0.05 mg/L | >0.5 mg/L |
| pH | 7.8–8.3 | <7.0 or >9.0 |
| Salinity | 10–25 ppt | <5 ppt (osmoregulation stress) |
| Alkalinity | >100 mg/L CaCO₃ | <50 mg/L (molting failure) |

### Diurnal DO Cycle (Primary Failure Mode)
Without adequate aeration, DO crashes at night:
```
DO_drop_per_hour = biomass_kg × 0.004 × 1.02^T  [mg/L/h]
```

---

## Reward Function

```
R_total = w_growth × R_growth + w_water × R_water + w_econ × R_econ + w_biosec × R_biosec
```

| Component | NurseryPond | SemiIntensive | CommercialGrowOut |
|-----------|------------|---------------|-------------------|
| Growth    | 35% | 30% | 25% |
| Water     | 35% | 25% | 20% |
| Economic  | 20% | 30% | 35% |
| Biosecurity | 10% | 15% | 20% |

All components and total reward range: **[−1.0, +1.0]**. Dense signal every step.

### Grade (normalized score)
```
grade = max(0.0, min(1.0, (mean_per_step_reward + 1.0) / 2.0))
```
Maps mean reward [−1, +1] → grade [0, 1]. Before any steps: grade = 0.5 (neutral).
Available in every observation (`obs.grade`) and via `GET /grade`.

---

## Baseline Scores

Measured with seed=42, 5 episodes. Format: `mean_per_step_reward / grade`.

| Agent | NurseryPond | SemiIntensive | CommercialGrowOut |
|-------|-------------|---------------|-------------------|
| Random | +0.09 / 0.55 | +0.23 / 0.62 | −0.03 / 0.48 |
| Rule-based | +0.21 / 0.61 | +0.37 / 0.69 | +0.10 / 0.55 |
| Near-optimal | +0.26 / 0.63 | +0.40 / 0.70 | +0.11 / 0.56 |

Reproduce these scores:
```bash
# Single combo
python aquashrimp/baselines/run_baseline.py --agent rule --task nursery_pond --seed 42 --episodes 5

# Full benchmark table (all 3 agents × all 3 tasks)
python scripts/benchmark_all.py --seed 42 --episodes 5
```

---

## HTTP API (FastAPI Server)

```bash
# Run locally (Task 1)
AQUASHRIMP_TASK=1 uvicorn aquashrimp.server.app:app --port 7860

# Metadata
curl http://localhost:7860/

# Reset episode
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"seed": 42}'

# Step (Task 1)
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" \
  -d '{"feed_kg": 8.0, "aeration_hours": 20.0, "water_exchange_frac": 0.05, "check_feeding_trays": true}'

# Get normalized grade [0.0, 1.0]
curl http://localhost:7860/grade
# {"grade": 0.607, "task_id": 1, "episode_done": false, "steps": 1, "max_steps": 30}

# Get full episode state (debugging)
curl http://localhost:7860/state
```

Task selection: `AQUASHRIMP_TASK=1` (NurseryPond) | `2` (SemiIntensive) | `3` (CommercialGrowOut)

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Environment metadata and task info |
| GET | `/health` | Health check |
| POST | `/reset` | Start new episode (`{"seed": 42}`) |
| POST | `/step` | Submit action, receive observation + reward |
| GET | `/grade` | Normalized episode grade [0.0, 1.0] |
| GET | `/state` | Full internal state (debugging) |

---

## Docker / HuggingFace Spaces

```bash
# Build
docker build -t aquashrimp .

# Run Task 1 (NurseryPond)
docker run -p 7860:7860 -e AQUASHRIMP_TASK=1 aquashrimp

# Run Task 3 (CommercialGrowOut)
docker run -p 7860:7860 -e AQUASHRIMP_TASK=3 aquashrimp
```

Deploy 3 separate HF Spaces from one image by setting `AQUASHRIMP_TASK=1`, `2`, or `3`.

> **Note**: `WORKERS=1` is required — the environment is stateful (one session per process). Scale by running multiple containers.

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v  # 92 tests, 7 files
```

Critical invariants verified:
- Nighttime DO crash without aeration (≥ 1.5 mg/L drop in 6 hours)
- WSSV tray consumption drops before mortality (pre-clinical early warning)
- Emergency harvest timing: 60–80% save within 3 days, <50% after day 3
- Antibiotic flag persists for entire episode
- Deterministic reset with seed
- Reward always in [−1.0, +1.0]
- Reward weights sum to 1.0 per task
- Grade in [0.0, 1.0] every step

---

## File Structure

```
aquashrimp/
├── openenv.yaml                     # OpenEnv manifest
├── Dockerfile                       # HF Spaces deployment
├── pyproject.toml
├── scripts/
│   └── benchmark_all.py             # Reproducible full benchmark
├── aquashrimp/
│   ├── models/
│   │   ├── enums.py                # TreatmentType, HarvestType, WeatherEvent
│   │   ├── actions.py              # Action dataclasses (all 3 tasks)
│   │   ├── observations.py         # Observation dataclasses
│   │   └── state.py                # Internal simulation state
│   ├── simulation/
│   │   ├── shrimp_growth.py        # DWG model, molting, FCR, stress
│   │   ├── water_quality.py        # DO/TAN/pH/salinity/alkalinity/H2S
│   │   ├── feeding_trays.py        # Partial observability mechanic
│   │   ├── weather.py              # Temperature, rainfall, heat waves
│   │   ├── disease.py              # WSSV + Vibrio models
│   │   └── market.py               # Price random walk, cost tables
│   ├── tasks/
│   │   ├── nursery_pond.py         # Task 1 environment
│   │   ├── semi_intensive_farm.py  # Task 2 environment
│   │   └── commercial_grow_out.py  # Task 3 environment
│   ├── rewards/
│   │   └── reward_calculator.py    # 4-component reward + grade
│   ├── baselines/
│   │   ├── random_agent.py
│   │   ├── rule_based_agent.py
│   │   ├── optimal_agent.py
│   │   └── run_baseline.py         # CLI single-combo runner
│   └── server/
│       ├── app.py                  # FastAPI app factory
│       └── router.py               # /reset /step /state /health /grade
└── tests/                          # 92 tests, 7 files
```

---

## Citation

```
AquaShrimp: A Shrimp Aquaculture Operations OpenEnv Environment for AI Agent Training
Species: Litopenaeus vannamei (Whiteleg shrimp)
Version: 1.0.0
```

*First OpenEnv environment for shrimp/prawn aquaculture. Targeting the $40B/year global shrimp industry.*
