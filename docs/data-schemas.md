# Data Schemas & Service Specifications

## Overview

This document details all data structures, schemas, API contracts, and data volume specifications for the Metro Train Scheduling Service.

---

## Table of Contents

1. [Core Data Models](#core-data-models)
2. [API Schemas](#api-schemas)
3. [Database Schemas](#database-schemas)
4. [Data Volume & Storage](#data-volume--storage)
5. [Service Resource Usage](#service-resource-usage)

---

## Core Data Models

All models use **Pydantic v2** for validation and serialization.

### 1. DaySchedule

**Purpose**: Complete daily schedule with all trainset assignments

```python
class DaySchedule(BaseModel):
    schedule_id: str                    # "KMRL-2025-10-25"
    date: str                           # "2025-10-25"
    route: Route                        # Route details
    trainsets: List[Trainset]           # All train assignments
    fleet_summary: FleetSummary         # Fleet statistics
    optimization_metrics: OptimizationMetrics
    alerts: List[Alert]                 # Warnings/issues
    generated_at: datetime
    generated_by: str = "ML-Optimizer"
```

**Size**: ~45 KB per schedule (30 trains, full day)

**Example**:
```json
{
  "schedule_id": "KMRL-2025-10-25",
  "date": "2025-10-25",
  "route": {...},
  "trainsets": [...],
  "fleet_summary": {
    "total_trainsets": 30,
    "in_service": 24,
    "standby": 4,
    "maintenance": 2
  },
  "optimization_metrics": {
    "total_service_blocks": 156,
    "avg_readiness_score": 0.87,
    "mileage_variance_coefficient": 0.12
  },
  "generated_at": "2025-10-25T04:30:00+05:30"
}
```

---

### 2. Trainset

**Purpose**: Individual train assignment and status

```python
class Trainset(BaseModel):
    trainset_id: str                    # "TS-001"
    status: TrainHealthStatus           # REVENUE_SERVICE, STANDBY, etc.
    depot_bay: str                      # "BAY-01"
    cumulative_km: int                  # 145250
    readiness_score: float              # 0.0-1.0
    service_blocks: List[ServiceBlock]  # Trip assignments
    fitness_certificates: FitnessCertificates
    job_cards: JobCards
    branding: Branding
```

**Size**: ~1.5 KB per trainset

**Status Enum**:
```python
class TrainHealthStatus(str, Enum):
    REVENUE_SERVICE = "REVENUE_SERVICE"  # Active service
    STANDBY = "STANDBY"                  # Ready, not assigned
    MAINTENANCE = "MAINTENANCE"          # Under repair
    SCHEDULED_MAINTENANCE = "SCHEDULED_MAINTENANCE"
    UNAVAILABLE = "UNAVAILABLE"          # Out of service
```

**Distribution** (typical 30-train fleet):
- REVENUE_SERVICE: 22-24 trains (73-80%)
- STANDBY: 3-5 trains (10-17%)
- MAINTENANCE: 1-3 trains (3-10%)
- UNAVAILABLE: 0-2 trains (0-7%)

---

### 3. ServiceBlock

**Purpose**: Single trip assignment for a train

```python
class ServiceBlock(BaseModel):
    block_id: str                       # "BLK-001-01"
    start_time: str                     # "05:00"
    end_time: str                       # "05:45"
    start_station: str                  # "Aluva"
    end_station: str                    # "Pettah"
    direction: str                      # "UP" or "DOWN"
    distance_km: float                  # 25.612
    estimated_passengers: Optional[int] # 450
    priority: str = "NORMAL"            # NORMAL, HIGH, PEAK
```

**Size**: ~250 bytes per service block

**Daily Trips per Train**: 
- Peak service train: 6-8 trips
- Standard service: 4-6 trips
- Average: ~5.2 trips per active train

**Total Service Blocks** (30-train fleet):
- 24 active trains × 5.2 trips = ~125 service blocks/day

---

### 4. Route

**Purpose**: Metro line configuration

```python
class Route(BaseModel):
    route_id: str                       # "KMRL-LINE-01"
    name: str                           # "Aluva-Pettah Line"
    stations: List[Station]             # 25 stations
    total_distance_km: float            # 25.612 km
    avg_speed_kmh: int                  # 32-38 km/h
    turnaround_time_minutes: int        # 8-12 minutes
```

**KMRL Route Details**:
- **Stations**: 25 (Aluva to Pettah)
- **Distance**: 25.612 km
- **Average Speed**: 35 km/h
- **One-way Time**: ~44 minutes
- **Round Trip**: ~100 minutes (including turnarounds)

---

### 5. Station

**Purpose**: Individual station on route

```python
class Station(BaseModel):
    station_id: str                     # "STN-001"
    name: str                           # "Aluva"
    code: str                           # "ALV"
    distance_from_start_km: float       # 0.0
    platform_count: int                 # 2
    facilities: List[str]               # ["PARKING", "ELEVATOR"]
```

**Size**: ~200 bytes per station

**Total Stations**: 25 (fixed)

---

### 6. FitnessCertificates

**Purpose**: Regulatory compliance tracking

```python
class FitnessCertificates(BaseModel):
    rolling_stock: FitnessCertificate   # Train body/chassis
    signalling: FitnessCertificate      # Signal systems
    telecom: FitnessCertificate         # Communication systems

class FitnessCertificate(BaseModel):
    valid_until: str                    # "2025-12-31"
    status: CertificateStatus           # VALID, EXPIRING_SOON, EXPIRED

class CertificateStatus(str, Enum):
    VALID = "VALID"                     # > 30 days remaining
    EXPIRING_SOON = "EXPIRING_SOON"     # 7-30 days remaining
    EXPIRED = "EXPIRED"                 # Past expiry date
```

**Validation Rules**:
- Trains with EXPIRED certificates: status = UNAVAILABLE
- Trains with EXPIRING_SOON: flagged in alerts, can operate

---

### 7. JobCards & Maintenance

**Purpose**: Maintenance tracking

```python
class JobCards(BaseModel):
    open: int                           # Number of open job cards
    blocking: List[str]                 # Critical issues: ["BRAKE_FAULT"]

# Example maintenance reasons
UNAVAILABLE_REASONS = [
    "SCHEDULED_MAINTENANCE",
    "BRAKE_SYSTEM_REPAIR",
    "HVAC_REPLACEMENT",
    "BOGIE_OVERHAUL",
    "ELECTRICAL_FAULT",
    "ACCIDENT_DAMAGE",
    "PANTOGRAPH_REPAIR",
    "DOOR_SYSTEM_FAULT"
]
```

**Impact on Scheduling**:
- 0 open cards: readiness = 1.0
- 1-2 cards: readiness = 0.9
- 3-4 cards: readiness = 0.7
- 5+ cards: readiness = 0.5, likely maintenance status

---

### 8. Branding

**Purpose**: Advertisement tracking

```python
class Branding(BaseModel):
    advertiser: str                     # "COCACOLA-2024"
    contract_hours_remaining: int       # 450 hours
    exposure_priority: str              # LOW, MEDIUM, HIGH, CRITICAL

# Available advertisers
ADVERTISERS = [
    "COCACOLA-2024",
    "FLIPKART-FESTIVE",
    "AMAZON-PRIME",
    "RELIANCE-JIO",
    "TATA-MOTORS",
    "SAMSUNG-GALAXY",
    "NONE"
]
```

**Priority Weights** (for optimization):
- CRITICAL: 4 points
- HIGH: 3 points
- MEDIUM: 2 points
- LOW: 1 point
- NONE: 0 points

**Scheduling Strategy**:
- HIGH/CRITICAL branded trains prioritized for peak hours
- Maximizes advertiser visibility during high-traffic periods

---

### 9. FleetSummary

**Purpose**: Aggregated fleet statistics

```python
class FleetSummary(BaseModel):
    total_trainsets: int                # 30
    in_service: int                     # 24
    standby: int                        # 4
    maintenance: int                    # 2
    unavailable: int                    # 0
    availability_percent: float         # 93.33
    total_mileage_today: int           # 3200 km
    avg_trips_per_train: float         # 5.2
```

**Size**: ~300 bytes

**Key Metrics**:
- **Availability %**: (in_service + standby) / total × 100
- **Target Availability**: ≥ 90%
- **Service Ratio**: in_service / (in_service + standby)
- **Target Service Ratio**: 85-90%

---

### 10. OptimizationMetrics

**Purpose**: Optimization quality measures

```python
class OptimizationMetrics(BaseModel):
    total_service_blocks: int           # 125
    avg_readiness_score: float          # 0.87
    mileage_variance_coefficient: float # 0.12
    branding_sla_compliance: float      # 0.95
    fitness_expiry_violations: int      # 0
    execution_time_ms: int              # 1250
    algorithm_used: str                 # "ensemble_ml" or "or_tools"
    confidence_score: Optional[float]   # 0.89 (if ML used)
```

**Size**: ~250 bytes

**Quality Thresholds**:
- avg_readiness_score: ≥ 0.80
- mileage_variance_coefficient: < 0.15
- branding_sla_compliance: ≥ 0.90
- fitness_expiry_violations: 0

---

## API Schemas

### Request: ScheduleRequest

**Endpoint**: `POST /api/v1/generate`

```python
class ScheduleRequest(BaseModel):
    date: str                           # "2025-10-25"
    num_trains: int = 25                # 25-40
    num_stations: int = 25              # Fixed for KMRL
    min_service_trains: int = 22        # Minimum active
    min_standby_trains: int = 3         # Minimum backup
    
    # Optional overrides
    peak_hours: Optional[List[int]] = None  # [7,8,9,17,18,19]
    force_optimization: bool = False    # Skip ML, use OR-Tools
```

**Size**: ~150 bytes per request

**Validation**:
- `num_trains`: 25 ≤ n ≤ 40
- `num_stations`: Fixed at 25 (KMRL specific)
- `min_service_trains`: ≤ num_trains - 3
- `min_standby_trains`: ≥ 2

**Example**:
```json
{
  "date": "2025-10-25",
  "num_trains": 30,
  "num_stations": 25,
  "min_service_trains": 24,
  "min_standby_trains": 4
}
```

---

### Response: DaySchedule

**Status**: 200 OK

**Content-Type**: application/json

**Size**: 45-55 KB (depends on fleet size)

**Headers**:
```
X-Algorithm-Used: ensemble_ml | or_tools | greedy
X-Confidence-Score: 0.89 (if ML)
X-Execution-Time-Ms: 1250
```

---

### Error Responses

**400 Bad Request**:
```json
{
  "error": "Validation Error",
  "details": {
    "num_trains": "Must be between 25 and 40"
  }
}
```

**500 Internal Server Error**:
```json
{
  "error": "Optimization Failed",
  "message": "Unable to find feasible schedule",
  "timestamp": "2025-10-25T10:30:00Z"
}
```

---

## Database Schemas

### Schedule Storage (JSON Files)

**Location**: `data/schedules/`

**Naming**: `{schedule_id}_{timestamp}.json`

**Example**: `KMRL-2025-10-25_20251025_043000.json`

**Structure**:
```json
{
  "schedule": {DaySchedule},
  "metadata": {
    "recorded_at": "2025-10-25T04:30:00",
    "quality_score": 87.5,
    "algorithm_used": "ensemble_ml",
    "confidence": 0.89
  },
  "saved_at": "2025-10-25T04:30:15"
}
```

**Size per File**: ~48 KB

---

### Model Storage (Pickle Files)

**Location**: `models/`

**Files**:
1. `models_latest.pkl` - Current ensemble (all 5 models)
2. `models_{timestamp}.pkl` - Historical snapshots
3. `training_history.json` - Training metrics log

**Model File Contents**:
```python
{
    "models": {
        "gradient_boosting": GradientBoostingRegressor(),
        "random_forest": RandomForestRegressor(),
        "xgboost": XGBRegressor(),
        "lightgbm": LGBMRegressor(),
        "catboost": CatBoostRegressor()
    },
    "ensemble_weights": {
        "xgboost": 0.215,
        "lightgbm": 0.208,
        ...
    },
    "best_model_name": "xgboost",
    "last_trained": datetime(2025, 10, 25, 4, 30),
    "config": {
        "version": "v1.0.0",
        "features": [...],
        "models_trained": [...]
    }
}
```

**Size**: ~15-25 MB (all 5 models combined)

---

### Training History (JSON)

**Location**: `models/training_history.json`

**Structure**:
```json
[
  {
    "timestamp": "2025-10-23T12:00:00",
    "metrics": {
      "gradient_boosting": {
        "train_r2": 0.8912,
        "test_r2": 0.8234,
        "test_rmse": 13.45
      },
      ...
    },
    "best_model": "xgboost",
    "ensemble_weights": {...},
    "config": {
      "models_trained": [...],
      "version": "v1.0.0"
    }
  },
  ...
]
```

**Growth**: ~1 KB per training run

**Retention**: All training runs (pruned after 1000 entries)

---

## Data Volume & Storage

### Production Estimates

#### Daily Operations

**Per Day** (single schedule generation):
- 1 schedule file: ~48 KB
- API request/response: ~50 KB total
- Logs: ~10 KB

**Total per day**: ~108 KB

#### Monthly Operations (30 days)

**Schedule files**: 
- 30 schedules × 48 KB = 1.44 MB

**Model files**:
- 1 retraining (every 48 hours) = 15 retrainings/month
- 15 × 25 MB = 375 MB

**Training history**:
- 15 entries × 1 KB = 15 KB

**Total per month**: ~377 MB

#### Annual Storage (1 year)

**Schedule data**: 
- 365 schedules × 48 KB = 17.5 MB

**Model snapshots**:
- 182 retrainings × 25 MB = 4.55 GB

**Training history**: 
- 182 KB

**Total per year**: ~4.57 GB

**With retention policy** (keep last 100 schedules, 50 models):
- Schedules: 100 × 48 KB = 4.8 MB
- Models: 50 × 25 MB = 1.25 GB
- History: 182 KB

**Total with retention**: ~1.26 GB

---

### ML Training Data Requirements

#### Minimum Training Dataset

**Initial training**: 100 schedules
- Storage: 100 × 48 KB = 4.8 MB
- Generation time: ~15 minutes (automated)
- Training time: 5-10 minutes

**Optimal training**: 500 schedules
- Storage: 500 × 48 KB = 24 MB
- Provides better generalization
- Covers more edge cases

#### Feature Matrix Size

**Per schedule**: 10 features × 8 bytes (float64) = 80 bytes

**Training set** (100 schedules):
- Features (X): 100 × 80 bytes = 8 KB
- Target (y): 100 × 8 bytes = 800 bytes
- Total: ~9 KB (minimal)

**Full dataset** (1000 schedules):
- Features: 80 KB
- Target: 8 KB
- Total: ~88 KB

**Memory during training**:
- Dataset: ~88 KB
- Models (5 × ~5 MB): ~25 MB
- Working memory: ~50 MB
- **Total**: ~75 MB

---

### Optimization Service Resource Usage

#### OR-Tools Optimization

**Input data**:
- 30 trains × 1.5 KB = 45 KB
- 25 stations × 200 bytes = 5 KB
- Constraints: ~10 KB
- **Total input**: ~60 KB

**Memory usage**:
- Solver state: ~10 MB
- Solution space: ~20 MB
- **Peak memory**: ~30 MB

**Execution time**: 1-5 seconds (CPU-bound)

**CPU utilization**: 100% single core

---

#### ML Ensemble Prediction

**Input data**:
- Feature vector: 10 × 8 bytes = 80 bytes
- **Total input**: < 1 KB

**Memory usage**:
- Loaded models: ~25 MB (shared)
- Prediction workspace: ~1 MB
- **Peak memory**: ~26 MB

**Execution time**: 50-100 milliseconds

**CPU utilization**: 20-30% single core

---

#### Greedy Optimization

**Input data**: ~60 KB (same as OR-Tools)

**Memory usage**:
- State tracking: ~5 MB
- Priority queue: ~2 MB
- **Peak memory**: ~7 MB

**Execution time**: < 1 second

**CPU utilization**: 50-70% single core

---

## Service Resource Usage

### DataService (FastAPI)

**Base memory**: 150 MB (Python + FastAPI + dependencies)

**Per request overhead**: ~10 MB

**Concurrent requests** (typical): 1-5

**Total memory** (under load): 200-250 MB

**Disk I/O**:
- Read: Minimal (configuration only)
- Write: ~50 KB per schedule generated

**Network**: 
- Inbound: ~150 bytes (request)
- Outbound: ~50 KB (response)

---

### SelfTrainService

**Base memory**: 200 MB (Python + ML libraries)

**During training**:
- Dataset loading: +20 MB
- Model training: +100 MB (peak)
- **Total during training**: ~320 MB

**During inference** (loaded models):
- Models in memory: +25 MB
- **Total during inference**: ~225 MB

**Disk I/O**:
- Read: 5 MB (load schedules)
- Write: 25 MB (save models)

**Frequency**:
- Training: Every 48 hours
- Inference: Per schedule request (if confidence ≥ 75%)

---

### Retraining Service (Background)

**Memory**: ~50 MB (idle), ~320 MB (during training)

**CPU**: 
- Idle: < 1%
- Training: 100% (5-10 minutes every 48 hours)

**Disk I/O**:
- Check interval: Every 60 minutes
- Read: ~1 MB (check schedule count)
- Write: ~25 MB (when retraining)

---

## Data Flow Summary

### Schedule Generation Request

```
Client Request (150 bytes)
    ↓
FastAPI Parser (~1 KB in memory)
    ↓
Feature Extraction (80 bytes)
    ↓
ML Prediction (25 MB models loaded) OR OR-Tools (30 MB solver)
    ↓
Schedule Generation (45 KB output)
    ↓
JSON Serialization (~50 KB response)
    ↓
Storage (48 KB file)
```

**Total data processed**: ~50 KB per request

**Response time**: 0.1-5 seconds

---

### Model Training Cycle

```
Load Schedules (100 × 48 KB = 4.8 MB)
    ↓
Extract Features (100 × 80 bytes = 8 KB)
    ↓
Train 5 Models (5-10 minutes, 100% CPU)
    ↓
Save Models (25 MB pickle file)
    ↓
Update History (1 KB append)
```

**Total data processed**: ~30 MB

**Frequency**: Every 48 hours

---

## Configuration Data

### Service Configuration

**Location**: `SelfTrainService/config.py`

**Size**: ~5 KB

**Key Parameters**:
```python
{
  "RETRAIN_INTERVAL_HOURS": 48,
  "MIN_SCHEDULES_FOR_TRAINING": 100,
  "MODEL_TYPES": ["gradient_boosting", "xgboost", ...],
  "USE_ENSEMBLE": true,
  "ML_CONFIDENCE_THRESHOLD": 0.75,
  "FEATURES": [10 feature names],
  "EPOCHS": 100,
  "LEARNING_RATE": 0.001
}
```

---

## Data Retention Policies

### Recommended Retention

**Schedule files**:
- Keep last 365 days (17.5 MB)
- Archive older to compressed storage

**Model snapshots**:
- Keep last 50 models (~1.25 GB)
- Delete older snapshots
- Keep 1 model per month for historical reference

**Training history**:
- Keep all entries (grows slowly)
- Compress after 1000 entries

**Logs**:
- Application logs: 30 days
- Error logs: 90 days
- Audit logs: 1 year

---

## Scaling Considerations

### Horizontal Scaling

**API Service** (DataService):
- Stateless - easy to scale
- Load balancer distributes requests
- Each instance: ~250 MB memory

**ML Service** (SelfTrainService):
- Share model files via NFS/S3
- Only one instance should train (avoid conflicts)
- Multiple instances can serve predictions

### Vertical Scaling

**Memory requirements**:
- Minimum: 1 GB RAM
- Recommended: 2 GB RAM
- Optimal: 4 GB RAM (allows concurrent training + serving)

**CPU requirements**:
- Minimum: 1 core
- Recommended: 2 cores (1 for API, 1 for training)
- Optimal: 4 cores (parallel model training)

**Storage requirements**:
- Minimum: 5 GB
- Recommended: 20 GB
- Optimal: 50 GB (1-year retention)

---

## Performance Benchmarks

### Schedule Generation Performance

| Fleet Size | Algorithm | Time | Memory | Output Size |
|------------|-----------|------|--------|-------------|
| 25 trains  | ML        | 0.08s | 225 MB | 38 KB |
| 30 trains  | ML        | 0.10s | 225 MB | 45 KB |
| 40 trains  | ML        | 0.12s | 225 MB | 60 KB |
| 25 trains  | OR-Tools  | 1.2s  | 30 MB  | 38 KB |
| 30 trains  | OR-Tools  | 2.8s  | 30 MB  | 45 KB |
| 40 trains  | OR-Tools  | 4.5s  | 30 MB  | 60 KB |
| 25 trains  | Greedy    | 0.3s  | 7 MB   | 38 KB |
| 30 trains  | Greedy    | 0.5s  | 7 MB   | 45 KB |
| 40 trains  | Greedy    | 0.8s  | 7 MB   | 60 KB |

### Training Performance

| Dataset Size | Training Time | Memory | Model Size |
|--------------|---------------|--------|------------|
| 100 schedules | 3 min       | 320 MB | 20 MB |
| 500 schedules | 8 min       | 350 MB | 24 MB |
| 1000 schedules | 15 min     | 400 MB | 28 MB |

---

**Document Version**: 1.0.0  
**Last Updated**: November 2, 2025  
**Maintained By**: ML-Service Team
